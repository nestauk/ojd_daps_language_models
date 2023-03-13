"""
Flow to fine tune a BERT-like model on job advert sentences for domain adaptation
for the purpose of next sentence prediction and masked language modelling.

NOTE: make sure you're logged into huggingface with `huggingface-cli login`
to upload the fine-tuned model to huggingface before running this flow.

python ojd_daps_language_models/pipeline/train_model/ojobert_flow.py.py run
"""
from ojd_daps_language_models import logger, get_yaml_config, PROJECT_DIR, BUCKET_NAME
import ojd_daps_language_models.utils.bert_training as bt

import os
from metaflow import FlowSpec, step, Parameter, batch

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements_ojobert.txt 1> /dev/null"
)

CONFIG = get_yaml_config(PROJECT_DIR / "ojd_daps_language_models/config/training.yaml")


class OjoBertFlow(FlowSpec):
    """
    Fine tune a masked DistilBERT model on job advert sentences for domain adaptation

    The workflow performs the following steps:
    1) Load training data from S3 and convert it to a Dataset object
    2) Tokenize the data
    3) Fine-tune DistilBERT on the train split and using test split for evaluation
    """

    production = Parameter(
        "production", help="to run in production mode", default=False
    )
    checkpoint = Parameter(
        "checkpoint",
        help="the name of the model checkout",
        default=CONFIG["checkpoint"],
    )
    job_sent_path = Parameter(
        "job_sent_path",
        help="the S3 path to job sentences file",
        default=CONFIG["job_sent_path"],
    )
    mlm_probability = Parameter("mlm_probability", default=CONFIG["mlm_probability"])
    train_size = Parameter("train_size", default=CONFIG["bert_train_size"])
    batch_size = Parameter("batch_size", default=CONFIG["bert_batch_size"])

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.next(self.get_training_data)

    @step
    def get_training_data(self):
        """Loads job sentences from S3"""
        from ojd_daps_language_models.getters.data_getters import get_s3_data_paths, s3
        from nesta_ds_utils.loading_saving.S3 import download_obj
        from itertools import chain
        import pandas as pd
        from datasets import Dataset

        sent_files = get_s3_data_paths(
            s3, bucket_name=BUCKET_NAME, root=self.job_sent_path, file_types=["*.json"]
        )

        sents = []
        for sent_file in sent_files:
            file = download_obj(BUCKET_NAME, sent_file, download_as="dict")
            sents.extend(list(chain(*file)))

        job_sents = list(set(sents))[1:]
        # convert into huggingface Dataset to use multi-threading
        self.job_sents_dataset = Dataset.from_pandas(pd.DataFrame({"text": job_sents}))

        logger.info(f"Loaded {len(job_sents)} job sentences from S3.")

        self.next(self.prepare_training_data)

    @step
    def prepare_training_data(self):
        """Prepare training data for BERT-like model by:
        1) Tokenizing the sentences
        2) Grouping the sentences into batches
        3) Masking some of the words in the sentences
        """
        from transformers import DataCollatorForLanguageModeling

        self.tokenized_dataset = self.job_sents_dataset.map(
            bt.tokenize_function, batched=True, remove_columns=["text"]
        )
        logger.info("Tokenized sentences.")

        self.lm_datasets = self.tokenized_dataset.map(bt.group_texts, batched=True)
        logger.info("grouped sentences into batches.")

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=bt.tokenizer, mlm_probability=self.mlm_probability
        )
        logger.info("prepared training data.")

        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        """Split the data into train and test sets"""
        if self.production:
            test_size = int(0.01 * self.train_size)
        else:
            test_size = int(0.1 * 10)

        self.downsampled_dataset = self.lm_datasets.train_test_split(
            train_size=self.train_size, test_size=test_size, seed=42
        )
        logger.info("Split data into train and test sets.")

        self.next(self.train_bert_model)

    @batch(gpu=1)  # I think this is all you need to do
    @step
    def train_bert_model(self):
        """Train a BERT-like model"""
        from transformers import TrainingArguments, Trainer
        from nesta_ds_utils.loading_saving.S3 import upload_obj
        import math
        import os
        from datetime import datetime

        logging_steps = len(self.downsampled_dataset["train"]) // self.batch_size
        model_name = self.checkpoint.split("/")[-1]
        output_dir = f"{model_name}-finetuned-ojo"

        push_to_hub = True if self.production else False

        training_args = TrainingArguments(
            output_dir=output_dir,  # output to huggingface
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            push_to_hub=push_to_hub,
            logging_steps=logging_steps,
        )

        trainer = Trainer(
            model=bt.bert_model,
            args=training_args,
            train_dataset=self.downsampled_dataset["train"],
            eval_dataset=self.downsampled_dataset["test"],
            data_collator=self.data_collator,
            tokenizer=bt.tokenizer,
        )
        logger.info("Training BERT-like model...")

        trainer.train()

        logger.info("...finished training!")

        logger.info("evaluate model...")

        eval_results = trainer.evaluate()

        eval_results["perplexity"] = round(math.exp(eval_results["eval_loss"]), 2)
        model_eval_path = "models/model_evaluation/"
        date = datetime.now().strftime("%Y-%m-%d")
        upload_obj(
            obj=eval_results,
            bucket=BUCKET_NAME,
            path_to=os.path.join(
                model_eval_path, f"{output_dir}_{date}_evaluation_results.json"
            ),
        )

        logger.info("pushing to huggingface hub...")

        trainer.push_to_hub()

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    OjoBertFlow()
