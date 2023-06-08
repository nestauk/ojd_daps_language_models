"""
Flow to fine tune a BERT-like model on job advert sentences for domain adaptation
for the purpose of next sentence prediction and masked language modelling. We can use this model and fine
tune i.e. NER heads to a domain adapted BERT model for NER.

if you're running locally (will take a long time and you should hash the @batch decorator):

python ojd_daps_language_models/pipeline/train_model/ojobert_flow.py run

if you're running on AWS:

python ojd_daps_language_models/pipeline/train_model/ojobert_flow.py --datastore=s3 run

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
    sents_per_file = Parameter("sents_per_file", default=CONFIG["sents_per_file"])
    seed = Parameter("random_seed", default=42)

    @step
    def start(self):
        """
        Starts the flow and sets parameters based on whether flow is in production or not.
        """
        self.batch_size = 64
        if self.production:
            self.train_size = 100000
            self.test_size = int(0.01 * self.train_size)
        else:
            self.train_size = 10
            self.test_size = int(0.1 * 10)

        self.next(self.get_training_data)

    @step
    def get_training_data(self):
        """Loads job sentences from S3"""
        from ojd_daps_language_models.getters.data_getters import (
            get_s3_data_paths,
            s3,
            load_s3_data,
        )
        from itertools import chain
        import pandas as pd
        from datasets import Dataset
        import random
        import math

        # we don't need all the sent files given we're not training the NN on all the data
        # so we can downsample the data to save time
        all_sent_files = get_s3_data_paths(
            s3, bucket_name=BUCKET_NAME, root=self.job_sent_path, file_types=["*.json"]
        )
        sent_files_num = math.ceil(self.train_size / self.sents_per_file)

        logger.info(
            f"loading {sent_files_num} files based on {self.train_size} sentences"
        )

        random.seed(self.seed)
        random.shuffle(all_sent_files)

        # load data
        sent_files = all_sent_files[:sent_files_num]

        sents = []
        for sent_file in sent_files:
            file = load_s3_data(BUCKET_NAME, sent_file)
            sents.extend(list(chain(*file)))

        job_sents = list(set(sents))[1:]  # remove empty string

        # convert into huggingface Dataset to use multi-threading
        self.job_sents_dataset = Dataset.from_pandas(pd.DataFrame({"text": job_sents}))

        logger.info(f"Loaded {len(job_sents)} job sentences from S3.")

        self.next(self.prepare_training_data)

    # make sure we have at least 100k job adverts in the training data
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
        self.downsampled_dataset = self.lm_datasets.train_test_split(
            train_size=round(len(self.lm_datasets) * 0.9),
            test_size=round(len(self.lm_datasets) * 0.1),
            seed=self.seed,
        )
        logger.info("Split data into train and test sets.")

        self.next(self.train_bert_model)

    @batch(gpu=1, memory=60000, cpu=8, queue="job-queue-GPU-nesta-metaflow")
    @step
    def train_bert_model(self):
        """Train a BERT-like model"""
        from transformers import TrainingArguments, Trainer
        from ojd_daps_language_models.getters.data_getters import save_to_s3
        import math
        import os
        from datetime import datetime

        logging_steps = (
            len(self.downsampled_dataset["train"]) // self.batch_size
            if self.production
            else 1
        )
        model_name = self.checkpoint.split("/")[-1]
        output_dir = PROJECT_DIR / f"outputs/models/{model_name}-finetuned-ojo/"

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
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
        save_to_s3(
            BUCKET_NAME,
            eval_results,
            os.path.join(
                model_eval_path, f"{output_dir}_{date}_evaluation_results.json"
            ),
        )

        # save model locally
        trainer.save_model(output_dir)

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    OjoBertFlow()
