"""
Flow to fine tune a BERT-like model on job advert sentences for domain adaptation
for the purpose of next sentence prediction and masked language modelling. We can use this model and fine
tune i.e. NER heads to a domain adapted BERT model for NER.

python ojd_daps_language_models/pipeline/train_model/ojobert/ojobert_flow.py --package-suffixes=.txt,.yaml --datastore=s3 run

"""
import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)

from metaflow import FlowSpec, step, Parameter, batch

from ojobert_flow_utils import (
    tokenizer,
    bert_model,
    tokenize_function,
    group_texts,
    CONFIG,
)

import logging
from datetime import datetime
from math import floor, exp
from fnmatch import fnmatch

from itertools import chain
import random
import json

import pandas as pd
from datasets import Dataset
import boto3
from transformers import TrainingArguments, Trainer

logger = logging.getLogger("dap-ojobert")


class OjoBertFlow(FlowSpec):
    """
    Fine tune a masked DistilBERT model on job advert sentences for domain adaptation

    The workflow performs the following steps:
    1) Load training data from S3 and convert it to a Dataset object
    2) Tokenize the data
    3) Fine-tune DistilBERT on the train split and using test split for evaluation
    """

    production = Parameter("production", help="to run in production mode", default=True)

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        pass
        self.next(self.get_training_data)

    @step
    def get_training_data(self):
        """Loads job sentences from S3"""
        s3 = boto3.resource("s3")
        bucket = s3.Bucket("dap-ojobert")
        all_sent_files = []
        for files in bucket.objects.filter(Prefix=CONFIG["job_sent_path"]):
            key = files.key
            if any([fnmatch(key, pattern) for pattern in ["*.json"]]):
                all_sent_files.append(key)

        random.seed(CONFIG["random_seed"])
        random.shuffle(all_sent_files)

        # load data
        sent_files = all_sent_files if self.production else all_sent_files[:1]

        logger.info(f"loading {len(sent_files)} sentence files...")

        sents = []
        for sent_file in sent_files:
            s3object = s3.Object("dap-ojobert", sent_file)
            file = json.loads(s3object.get()["Body"].read().decode("utf-8"))
            sents.extend(list(chain(*file)))

        job_sents = list(set(sents))[1:]  # remove empty string

        job_sents = (
            job_sents[: int(CONFIG["bert_train_size"])]
            if self.production
            else job_sents[:500]
        )

        logger.info(f"loaded {len(job_sents)} sentences...")

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
            tokenize_function, batched=True, remove_columns=["text"]
        )
        logger.info("Tokenized sentences.")

        self.lm_datasets = self.tokenized_dataset.map(group_texts, batched=True)
        logger.info("grouped sentences into batches.")

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=CONFIG["mlm_probability"]
        )
        logger.info("prepared training data.")

        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        """Split the data into train and test sets"""

        train_size = floor(self.lm_datasets.num_rows * 0.9)
        test_size = floor(self.lm_datasets.num_rows * 0.1)

        # to accomodate for testing if training data is v small
        test_size = 1 if test_size == 0 else test_size

        self.downsampled_dataset = self.lm_datasets.train_test_split(
            train_size=train_size,
            test_size=test_size,
            seed=CONFIG["random_seed"],
        )
        logger.info("Split data into train and test sets.")

        self.next(self.train_bert_model)

    @batch(gpu=1, memory=80000, cpu=8, queue="job-queue-GPU-nesta-metaflow")
    @step
    def train_bert_model(self):
        """Train a BERT-like model"""
        logging_steps = (
            len(self.downsampled_dataset["train"]) // CONFIG["bert_batch_size"]
            if self.production
            else 1
        )
        self.model_name = CONFIG["checkpoint"].split("/")[-1]
        self.output_dir = f"models/{self.model_name}-finetuned-ojo/"

        training_args = TrainingArguments(
            output_dir=f"outputs/{self.output_dir}",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=CONFIG["bert_batch_size"],
            per_device_eval_batch_size=CONFIG["bert_batch_size"],
            logging_steps=logging_steps,
        )

        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=self.downsampled_dataset["train"],
            eval_dataset=self.downsampled_dataset["test"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        logger.info("Training BERT-like model...")

        trainer.train()

        logger.info("evaluate model...")

        self.eval_results = trainer.evaluate()

        self.next(self.save_evaluation)

    @step
    def save_evaluation(self):
        """Save the model's evaluation results"""
        import io

        self.eval_results["perplexity"] = round(exp(self.eval_results["eval_loss"]), 2)
        date = datetime.now().strftime("%Y-%m-%d")

        obj = io.BytesIO(json.dumps(self.eval_results).encode("utf-8"))
        s3 = boto3.client("s3")
        s3.upload_fileobj(
            obj,
            "dap-ojobert",
            f"models/model_evaluation/{self.model_name.replace('-', '_')}_finetuned_ojo_{str(date).replace('-', '_')}_production_{str(self.production).lower()}_evaluation_results.json",
        )

        self.next(self.end)

    @step
    def end(self):
        """Ends the flow"""
        pass


if __name__ == "__main__":
    # run flow to train the model using batch
    OjoBertFlow()
