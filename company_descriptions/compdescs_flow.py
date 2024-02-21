"""
Flow to fine tune a Sequence Classification head of jobbert to
binarily classify sentences as a company description or not.

python compdescs_flow.py --package-suffixes=.txt,.yaml --datastore=s3 run
"""

import os

os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/requirements.txt 1> /dev/null"
)

from metaflow import FlowSpec, step, Parameter, batch

from compdescs_flow_utils import tokenizer, model, preprocess_function

from math import exp
import pandas as pd

import yaml

import wandb
from wandb.integration.metaflow import wandb_log

wandb.login()

# read config from training.yaml file
CONFIG = yaml.safe_load(open("training.yaml"))


@wandb_log(datasets=False, models=False)
class CompDescFlow(FlowSpec):
    """
    Fine tune a Sequence Classification head to binarily classify
    sentences as a company description or not.

    The workflow performs the following steps:
    """

    production = Parameter(
        "production", help="to run in production mode", default=False
    )
    train_frac = Parameter(
        "train_frac", help="fraction of data to train on", default=0.9
    )

    @step
    def start(self):
        """
        Starts the flow.
        """
        self.tokenizer = tokenizer
        self.model = model

        self.next(self.get_labelled_data)

    @step
    def get_labelled_data(self):
        """Load labelled job sentences from S3"""

        # load pandas dataframe from s3
        training_data_path = os.path.join(
            "prinz-green-jobs",
            "outputs/data/labelled_job_adverts/company_desc_sic_labelled_sentences.csv"
        )
        self.training_data = pd.read_csv(f"s3://{training_data_path}")

        self.next(self.split_labelled_data)

    @step
    def split_labelled_data(self):
        """
        Split labelled data by:

        1) Deal with class embalance by undersampling
            the majority class
        2) Split into train and test sets
        3) Convert to Dataset object
        """
        from datasets import Dataset, DatasetDict

        sample = 1000 if self.production else 100

        non_comp_descs = (
            self.training_data.query("label == 0")
            .query(
                "sentence.str.len() < 250"
            )  # make sure you don't get really long sentences
            .sample(sample, random_state=69)
            .reset_index(drop=True)
        )

        training_data_equal = pd.concat(
            [non_comp_descs, self.training_data.query("label == 1")]
        ).reset_index(drop=True)

        print(
            f"training data has {len(training_data_equal.query('label == 1'))} company descriptions and {len(training_data_equal.query('label == 0'))} non company descriptions"
        )

        # split into train and test sets
        train = training_data_equal.drop(columns="job_id").sample(
            frac=self.train_frac, random_state=69
        )
        train_indx = train.index

        test = training_data_equal.drop(train_indx).drop(columns="job_id")

        # convert to Dataset object
        train_ds, test_ds = Dataset.from_pandas(train), Dataset.from_pandas(test)

        self.ds = DatasetDict({"train": train_ds, "test": test_ds})

        self.next(self.prepare_training_data)

    @step
    def prepare_training_data(self):
        """Prepare training data by:
        1) tokenizing
        2) collating into batches
        """
        from transformers import DataCollatorWithPadding

        self.tokenized_ds = self.ds.map(preprocess_function, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        self.next(self.train_model)

    @batch(gpu=1, memory=60000, cpu=6, queue="job-queue-GPU-nesta-metaflow")
    @wandb_log(
        learning_rate=True,
        weight_decay=True,
        num_train_epochs=True,
        per_device_train_batch_size=True,
        per_device_eval_batch_size=True,
        test_results=True,
        training_results=True,
        eval_results=True
    )
    @step
    def train_model(self):
        """
        Fine tune the model on the labelled data
        and evaluate it on a test set.
        """
        from transformers import TrainingArguments, Trainer
        from scipy.special import softmax
        import numpy as np
        from sklearn.metrics import classification_report

        print("defining training arguments...")
        self.output_dir = "jobbert-base-cased-compdecs"
        training_args = TrainingArguments(
            output_dir="jobbert-base-cased-compdecs",
            learning_rate=2e-5,
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
            num_train_epochs=CONFIG["num_train_epochs"],
            weight_decay=CONFIG["weight_decay"],
        )
        print("instantiating trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_ds["train"],
            eval_dataset=self.tokenized_ds["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("training...")
        trainer.train()

        trainer_results = trainer.evaluate()
        trainer_results["perplexity"] = round(exp(trainer_results["eval_loss"]), 2)

        for config_name, config_value in CONFIG.items():
            trainer_results[config_name] = config_value

        # convert logits to probabilities
        probs = softmax(trainer.predict(self.tokenized_ds["test"]).predictions, axis=1)
        # convert probabilities to predicted labels
        y_pred = np.argmax(probs, axis=-1)
        # get true labels
        y_true = self.tokenized_ds["test"]["label"]
        # calculate classification report
        test_results = classification_report(y_true, y_pred, output_dict=True)

        self.eval_results = {}
        self.eval_results["training_evaluation"] = trainer_results
        self.eval_results["test_set_evaluation"] = test_results

        self.next(self.end)

    @step
    def end(self):
        """Saves evaluation data to S3"""
        from datetime import datetime
        import io
        import json
        import boto3

        date = datetime.now().strftime("%Y-%m-%d").replace("-", "")
        self.eval_results["date"] = date

        obj = io.BytesIO(json.dumps(self.eval_results).encode("utf-8"))
        s3 = boto3.client("s3")
        s3.upload_fileobj(
            obj,
            "prinz-green-jobs",
            f"outputs/models/comp_desc_classifier/{CONFIG['model_name'].replace('-', '_')}_{date}_{self.production}.json",
        )


if __name__ == "__main__":
    CompDescFlow()
