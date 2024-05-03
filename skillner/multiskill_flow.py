"""
Flow to train a classifier to predict if skill spans
    are multiskills or not. 

python multiskill_flow.py --package-suffixes=.txt run
"""
import os
from pathlib import Path

os.system(f"pip install -r {Path.cwd()}/multiskill_requirements.txt 1> /dev/null")
import boto3
from dotenv import load_dotenv
from metaflow import FlowSpec, Parameter, step
from wasabi import msg
from sklearn.base import BaseEstimator, TransformerMixin

from utils import config

load_dotenv()

class MultiSkillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.transform_skill(skill) for skill in X]

    @staticmethod
    def transform_skill(skill: str):
        return [len(skill), int(" and " in skill), int("," in skill)]


class MultiSkillFlow(FlowSpec):
    """
    Train a Support Vector Machine (svm) to predict if
        a skill span is a multiskill (1) or not (0).
    """

    production = Parameter(
        "production", help="to run in production mode", default=False
    )
    hf_push = Parameter("hf_push", help="push model to huggingface?", default=False)

    @step
    def start(self):
        """
        Starts the flow.
        """
        msg.info("Starting flow to train MultiSkill classifier...")

        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Loads labelled data from s3.
        """
        import json

        s3 = boto3.resource("s3")
        obj = s3.Object(config.data.bucket_name, str(config.data.data_path))
        self.data = json.loads(obj.get()["Body"].read().decode("utf-8"))

        if not self.production:
            job_ids_sample = list(self.data.keys())[:10]
            self.data = {k: v for k, v in self.data.items() if k in job_ids_sample}

        msg.good(f"Loaded {len(self.data)} labelled job adverts successfully.")

        self.next(self.process_data)

    @step
    def process_data(self):
        """
        Process the labelled data.
        """
        from utils import _process_data

        self.skills_list = []
        self.multiskills_list = []
        for labelled_data in self.data.values():
            text, ent_list = _process_data(labelled_data, config.data.all_labels)
            for start, end, label in ent_list:
                if label == "SKILL":
                    self.skills_list.append(text[start:end])
                elif label == "MULTISKILL":
                    self.multiskills_list.append(text[start:end])

        self.next(self.split_data)

    @step
    def split_data(self):
        """
        Split the data into training and test sets.
        """
        from sklearn.model_selection import train_test_split
        import random

        random.seed(config.train.random_seed)
        random.shuffle(self.skills_list)

        # balance dataset
        self.X = self.multiskills_list + self.skills_list[: len(self.multiskills_list)]
        self.y = [1] * len(self.multiskills_list) + [0] * len(
            self.skills_list[: len(self.multiskills_list)]
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=config.train.test_size,
            random_state=config.train.random_seed,
        )

        msg.info(f"train size: {len(self.X_train)} test size: {len(self.X_test)}")

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train the SVM model.
        """
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline(
            [
                ("transformer", MultiSkillTransformer()),
                (
                    "clf",
                    SVC(
                        kernel=config.train.kernel,
                        C=1,
                        class_weight=config.train.class_weight,
                    ),
                ),
            ]
        )

        self.pipeline.fit(self.X_train, self.y_train)

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        evaluate the SVM model.
        """
        from sklearn.metrics import classification_report

        y_pred = self.pipeline.predict(self.X_test)
        self.evaluation_metrics = classification_report(
            self.y_test, y_pred, target_names=["SKILL", "MULTISKILL"], output_dict=True
        )

        msg.info(
            f"MULTISKILL precision: {self.evaluation_metrics['MULTISKILL']['precision']}"
        )
        msg.info(f"SKILL precision: {self.evaluation_metrics['SKILL']['precision']}")

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        msg.good("MultiSkill Classifier trained successfully.")

        if self.hf_push:
            import pickle
            from pathlib import Path
            from tempfile import mkdtemp, mkstemp

            import pandas as pd
            import sklearn
            from skops import card, hub_utils

            _, pkl_name = mkstemp(prefix=config.hf.ms_model_name, suffix=".pkl")

            with open(pkl_name, mode="bw") as f:
                pickle.dump(self.pipeline['clf'], file=f)

            local_repo = mkdtemp(prefix="skops-")

            hub_utils.init(
                model=pkl_name,
                requirements=[f"scikit-learn={sklearn.__version__}"],
                dst=local_repo,
                task="text-classification",
                data=self.X,
            )
            model_card = card.Card(
                self.pipeline, metadata=card.metadata_from_config(Path(local_repo))
            )
            model_card.add(
                model_description="Support Vector Machine (SVM) trained to predict if a skill span is a multiskill or not."
            )
            clf_report = pd.DataFrame(self.evaluation_metrics).T.reset_index()
            model_card.add_table(
                folded=True,
                **{
                    "Classification Report": clf_report,
                },
            )
            model_card.metadata.license = "mit"
            model_card.save(Path(local_repo) / "README.md")

            token = os.environ.get("HF_TOKEN")
            assert (
                token is not None
            ), "Please set HF_TOKEN in your environment variables."

            hub_utils.push(
                repo_id=f"{config.hf.namespace}/{config.hf.ms_model_name}",
                source=local_repo,
                token=token,
                commit_message="pushing model files to huggingface",
                create_remote=True,
            )
            msg.good("Model pushed to huggingface successfully.")


if __name__ == "__main__":
    MultiSkillFlow()
