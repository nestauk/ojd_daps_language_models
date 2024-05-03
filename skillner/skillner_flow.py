"""
Flow to train an NER model to extract skills from jobs.

python skillner_flow.py --package-suffixes=.txt run
"""
import os
from pathlib import Path

os.system(f"pip install -r {Path.cwd()}/skillner_requirements.txt 1> /dev/null")
import boto3
from dotenv import load_dotenv
from metaflow import FlowSpec, Parameter, step
from wasabi import msg

from utils import config

load_dotenv()


class SkillNerFlow(FlowSpec):
    """
    Train a Named Entity Recognition (NER) model
        to extract skills from job adverts.
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
        msg.info("Starting flow to train SkillNER model...")

        self.all_labels = [
            label for label in config.data.all_labels if label != "MULTISKILL"
        ]

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
        Process the labelled data. This step:
            - converts "MULTISKILL" labels to "SKILL"
            - creates a list of tuples for each labelled job advert
        """
        from utils import _process_data

        self.clean_data = []

        for _, label_data in self.data.items():
            text, ent_list = _process_data(label_data, config.data.all_labels)
            # let's convert the MULTISKILL label to SKILL
            ent_list = [
                (start, end, "SKILL" if label == "MULTISKILL" else label)
                for start, end, label in ent_list
            ]
            self.clean_data.append(
                (
                    text,
                    {"entities": ent_list},
                )
            )

        self.next(self.split_data)

    @step
    def split_data(self):
        """
        Split the data into training and testing sets.
        """
        import random

        train_n = round(len(self.clean_data) * config.train.train_prop)

        random.seed(config.train.random_seed)
        random.shuffle(self.clean_data)

        self.train_data = self.clean_data[0:train_n]
        self.test_data = self.clean_data[train_n:]

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        train the NER model
        """
        import spacy

        from utils import train_ner

        # load spacy model
        msg.info(f"Loading spacy model: {config.train.spacy_model}")
        self.nlp = spacy.load(config.train.spacy_model)

        msg.info("Training NER component...")
        train_ner(nlp=self.nlp, train_data=self.train_data)

        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        from spacy.training import offsets_to_biluo_tags

        from utils import evaluate_ner

        # get ground truth
        y_true, y_pred = [], []
        for text, entity_annotations in self.test_data:
            doc = self.nlp(text)
            # let's get the true tags from the labelled data
            true_tags = offsets_to_biluo_tags(doc, entity_annotations["entities"])
            # let's get the predicted tags from the model
            pred_tags = offsets_to_biluo_tags(
                doc, [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            )
            y_true.append(true_tags)
            y_pred.append(pred_tags)

        self.evaluation_results = evaluate_ner(
            y_true, y_pred, all_labels=self.all_labels
        )
        # Let's print an overall metric
        msg.info(
            f"Overall F1 Score: {self.evaluation_results['results_summary']['All']['f1']}"
        )

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        msg.good("Flow completed successfully.")

        if self.hf_push:
            import huggingface_hub
            from spacy_huggingface_hub import push
            import srsly

            output_path = Path.cwd() / "output"
            if not output_path.exists():
                output_path.mkdir()

            self.nlp.to_disk(Path.cwd() / config.hf.sn_model_name)

            ###create metadata here
            ents_per_type_dict = {}
            for l in self.all_labels:
                ents_per_type_dict[l] = self.evaluation_results["results_per_tag"][l][
                    "ent_type"
                ]

            metadata = {
                "lang": self.nlp.lang,
                "name": f"{self.nlp.lang}_{config.hf.sn_model_name}",
                "version": self.nlp.meta["version"],
                "description": "A Named Entity Recognition (NER) model to extract SKILL, EXPERIENCE and BENEFIT from job adverts.",
                "labels": {
                    "ner": [l for l in config.data.all_labels if l != "MULTISKILL"]
                },
                "pipeline": self.nlp.meta["pipeline"],
                "components": self.nlp.meta["components"],
                "disabled": self.nlp.meta["disabled"],
                "performance": {
                    "ents_p": self.evaluation_results["results_summary"]["All"][
                        "precision"
                    ],
                    "ents_r": self.evaluation_results["results_summary"]["All"][
                        "recall"
                    ],
                    "ents_f": self.evaluation_results["results_summary"]["All"]["f1"],
                    "ents_per_type": ents_per_type_dict,
                },
                "author": config.hf.namespace,
            }

            srsly.write_json(output_path / "metadata.json", metadata)

            os.system(
                f"python -m spacy package {Path.cwd() / config.hf.sn_model_name} {Path.cwd() / 'output'} --name {config.hf.sn_model_name} --meta-path {output_path / 'metadata.json'} --build wheel"
            )

            # get filename of the wheel file
            model_path = (
                f'output/en_{config.hf.sn_model_name}-{self.nlp.meta["version"]}/dist'
            )
            wheel_file = list((Path.cwd() / model_path).glob("*.whl"))[0]

            token = os.environ.get("HF_TOKEN")
            assert (
                token is not None
            ), "Please set HF_TOKEN in your environment variables."

            huggingface_hub.login(token=token, add_to_git_credential=True)
            push(wheel_file, namespace=config.hf.namespace)

            # delete the output folders
            os.system(f"rm -rf {output_path}")
            os.system(f"rm -rf {Path.cwd() / config.hf.sn_model_name}")
            msg.good("Model pushed to Hugging Face Hub successfully.")


if __name__ == "__main__":
    SkillNerFlow()
