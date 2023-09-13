"""
Utilites for the compdesc pipeline
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import yaml

# import config file training.yaml
config_file_path = f"{os.path.dirname(os.path.realpath(__file__))}/training.yaml"

with open(config_file_path, "rt") as f:
    CONFIG = yaml.load(f.read(), Loader=yaml.FullLoader)

# define the model and tokenizer from the config file
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["checkpoint"], num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["checkpoint"])


def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)
