"""
Utilites for the ojobert pipeline
"""
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import yaml

# import config file training.yaml
config_file_path = f"{os.path.dirname(os.path.realpath(__file__))}/training.yaml"

with open(config_file_path, "rt") as f:
    CONFIG = yaml.load(f.read(), Loader=yaml.FullLoader)

# load tokenizer and model from config file
tokenizer = AutoTokenizer.from_pretrained(CONFIG["checkpoint"])
bert_model = AutoModelForMaskedLM.from_pretrained(CONFIG["checkpoint"])


def tokenize_function(examples):
    result = tokenizer(
        examples["text"], max_length=tokenizer.model_max_length, truncation=True
    )
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def group_texts(examples, chunk_size: int = CONFIG["bert_chunk_size"]):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
