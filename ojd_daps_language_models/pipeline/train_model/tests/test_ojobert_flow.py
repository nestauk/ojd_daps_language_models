"""test functions from bert_training.py used in ojobert_flow.py"""
import pytest

import collections
import numpy as np
from datasets import Dataset
import pandas as pd

from ojd_daps_language_models.utils.bert_training import tokenize_function, group_texts

job_sents = [
    "Nike is a shoe manufacturing company.",
    "You should have strong communication skills.",
]

test_examples = Dataset.from_pandas(pd.DataFrame({"text": job_sents}))


def test_tokenize_function(examples=test_examples):
    result = tokenize_function(examples)

    assert list(result.keys()) == ["input_ids", "attention_mask", "word_ids"]
    assert len(result["input_ids"]) == len(job_sents)
    assert type(result["input_ids"]) == list
    for input_id_list in result["input_ids"]:
        for input_id in input_id_list:
            assert type(input_id) == int

    assert type(result["attention_mask"]) == list
    assert type(result["word_ids"]) == list


def test_group_texts(examples=test_examples, chunk_size=1):
    result = tokenize_function(examples)
    text_group = group_texts(result, chunk_size=chunk_size)

    assert type(text_group) == dict
    assert list(text_group.keys()) == [
        "input_ids",
        "attention_mask",
        "word_ids",
        "labels",
    ]
    assert type(text_group["input_ids"]) == list
