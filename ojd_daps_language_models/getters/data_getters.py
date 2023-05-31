"""
Functions to load data.
"""

from fnmatch import fnmatch
import boto3
from typing import List

from fnmatch import fnmatch
import json
import pickle
import gzip
import os

import pandas as pd
from pandas import DataFrame
import boto3
from decimal import Decimal
import numpy

from ojd_daps_language_models import BUCKET_NAME, PROJECT_DIR, logger

s3 = boto3.resource("s3")


def get_s3_data_paths(
    s3: boto3.resource, bucket_name: str, root: str, file_types: List[str] = ["*.json"]
) -> List[str]:
    """Get a list of s3 data paths

    Args:
        s3 (boto3.resource): s3 resource
        bucket_name (str): Name of the bucket in s3
        root (str): Root directory of the data
        file_types (List[str], optional): File type(s) to search for in root directory. Defaults to ["*.json"].

    Returns:
        List[str]: List of s3 data paths
    """
    if isinstance(file_types, str):
        file_types = [file_types]

    bucket = s3.Bucket(bucket_name)

    s3_keys = []
    for files in bucket.objects.filter(Prefix=root):
        key = files.key
        if any([fnmatch(key, pattern) for pattern in file_types]):
            s3_keys.append(key)

    return s3_keys


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(CustomJsonEncoder, self).default(obj)


def load_data(file_name: str, local=True) -> DataFrame:
    """Loads data from path.
    Args:
            file_name (str): Local path to data.
    Returns:
            file (pd.DataFrame): Loaded Data in pd.DataFrame
    """
    if local:
        if fnmatch(file_name, "*.csv"):
            return pd.read_csv(file_name)
        else:
            logger.error(f'{file_name} has wrong file extension! Only supports "*.csv"')


def load_json_dict(file_name: str) -> dict:
    """Loads a dict stored in a json file from path.
    Args:
            file_name (str): Local path to json.
    Returns:
            file (dict): Loaded dict
    """
    if fnmatch(file_name, "*.json"):
        with open(file_name, "r") as file:
            return json.load(file)
    else:
        logger.error(f'{file_name} has wrong file extension! Only supports "*.json"')


def save_json_dict(dictionary: dict, file_name: str):
    """Saves a dict to a json file.

    Args:
            dictionary (dict): The dictionary to be saved
            file_name (str): Local path to json.
    """
    if fnmatch(file_name, "*.json"):
        with open(file_name, "w") as file:
            json.dump(dictionary, file)
    else:
        logger.error(f'{file_name} has wrong file extension! Only supports "*.json"')


def load_txt_lines(file_name: str) -> list:
    txt_list = []
    if fnmatch(file_name, "*.txt"):
        with open(file_name) as file:
            for line in file:
                txt_list.append(line.rstrip())
    else:
        logger.error(f'{file_name} has wrong file extension! Only supports "*.txt"')

    return txt_list


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def save_to_s3(bucket_name, output_var, output_file_dir):
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.csv"):
        output_var.to_csv("s3://" + bucket_name + "/" + output_file_dir, index=False)
    elif fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        obj.put(Body=pickle.dumps(output_var))
    elif fnmatch(output_file_dir, "*.gz"):
        obj.put(Body=gzip.compress(json.dumps(output_var).encode()))
    elif fnmatch(output_file_dir, "*.txt"):
        obj.put(Body=output_var)
    else:
        obj.put(Body=json.dumps(output_var, cls=CustomJsonEncoder))

    logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def load_s3_json(s3, bucket_name, file_name):
    """
    Load a file from S3 without relying on the file_name extension
    as load_s3_data does. Good for files which have no extension.
    """

    obj = s3.Object(bucket_name, file_name)
    file = obj.get()["Body"].read().decode()
    return json.loads(file)


def load_s3_data(bucket_name, file_name):
    """
    Load data from S3 location.

    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.jsonl.gz", "*.jsonl", or "*.json"'
        )


def load_file(bucket_name, file_path, s3=True):
    """
    Load a file either from the repos s3 bucket or locally
    """
    if s3:
        S3 = get_s3_resource()
        data = load_s3_data(S3, bucket_name, file_path)
    else:
        if fnmatch(file_path, "*.json"):
            data = load_json_dict(str(PROJECT_DIR) + "/" + file_path)
        if fnmatch(file_path, "*.csv"):
            data = load_data(str(PROJECT_DIR) + "/" + file_path)
        if fnmatch(file_path, "*.txt"):
            data = load_txt_lines(str(PROJECT_DIR) + "/" + file_path)

    return data
