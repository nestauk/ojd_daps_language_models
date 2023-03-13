"""
Functions to load data.
"""

from fnmatch import fnmatch
import boto3
from typing import List

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
