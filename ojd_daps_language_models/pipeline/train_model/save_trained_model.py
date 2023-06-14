import sys

sys.path.append(
    "/Users/india.kerlenesta/Projects/dap_green_jobs/ojd_daps_language_models"
)

"""
If you would like to save a trained model from a flow to s3 based on evaluation results,
run this script from the command line to save results locally and to s3, i.e.:

python ojd_daps_language_models/pipeline/train_model/save_trained_model.py --flow_name=OjoBertFlow
"""

from metaflow import (
    Flow,
)

import os
from ojd_daps_language_models import logger, PROJECT_DIR
import boto3

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_name", type=str, required=True)

    args = parser.parse_args()

    flow = args.flow_name

    # save model and model evaluation locally and to s3 based on latest successful run
    run = Flow(flow).latest_successful_run
    logger.info(f"Using run {str(run)} to save trained model to {run.data.output_dir}")

    model = run.data.bert_model
    tokenizer = run.data.tokenizer

    local_model_path = str(PROJECT_DIR / f"outputs/{run.data.output_dir}")

    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)
    logger.info(local_model_path)

    logger.info("saving model to s3...")
    s3 = boto3.client("s3")
    for file in os.listdir(local_model_path):
        s3.upload_file(
            os.path.join(local_model_path, file),
            "dap-ojobert",
            f"models/{run.data.output_dir}{file}",
        )
    logger.info("...finished saving model to s3")
