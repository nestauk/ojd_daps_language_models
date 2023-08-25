"""
If you would like to save a trained model from a flow to s3 based on evaluation results,
run this script from the command line to save results locally and to huggingface, i.e.:

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
    parser.add_argument("--hf_token", type=str, required=False)

    args = parser.parse_args()

    flow = args.flow_name
    hf_token = args.hf_token

    # save model and model evaluation locally and to s3 based on latest successful run
    run = Flow(flow).latest_successful_run
    logger.info(f"Using run {str(run)} to save trained model to {run.data.output_dir}")

    model = run.data.model
    tokenizer = run.data.tokenizer

    local_model_path = str(PROJECT_DIR / f"outputs/{run.data.output_dir}")

    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)
    logger.info(local_model_path)

    # push model to huggingface hub
    if hf_token:
        hub_command = f"from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('{hf_token}')"
        os.system(f"python -c {hub_command}")
        model.push_to_hub(run.data.output_dir)
