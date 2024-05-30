"""
If you would like to save a trained model from a flow based on evaluation results,
run this script from the command line to save results locally:

python save_model/save_trained_model.py --flow_name=CompDescFlow
"""

from metaflow import (
    Flow,
)

import boto3

import argparse

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_name", type=str, required=True)

    args = parser.parse_args()
    flow = args.flow_name

    # save model and model evaluation locally
    run = Flow(flow).latest_successful_run
    print(f"Using run {str(run)} to save trained model")

    model = run.data.model
    tokenizer = run.data.tokenizer

    model_name = run.data.name.lower().split("flow")[0]
    local_model_path = str(PROJECT_DIR / f"outputs/models/{model_name}")

    print("saving model and tokenizer locally...")
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)
