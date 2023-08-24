# 📠 Company Description Classification

Scripts in this directory trains a Sequence Classification head of a [JobBERT model](https://huggingface.co/jjzha/jobbert-base-cased) to binarily classifiy sentences as either a company description or not.

`JobBERT` is a continuously pre-trained `bert-base-cased` checkpoint on ~3.2M sentences from job postings.

To run the flow in production using AWS batch and the model configurations as described in `training.yaml`, run:

`python ojd_daps_language_models/pipeline/train_model/company_descriptions/compdescs_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

If you're happy with the evaluation metrics, you can save the model both locally and to s3 by running:

`python ojd_daps_language_models/pipeline/train_model/save_trained_model.py --flow_name=CompDescFlow`

This will save the trained model from last successful metaflow run both locally to to s3.

## Training data

### 20230824

- **486** company description sentences
- **1000** non company description sentences less than 500 characters in length (random seed: 42)

## Evaluation metrics
