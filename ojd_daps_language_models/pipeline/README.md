# Fine-tuning lanuage models with OJO

This directory contains flows to:

1. Prepare data for fine-tuning large language models and;
2. Fine-tune language models for downstream tasks (NER and Semantic Textual Similarity - STS) and for domain adaptation, namely using job advertistment language.

## :scissors: `make_training_data/split_job_ads_flow.py`

This script preprocesses and split job advertisements into sentences using spaCy. The job advertisements used are from the [Open Jobs Observatory](https://www.nesta.org.uk/project/open-jobs-observatory/), a pilot project that scrapes online job adverts from job board sites.

The path to the sample of processed job advertisements is (you must have access to Nesta's S3 bucket):

`open-jobs-lake/escoe_extension/outputs/data/model_application_data/raw_job_adverts_sample.csv`

The sample contains 100,000 job adverts from 2021-05-31 to 2021-06-15 and 1,027,797 unique job advert sentences.

To run the flow:

`python ojd_daps_language_models/pipeline/make_training_data/split_job_ads.py run --chunk_size 1000`

This will save out job advert sentences into chunk_size # of files.

## :office: `/train_model/`

There are a number of flows in the `train_model` directory that fine-tune large language models for a number of different tasks:

1. 💠 Fine-tuning for domain language adaptation: `/ojobert/`
2. 📠 Fine-tuning Sequence Classification head to binarily label sentences as `company description` or not: `/company_descriptions/`

## :sparkles: setting up with AWS and metaflow

If you haven't used batch processing with Metaflow before and want to run any of the flows that make use of batch (e.g. `ojobert_flow.py`), you'll need to ensure a few things are set up first:

1. Your metaflow config file needs be setup with the correct parameters. You can find your config file by executing `metaflow configure show`. If you don't have parameters such as `METAFLOW_ECS_S3_ACCESS_IAM_ROLE` and `METAFLOW_ECS_FARGATE_EXECUTION_ROLE`, contact the DE team.
2. If your laptop username contains a `.` (e.g. if you run `whoami` from the command line and it returns `jack.vines` rather than `jackvines`), you'll need to change your username to remove the `.`. This is because the AWS Batch job will fail to run if your username contains a `.`. To fix this, add `export METAFLOW_USER=<your name without the period>` to a `.env` file at the root of the project. Then, [one time only] run `source .env` to trigger reloading of the variable.
