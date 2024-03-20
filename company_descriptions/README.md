# 📠 Company Description Classification

Scripts in this directory trains a Sequence Classification head of a [JobBERT model](https://huggingface.co/jjzha/jobbert-base-cased) to binarily classifiy sentences as either a company description or not.

`JobBERT` is a continuously pre-trained `bert-base-cased` checkpoint on ~3.2M sentences from job postings.

To run the flow in production using AWS batch and the model configurations as described in `training.yaml`, run:

`python compdescs_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

If you're happy with the evaluation metrics, you can save the model locally by running:

`python train_model/save_trained_model.py --flow_name=CompDescFlow`

### Training Data

The training data is produced from source online job adverts by:
    - loading a source file of online job adverts
    - removing numbers and stop words
    - creating new sentences by identifying instances of Camel Case
    - splitting sentences

This results in a dataset of online job adverts with cleaned sentences.

If internal to Nesta, generate the training data by running:

`python make_training_data/split_job_ads_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

The `compdescs_flow.py` script will then use this data to train the model.

## 📠 Using the model

To use the model, you can load it from huggingface hub:

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

model = AutoModelForSequenceClassification.from_pretrained("nestauk/jobbert-base-cased-compdecs")
tokenizer = AutoTokenizer.from_pretrained("nestauk/jobbert-base-cased-compdecs")

comp_desc = pipeline('text-classification', model=model, tokenizer=tokenizer)

job_sent = "Would you like to join a major manufacturing company?"
comp_desc(job_sent)

>> [{'label': 'LABEL_1', 'score': 0.9953641891479492}]
```

where `LABEL_1` is the label for company description sentence.

Alternatively, we've refactored this into an easy to use python package, located [here](https://github.com/nestauk/ojd_daps_company_descriptions), which can be pip installed, using:

```bash
pip install git+https://github.com/nestauk/ojd_daps_company_descriptions.git
```

## 📠 20230825 model metrics

- **486** company description sentences
- **1000** non company description sentences less than 250 characters in length (random seed: 69)
- overall **accuracy** of **0.92517** on a held out test set of 147 sentences.

Fine-tuning metrics:

| Metric                  | Value      |
| ----------------------- | ---------- |
| eval_loss               | 0.462236   |
| eval_runtime            | 0.629300   |
| eval_samples_per_second | 233.582000 |
| eval_steps_per_second   | 15.890000  |
| epoch                   | 10.000000  |
| perplexity              | 1.590000   |

Test set metrics:

| label                   | precision | recall   | f1-score | support |
| ----------------------- | --------- | -------- | -------- | ------- |
| not company description | 0.930693  | 0.959184 | 0.944724 | 98      |
| company description     | 0.913043  | 0.857143 | 0.884211 | 49      |
