# 💡 skillNER

This directory contains the scripts needed to train models associated to extracting 'SKILL' spans from text. There are **two** models that are trained:

1. **Multi-Skill Classification**: This model predicts whether an extracted 'SKILL' span contains multiple skills or not. 
2. **SkillNER**: This model predicts the start and end spans of 'SKILL' entities in text. 

## 🤔 Multi-Skill Classification

Scripts beginning with `multiskill_` in this directory train a Support Vector Machine (SVM) classifier to predict whether an extracted 'SKILL' span contains multiple skills or not. 

The features used for training the classifier include:
1. The number of tokens in the 'SKILL' span;
2. binary encoding if the token "and" is present in the 'SKILL' span and;
3. binary encoding if the token "," is present in the 'SKILL' span.

To run the flow in production, run:

`python multiskill_flow.py --package-suffixes=.txt run --production=True`

You will need access to Nesta's s3 bucket to train the model. If you would like to push the model to huggingface hub, you will need to have a huggingface account and fine-grained access API token to the `nestauk` organisation. Once you have the token, you can set it as an environment variable:

```
export HF_TOKEN=<your_token> > .env
```

Then, you can use the `--push-to-hub=True` flag.

### 🤔📠 Using the model

To use the SVM model, you can load it from huggingface hub:

```
import joblib
from pathlib import Path
from skops.hub_utils import download

ms_path = Path.cwd() / "outputs"
download(repo_id="nestauk/multiskill-classifier", dst=ms_path)
model = joblib.load(
	ms_path / "skops-pl2mwv65.pkl"
)
# only load pickle files from sources you trust
# read more about it here https://skops.readthedocs.io/en/stable/persistence.html
```

The model metrics are also reported from huggingface hub. 

## 🍄 skillNER

Scripts beginning with `skillner_` in this directory train a Named Entity Recognition (NER) model to predict the start and end spans of 'SKILL', 'EXPERIENCE', and 'BENEFIT' entities in text.

To run the flow in production, run:

`python skillner_flow.py --package-suffixes=.txt run --production=True`

You will need access to Nesta's s3 bucket to train the model. If you would like to push the model to huggingface hub, you will need to have a huggingface account and fine-grained access API token to the `nestauk` organisation. Once you have the token, you can set it as an environment variable:

```
export HF_TOKEN=<your_token> > .env
```

Then, you can use the `--push-to-hub=True` flag.

### 🍄📠 Using the model

To use the NER model, you can load it from huggingface hub:

```
!pip install https://huggingface.co/nestauk/en_core_web_lg/resolve/main/en_core_web_lg-any-py3-none-any.whl

# Using spacy.load().
import spacy
nlp = spacy.load("en_core_web_lg")

```

The model metrics are also reported from huggingface hub. 