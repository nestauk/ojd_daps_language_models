# 🤔 Multi-Skill Classification

Scripts in this directory train a Support Vector Machine (SVM) classifier to predict whether an extracted 'SKILL' span contains multiple skills or not. 

The features used for training the classifier include:
1. The number of tokens in the 'SKILL' span;
2. binary encoding if the token "and" is present in the 'SKILL' span and;
3. binary encoding if the token "," is present in the 'SKILL' span.

To run the flow in production, run:

`python multiskill_flow.py --package-suffixes=.txt run --production=True`

You will need access to Nesta's s3 bucket to train the model. 

If you're happy with the evaluation metrics, you can push the model to HuggingFace by running:

`python multiskill_flow.py --package-suffixes=.txt run --production=True --push_to_hf=True`

You will need access to Nesta's s3 bucket to train the model and be logged into huggingface hub to push the model. 

## 📠 Using the model

To use the model, you can load it from huggingface hub:

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