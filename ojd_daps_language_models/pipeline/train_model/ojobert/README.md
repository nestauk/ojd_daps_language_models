# :diamond_shape_with_a_dot_inside: Fine-tuning for domain language adaptation

Scripts in this directory fine-tunes a DistilBERT model on job advertisements for domain adaptation for the purposes of next sentence prediction and masked language modelling.

To run the flow in production using AWS batch and the model configurations as described in `training.yaml`, run:

`python ojd_daps_language_models/pipeline/train_model/ojobert/ojobert_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

The model will be fine-tuned on 100,000 sentences and model evaluation metrics are saved to s3.

If you're happy with the evaluation metrics, you can save the model and tokenizer locally by running:

`python ojd_daps_language_models/pipeline/train_model/ojobert/save_trained_model.py --flow_name=OjoBertFlow`

To test functions used in the flow:

`pytest ojd_daps_language_models/pipeline/train_model/tests/test_ojobert_flow.py`

## 📠 Using the model

To use the model, you can load it from huggingface hub:

```
from transformers import pipeline

model = pipeline('fill-mask', model='ihk/ojobert', tokenizer='ihk/ojobert')

text = "Would you like to join a major [MASK] company?"
model(text, top_k=3)

>> [{'score': 0.1886572688817978,
  'token': 13859,
  'token_str': 'pharmaceutical',
  'sequence': 'would you like to join a major pharmaceutical company?'},
 {'score': 0.07436735928058624,
  'token': 5427,
  'token_str': 'insurance',
  'sequence': 'would you like to join a major insurance company?'},
 {'score': 0.06400047987699509,
  'token': 2810,
  'token_str': 'construction',
  'sequence': 'would you like to join a major construction company?'}]
```
