# :diamond_shape_with_a_dot_inside: Fine-tuning for domain language adaptation

Scripts in this directory fine-tunes a DistilBERT model on job advertisements for domain adaptation for the purposes of next sentence prediction and masked language modelling.

To run the flow in production using AWS batch and the model configurations as described in `training.yaml`, run:

`python ojd_daps_language_models/pipeline/train_model/ojobert/ojobert_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

The model will be fine-tuned on 100,000 sentences and model evaluation metrics are saved to s3.

If you're happy with the evaluation metrics, you can save the model both locally running:

`python ojd_daps_language_models/pipeline/train_model/ojobert/save_trained_model.py --flow_name=OjoBertFlow`

If you would like to also push it to huggingface hub, you need to pass your huggingface API key:

`python ojd_daps_language_models/pipeline/train_model/save_trained_model.py --flow_name=OjoBertFlow --hf_token=<your huggingface API key>`

To test functions used in the flow:

`pytest ojd_daps_language_models/pipeline/train_model/tests/test_ojobert_flow.py`
