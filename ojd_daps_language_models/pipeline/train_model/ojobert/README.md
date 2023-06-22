# :diamond_shape_with_a_dot_inside: Fine-tuning for domain language adaptation

Scripts in this directory fine-tunes a DistilBERT model on job advertisements for domain adaptation for the purposes of next sentence prediction and masked language modelling.

To run the flow in production using AWS batch and the model configurations as described in `training.yaml`, run:

`python ojd_daps_language_models/pipeline/train_model/ojobert/ojobert_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

The model will be fine-tuned on 100,000 sentences and model evaluation metrics are saved to s3.

If you're happy with the evaluation metrics, you can save the model both locally and to s3 by running:

`python ojd_daps_language_models/pipeline/train_model/ojobert/save_trained_model.py --flow_name=OjoBertFlow`

This will save the trained model from last successful metaflow run both locally to to s3.

To test functions used in the flow:

`pytest ojd_daps_language_models/pipeline/train_model/tests/test_ojobert_flow.py`
