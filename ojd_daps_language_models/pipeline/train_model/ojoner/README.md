# 🏝️ OjoNER

Scripts in this directory train a spaCy NER model to extract the following entities:

1. `SKILL`
2. `MULTISKILL`
3. `EXPERIENCE`
4. `BENEFITS`

To run the flow in production using AWS batch and the model configurations as described in `training.yaml`, run:

`python ojd_daps_language_models/pipeline/train_model/ojoner/ojoner_flow.py --package-suffixes=.txt,.yaml --datastore=s3 --production=True run`

If you're happy with the evaluation metrics, you can save the model locally by running:

`python ojd_daps_language_models/pipeline/train_model/save_trained_model.py --flow_name=OjoNerFlow`
