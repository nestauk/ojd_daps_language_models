#TO DO: WRITE A FLOW FOR HOW TO FINE TUNE A SENTENCE TRANSFORMER 
#MAKE UP PRETEND LABELLED DATA FOR NOW
#USE THIS: https://huggingface.co/blog/how-to-train-sentence-transformers  
"""
Flow to fine tune the last layer of a BERT model on labelled skill
phrase pairs from OJO for semantic similarity of skills.

NOTE: make sure you're logged into huggingface with `huggingface-cli login` 
to upload the fine-tuned model to huggingface before running this flow.

python ojd_daps_language_models/pipeline/train_model/semantic_skill_similarity_flow.py run
"""