# :pager: Fine-tuning transformer models with job adverts

**_Flows to fine-tune transformer models for a variety of downstream tasks using job adverts_**

## :wave: Welcome!

This repo contains metaflows that train transformer models for both domain adaptation and a variety of downstream tasks using job adverts from Nesta's [Open Jobs Observatory](https://www.nesta.org.uk/project/open-jobs-observatory/). With the permission of job board sites, we have been collecting online job adverts since 2021 and building algorithms to extract and structure information. We have collected millions of job adverts since the project's inception.

Although we are unable to share the raw data openly, we aim to build open source tools, algorithms and models that anyone can use for their own research and analysis. For example, we have built an [open-source Skills Extractor library](https://nestauk.github.io/ojd_daps_skills/build/html/about.html) and have [an open locations extraction repo](https://github.com/nestauk/ojd_daps_locations).

This repo contains the metaflows used to fine-tune transformer models with job adverts for a variety of downstream tasks, including:

1. next-sentence prediction
2. masked language modelling
3. skill semantic similarity
4. named entity recognition

## :cupid: Using fine-tuned model checkpoints

The fine-tuned models (and their associated model cards) can be accessed via huggingface's hub:

- [📠 Company Description Classification](https://huggingface.co/ihk/jobbert-base-cased-compdecs)

## Setup

To run the flows, you will need to:

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`
- Download spacy model: `python -m spacy download en_core_web_sm`
- install Pytorch: `conda install pytorch torchvision -c pytorch` (if you are using mac OS x 13.4 operating system - `pip install torch`)
- Set up batch processing with Metaflow
- Sign into huggingface hub to push models to huggingface
- run `export LC_ALL="en_GB.UTF-8"` in your terminal

However, to simply use the models, please refer to [:cupid: Using fine-tuned model checkpoints](https://github.com/nestauk/ojd_daps_language_models#cupid-using-fine-tuned-model-checkpoints) section.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
