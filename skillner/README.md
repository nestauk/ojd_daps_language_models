# 💡 skillNER

This directory contains the scripts needed to train models associated to extracting 'SKILL' spans from text. There are **two** models that are trained:

1. **Multi-Skill Classification**: This model predicts whether an extracted 'SKILL' span contains multiple skills or not. The features used for training the classifier include:
    - The number of tokens in the 'SKILL' span;
    - binary encoding if the token "and" is present in the 'SKILL' span;
    - binary encoding if the token "," is present in the 'SKILL' span.

2 **SkillNER**: This model predicts the start and end spans of 'SKILL' entities in text. 