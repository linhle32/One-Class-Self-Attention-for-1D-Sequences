# One-class Self-attention for Anomaly Detection in 1D Sequences

<b>*** Update 8/4/2021: my paper has been published at https://link.springer.com/chapter/10.1007/978-3-030-82196-8_20 </b>

This repository consists of my implementation for One-Class Self-Attention model for anomaly detection in 1D sequences. Unfortunately, this one is code only because the research is in collaboration with an industrial partner whose data I cannot publish. 

### Libraries
- tensorflow
- numpy
- scikit-learn

### Summary

There are three (or four) deep learning models implemented for the task:
- 1D-CNN Auto-Encoder: standard model for benchmarking. Transforms 1D sequences into embedded vectors. Anomalies are detected by determining sequences that decoded versions are too different from originals. 
- Deep One-Class Classifier - implementation of the model in <i>Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., & Kloft, M. (2018, July). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR. </b> Maps 1D sequences to a vector space where regular instances are within a hyper-sphere and anomalies outside of that.
- One-class self-attention: my model, two versions
    1. LeakyReLU output with regular centers
    2. Sigmoid output without centers

Instead of building a chain of class inheritance, I decided to implement the model separately so that it is more convenient if you are just interested in one of them.

### Files
- <b>benchmark_models.py</b>: 1D-CNN Auto-Encoder and Deep One-Class Classifier
- <b>oneclass-selfattention.py</b>: My models
- <b>benchmark.ipynb</b>: template to test all models along with SKLearn One-class Support Vector Machine and Isolation Forest

### Model architecture

![image](https://user-images.githubusercontent.com/5643444/232246660-6ce5e721-5422-46f8-b5a9-49540491581a.png)
