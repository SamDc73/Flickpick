# Movie Recommendation System with PyTorch

This repository contains Jupyter notebooks for training and tuning a movie recommendation system using PyTorch and the MovieLens dataset.

## Notebooks

### 1. main.ipynb

This notebook contains the main code for training the movie recommendation model. It includes:

- Implementing a collaborative filtering approach using matrix factorization, specifically the Neural Collaborative Filtering (NCF) algorithm.
- Employing techniques such as embedding layers, fully connected layers with non-linear activations, dropout regularization, and optimization algorithms like Adam to capture complex user-movie interactions and prevent overfitting.
- Evaluating the trained model's performance useing Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), and generating personalized movie recommendations based on the learned user and movie embeddings.

## Models

- The trained movie recommendation model has been serialized and saved in two different formats: PyTorch model checkpoint (.pth) and ONNX (Open Neural Network Exchange) format (.onnx).
- PyTorch model checkpoint (final_model_state_dict.pth): The model's parameters have been converted to half-precision (FP16).
- ONNX format (model.onnx) has been exported using torch.onnx.export().

## Dataset

- used the MovieLens 25M Dataset which can be found [here](https://grouplens.org/datasets/movielens/)

## Prerequisites

- Python 3.6+
- Jupyter Notebook
- PyTorch
- Additional Python libraries: pandas, numpy, matplotlib, tqdm, scikit-learn.


