# KNN from Scratch
An attempt to make KNN (K-nearest neighbors) algorithm from scratch for [PIMA Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) dataset.

## Features
* Written in pure Python
* Accuracy between 68% (worst) and 84% (best), averaged 75%
* Only use built-in libraries (`csv`, `statistics`, `copy`, `random`)
* Import and export dataset from/to CSV file (with/without label)
* Convert zero to null (use it together with imputer)
* Imputer (fill null values with mean/median/mode from closest neighbors)
* Min-max scaling (rescale values in dataset to range between 0 and 1)
* SMOTE oversampling (scale minorities to have the same size as majorities)
* Predict class and regression test (find the most optimal number of neighbors)
* K-fold cross validation (divide 1 dataset into X parts and test the accuracy)

## References
* [Develop k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
* [Let’s Make a KNN Classifier from Scratch](https://towardsdatascience.com/lets-make-a-knn-classifier-from-scratch-e73c43da346d)
* [kNN Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/)
* [Everything you need to know about Min-Max normalization](https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79)
* [Pima Indians Diabetes - Prediction & KNN Visualization](https://towardsdatascience.com/pima-indians-diabetes-prediction-knn-visualization-5527c154afff)
* [Machine Learning Basics with the K-Nearest Neighbors Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
* [SMOTE: Synthetic Minority Over-sampling Technique](https://www.researchgate.net/publication/220543125_SMOTE_Synthetic_Minority_Over-sampling_Technique)
