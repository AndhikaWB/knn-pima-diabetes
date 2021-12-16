# KNN from Scratch
An attempt to make KNN (K-nearest neighbors) algorithm from scratch for [PIMA Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) dataset.

## Features
* Written in pure Python
* Minimal libraries (`csv`, `statistics`, `copy`)
* Convert zero to null (for use with imputer)
* Imputer (fill null values with mean/median/mode from closest neighbors)
* Predict class and regression test (find the most optimal number of neighbors)
* K-fold cross validation (divide 1 dataset into X parts and test the accuracy)
* Import and export dataset from/to CSV file (with/without label)

## References
* [Develop k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
* [Let’s Make a KNN Classifier from Scratch](https://towardsdatascience.com/lets-make-a-knn-classifier-from-scratch-e73c43da346d)
* [kNN Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/)
* [Everything you need to know about Min-Max normalization](https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79)
