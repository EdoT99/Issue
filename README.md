# Issue
This repository is meant to share the code and a toy dataset to be analyzed.

The current issue are wrong ROC curves produced using 'macro' - average procedure from sklearn.

The code 10_fold_cross_validaiton.py performs a stratified 10 fold CV on an input dataset. Given an imbalanced dataset the code is developed to in such a way to oversample the minority classes found in each fold with two different techniques. In this way, three different training datasets (SMOTE,KDE and baseline) are used for training a classifier. Then, the models are tested on the same validation set. The same procedure is repeated for the other k-1 folds. Predicted probabilities are aggregated for all three models in three final vectors.

As a toy dataset, a modified Iris dataset: 'Iris_imbalanced.csv' is provided as an exmaple. The dataset consist in 3 classes and 4 attributes. Classes are imbalanced by removing some examples as it is shown below:
Classes:
    Iris-versicolor    50
    Iris-setosa        45
    Iris-virginica     20

What I really care about is understanding why in the case of macro-average ROC curves, the plot does not start from (0,0) in this case.

You can choose between three different classifiers namely: NB, DT and RF

Therefore the script outputs two distinct graphs:
 - micro averaged ROC curves
 - macro averaged ROC curves

Thanks for helping !!
