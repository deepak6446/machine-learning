# neural networks

1. [autoencoders](https://github.com/deepak6446/machine-learning/blob/master/DEEP_AUTOENCODERS/Untitled.ipynb): autoencoders are neural networks that aims to copy their inputs to their outputs. They work by compressing the input into a latent-space representation(encoders), and then reconstructing the output from this representation(decoders).
where: autoencoders are used for data denoising and dimensionality reduction.

# machine learning algorithms

2. [Gaussian Naive Bayes](https://github.com/deepak6446/machine-learning/blob/master/NAIVE_BAYES/Untitled.ipynb)
GNB is a Multinomial naive Bayes. it assumes that all the predictors are dependent of each other. but in most cases they are dependent.it can be used in cases where predictors are independent.
iris data [here](https://github.com/deepak6446/machine-learning/blob/master/NAIVE_BAYES/Untitled.ipynb) got 95.55 as compared to randomforest 97.77 [here](https://github.com/deepak6446/machine-learning/blob/master/radom_forest/irisDataset%20random%20forest.ipynb)

3. [XGBoost](https://github.com/deepak6446/machine-learning/blob/master/XGBoost/Untitled.ipynb)
similar to random forest gradient boosting is an ensemble technique which builds final model based on individual weak models.
XGBoost took  0.031 sec to train iris model with same accuracy as randomforest[here](https://github.com/deepak6446/machine-learning/blob/master/radom_forest/irisDataset%20random%20forest.ipynb) which took 0.118 sec.
this shows XGBoost is 3.80 times faster as compared to random forest.

4. [Decesion Tree](https://github.com/deepak6446/machine-learning/blob/master/decision_tree_%20loan_processing/Untitled.ipynb): decision tree starts by splitting using all the features one by one as the root, the tree with the least cost function is chosen. as the size of tree increases, the model becomes complex and small change in data can create a variance. this variance can be removed by purning, bagging boosting. 

5. [Random Tree](https://github.com/deepak6446/machine-learning/blob/master/random_forest/irisDataset%20random%20forest.ipynb): Random tree is an ensemble technique where number of decesion trees are created using features in test data at random and predictions form this tree's are combined to get a more accurate result and reduce variance caused by decision tree because of complexity of single tree.


