# neural networks

1. [autoencoders](https://github.com/deepak6446/machine-learning/blob/master/DEEP_AUTOENCODERS/Untitled.ipynb): autoencoders are neural networks that aims to copy their inputs to their outputs. They work by compressing the input into a latent-space representation(encoders), and then reconstructing the output from this representation(decoders). <br>
where to use: autoencoders are used for data denoising and dimensionality reduction.

# machine learning algorithms

1. [Gaussian Naive Bayes](https://github.com/deepak6446/machine-learning/blob/master/NAIVE_BAYES/Untitled.ipynb)
GNB is a Multinomial naive Bayes. it assumes that all the predictors are dependent of each other. but in most cases they are dependent.it can be used in cases where predictors are independent.
iris data [here](https://github.com/deepak6446/machine-learning/blob/master/NAIVE_BAYES/Untitled.ipynb) got 95.55 as compared to randomforest 97.77 [here](https://github.com/deepak6446/machine-learning/blob/master/radom_forest/irisDataset%20random%20forest.ipynb)

2. [XGBoost](https://github.com/deepak6446/machine-learning/blob/master/XGBoost/Untitled.ipynb)
similar to random forest gradient boosting is an ensemble technique which builds final model based on individual weak models.
XGBoost took  0.031 sec to train iris model with same accuracy as randomforest[here](https://github.com/deepak6446/machine-learning/blob/master/radom_forest/irisDataset%20random%20forest.ipynb) which took 0.118 sec.
this shows XGBoost is 3.80 times faster as compared to random forest.

3. [Decesion Tree](https://github.com/deepak6446/machine-learning/blob/master/decision_tree_%20loan_processing/Untitled.ipynb): decision tree starts by splitting using all the features one by one as the root, the tree with the least cost function is chosen. as the size of tree increases, the model becomes complex and small change in data can create a variance. this variance can be removed by purning, bagging boosting. 

4. [Random Tree](https://github.com/deepak6446/machine-learning/blob/master/random_forest/irisDataset%20random%20forest.ipynb): Random tree is an ensemble technique where number of decesion trees are created using features in test data at random and predictions form this tree's are combined to get a more accurate result and reduce variance caused by decision tree because of complexity of single tree.

5. [KNN](https://github.com/deepak6446/machine-learning/tree/master/KNN_Algorithm): k nearest neighbour can be used for both classification and regression. knn is also known as lazy learning algorithm because it does not learn anything during training time, it stores all data in memory whenever any new data comes in it try to find the k nearest data point that has min distance with the new data point. average classification values of this k nearest data point is considered as the output.

6. [K means Clustering]: k means is a clustering algorithm that forms k cluster in such a way that data point in same cluser are very similar while data point in diffrent cluster are very diffrent. it starts with assigning k centroids to random data set points. now each data set distance from centroid is calculated using distance formula like euclidien distance. The data point is assigned to centroid from which its distance is min.

