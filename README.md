# Note: 
some dataset may not be available in repo try googling or drop me a mail at deepak.r.poojari@gmail.com

# install dependency 
conda create --name \<your env\> --file requirnment.txt

# Neural Networks

1. [autoencoders](https://github.com/deepak6446/machine-learning/blob/master/DEEP_AUTOENCODERS/Untitled.ipynb): autoencoders are neural networks that aim to copy their inputs to their outputs. They work by compressing the input into a latent-space representation(encoders) and then reconstructing the output from this representation(decoders). <br>
where to use: autoencoders are used for data denoising and dimensionality reduction.

2. [Logistic Regression](https://github.com/deepak6446/machine-learning/blob/master/DEEP_logistic_regression_cat_prediction/Untitled.ipynb): logistic regression using the perceptron model and sigmoid activation function. LR is a classification problem which can be used for binary or multiclassification problem, unlike Linear Regression which gives continuous value.

3. [CNN convolution neural networks](https://github.com/deepak6446/machine-learning/blob/master/DEEP_CNN_digit_recognition/98%25handWrittenDigitPrediction.ipynb): Image contains a huge amount of pixels, training data with a fully connected neural networks make computation expensive, CNN used a combination of filters to reduce the number of parameters required to be learned by the network. further using polling we can reduce the image to some more extend keeping the imp features, using maxpool and avgpool.

4. [LSTM](https://github.com/deepak6446/machine-learning/blob/master/DEEP_LSTM/Generating%20next%20word%20in%20sequence%20using%20rnn.ipynb): fully connected dense layer does not consider the sequence of input while predicting the output. In long short term memory has memory in form of a cell which has forget gates and update gates which decides what to forget and what needs to be feed into LSTM cell later to predict output based on previous input. LSTM is used because RNN are weak at remembering long term dependencies as long term dependencies cause gradient decent problem in RNN.

# Machine Learning Algorithms

1. [Gaussian Naive Bayes](https://github.com/deepak6446/machine-learning/blob/master/NAIVE_BAYES/Untitled.ipynb)
GNB is a Multinomial naive Bayes. it assumes that all the predictors are dependent on each other. but in most cases, they are dependent.it can be used in cases where predictors are independent.
iris data [here](https://github.com/deepak6446/machine-learning/blob/master/NAIVE_BAYES/Untitled.ipynb) got 95.55 as compared to the randomforest 97.77 [here](https://github.com/deepak6446/machine-learning/blob/master/radom_forest/irisDataset%20random%20forest.ipynb)

2. [XGBoost](https://github.com/deepak6446/machine-learning/blob/master/XGBoost/Untitled.ipynb)
similar to random forest gradient boosting is an ensemble technique which builds a final model based on individual weak models.
XGBoost took  0.031 sec to train iris model with the same accuracy as randomforest[here](https://github.com/deepak6446/machine-learning/blob/master/radom_forest/irisDataset%20random%20forest.ipynb) which took 0.118 sec.
this shows XGBoost is 3.80 times faster as compared to random forest.

3. [Decision Tree](https://github.com/deepak6446/machine-learning/blob/master/decision_tree_%20loan_processing/Untitled.ipynb): Decision trees can handle both categorical and numerical data. Decision tree starts by splitting using all the features one by one as the root, the tree with the least cost function is chosen. as the size of tree increases, the model becomes complex and a small change in data can create a variance. this variance can be removed by purning, bagging boosting. 

4. [Random Tree](https://github.com/deepak6446/machine-learning/blob/master/random_forest/irisDataset%20random%20forest.ipynb): Random tree is an ensemble technique where a number of decision trees are created using features in test data at random and predictions form these trees are combined to get a more accurate result and reduce variance caused by decision tree because of the complexity of single tree.

5. [KNN](https://github.com/deepak6446/machine-learning/tree/master/KNN_Algorithm): k nearest neighbor can be used for both classification and regression. knn is also known as lazy learning algorithm because it does not learn anything during training time, it stores all data in memory whenever any new data comes in it try to find the k nearest data point that has min distance with the new data point. average classification values of this k nearest data point is considered as the output.

6. [K means Clustering](https://github.com/deepak6446/machine-learning/blob/master/K_MEANS_CLUSTERING/titanic%20dataset.ipynb): k means is a clustering algorithm that forms k cluster in such a way that data point in the same cluster are very similar while data point in different cluster is very different. it starts with assigning k centroids to random data set points. now each data set distance from centroid is calculated using distance formula like euclidian distance. The data point is assigned to centroid from which its distance is min.

7. [SVM](https://github.com/deepak6446/machine-learning/blob/master/SVM/spam%20non%20spam%20using%20svm.ipynb): email spam, non spam using support vector machine. The objective of SVM is to find a hyperplane in N-dim space that distinctly classifies data point.it looks for support vector to separate decision boundary with the max seperate space.
SVM uses kernel trick to find hyperplane using 3d image of scattered points(if the points are not linearly separable)
regularization(or C parameter) tells SVM how much you want to avoid misclassification.
Gamman: defines how far the influence of single training example in finding the hyperplane.
[visualize hyperplane and margin](https://github.com/deepak6446/machine-learning/blob/master/support_vector_machine_supervised/blobDataset.ipynb)

8. [Regression](https://github.com/deepak6446/machine-learning/blob/master/Regression/linearRegression.ipynb): Multiple Linear Regression is Simple Linear Regression, but with more Relationships. In regression problem the goal of the algorithm is to predict the real-valued output. while training multilinear regression model we are trying to find out coefficient for function ( y = c + a1x + a2x + a3x + a4x ....) that best fits input. the error is minimized by using gradient descent ( J(Y, Y`) ~ 0). Keep changing a1 and a2 to reduce cost function until we hopefully end up at a minimum. to reduce error we will try to remove unwanted features from data. 
also trained the same dataset using GradientBoostingRegressor got higher accuracy but the time taken to train model also increased

# Visualization

1. [matplotlib](https://github.com/deepak6446/machine-learning/blob/master/VISUALIZATION/visualization.ipynb): Line Plot, Scatter Plot, Histogram, Bar chart, Pie chart.
