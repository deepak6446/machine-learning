# This is a basic implementation of logistic regression for cat prediction with simple sigmoid activation function

### The programme is divided as following 
1. initialize weights and bais with 0 (w, b)
2. calulate output y = activation (A) function where A = W*X +b, where X is input feature vector **Forward propogation**
3. using **Backword propogation using gradient decent** calculate dw = 1/m*(X*[A-Y].T), db = 1/m * (np.sum(A-Y))
4. **cost** = -1/m*(np.sum( Y*(np.log(A)) + (1-Y)*(np.log(1-A)) ))
5. update weights as w = w - learning_rate*dw,b = b - learning_rate*db
6. repeat step 2-5 for the number of iterations.
7. now we have the final weights **W** and bais **b**
8. using this we can predict the output A = sigmoid(w.T.dot(X)+b)   **sigmoid(z)**= 1/(1+np.exp(-z))
