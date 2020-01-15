import numpy as np

class Perceptron(object):

    def __init__(self , dim):
        self._dim = dim
        self._w = [0 for i in range(dim+1)]

    def fit (self , X , Y , eta = 0.1 , iterations = 50 , random_seed = 1):

        #validate input values
        if X.shape[1] is not self._dim-1:
            raise Error("input array dimension not valid")

        if X.shape[1] is not Y.shape[1]:
            raise Error("input and predictor arrays are not same dimension")

        #initalize weights randomly (pick with normal distribution and zero mean)
        rgen = np.random.RandomState(random_seed)
        self._w = rgen.normal(loc=0.0 , scale = 0.1 , size = self._dim)

        #start iterations
        for _ in range(iterations):
            
            #for each example provided
            for x_i , y_i in zip(X , Y):

                #update the weight calculating error with ith example
                deltaW = eta*(y_i - self.predict(x_i))
                self._w[0] += deltaW
                self._w[1:] += deltaW * x_i
        
        return self


    def predict (self , X):
        z = np.dot(X , self._w[1:]) + self._w[0]
        return np.where(z >= 0 , 1 , -1 )