from abc import ABC, abstractmethod

class Neuron (ABC) : 

    def __init__(self , dim):
        self._dim = dim
        self._w = [0 for i in range(dim+1)]
        super().__init__

    @abstractmethod
    def fit (self , X , Y , eta = 0.1 , iterations = 50 , random_seed = 1):
        pass

    @abstractmethod
    def predict (self , X):
        pass
