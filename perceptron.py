import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class perceptron_model:
    """
    we need
    input, weights and learning rate
    """
    def __init__(self, learning_rate=0.8 , epochs=200 ):
        self.learning_rate= learning_rate
        self.epochs = epochs
        self.cache= defaultdict(list)

    def sigmoid(self , X):
        return 1 / ( 1 + np.exp(-X))

    def loss(self , y_true , y_pred ):
        # print(y_true.shape[0])
        n = y_true.shape[0]
        e = y_true - y_pred
        loss = 1/(2 * n ) * np.sum( np.square( e ))
        return loss

    def weight_update(self, y_true , y_pred):
        n = y_true.shape[0]
        delta_W =   self.learning_rate / n *  np.dot( ( y_true - y_pred )  , self.X )
        delta_B =  self.learning_rate  * np.mean(( y_true - y_pred ))
        self.bias += delta_B
        self.weights += delta_W

    def train(self , X , Y ):
        self.cache.clear()
        self.X = X
        self.Y = Y
        self.input_size = self.X.shape
        # Xavier initialisation
        self.weights = np.random.normal(0, np.sqrt( 1 / self.input_size[1] ) , size=self.input_size[1] )
        self.bias= 0
        output = None
        for epoch in range(self.epochs):
            state = np.dot( self.weights , self.X.T ) + self.bias
            activation = self.sigmoid( state )
            self.cache["states"].append(output)
            self.cache["activations"].append(activation)
            self.cache["Losses"].append( self.loss( self.Y , activation ) )
            self.weight_update( self.Y , activation )
            print( f"the loss for epoch {1+epoch} is  { self.cache['Losses'][-1] }" )
        plt.plot( range(self.epochs) , self.cache["Losses"] )
        plt.xlabel('X-axis ,  epochs')
        plt.ylabel('Y-axis ,  Loss')

    def test(self, y_test ):
        self.y_pred = np.dot( self.weights , y_test.T )  + self.bias
        return self.y_pred
