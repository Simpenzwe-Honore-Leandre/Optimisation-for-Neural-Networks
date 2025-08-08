from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class RNN:
    """
    RNN implementation from staudemeyer et al
    """
    def __init__(self, learning_rate=0.5 , epochs=1000, hidden_units=10 ):
        self.learning_rate= learning_rate
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.cache= defaultdict(list)

    def sigmoid(self , X):
        return 1 / ( 1 + np.exp(-X))

    def loss(self , y_true , y_pred ):

        n = self.input_size[0]

        e = y_true - y_pred

        loss = 1/(2 * n ) * np.sum( np.square( e ))

        return loss

    def weight_update(self):

        error_t  = 1/self.input_size[0] * self.cache["activations"][-1] * (1 - self.cache["activations"][-1] ) * ( self.Y[-1] - self.cache["activations"][-1] )

        delta_Wi = np.zeros( ( self.hidden_units , self.input_size[1] ) )

        delta_Wc = np.zeros( (self.hidden_units , self.hidden_units) )

        #Note to self, this can be replaced with np.outer

        for i,u in enumerate(error_t):

            #for each hidden unit sum up product of recurrent connections and respective unit error signal at that specific timestep. so we get an H size matrix containing sums of error signal into all previous inputs

            delta_Wc[i] += ( u * self.cache["activations"][-2] )

            delta_Wi[i] += ( u * self.X[-2] )

        for timestep in range( self.input_size[0]-2,0,-1  ):

            error_t = 1/self.input_size[0] * self.cache["activations"][timestep-1] * (1 - self.cache["activations"][timestep-1] ) * np.dot( self.context_weights , error_t )

            for i,u in enumerate(error_t):

                delta_Wc[i] += ( u * self.cache["activations"][timestep-1] )

                delta_Wi[i] += ( u * self.X[timestep-1] )


        self.input_weights += ( self.learning_rate * delta_Wi )

        self.context_weights += ( self.learning_rate *  delta_Wc )



    def train(self , X , Y ):
        self.cache.clear()
        self.X = X
        self.Y = Y
        self.input_size = self.X.shape

        # Glorot initialisation
        self.input_weights = np.random.randn(self.hidden_units, self.input_size[1]) * np.sqrt(1.0 / self.input_size[1])

        self.context_weights = np.random.randn(self.hidden_units, self.hidden_units) * np.sqrt(1.0 / self.hidden_units)

        self.epoch_loss=[]

        training_counter = 1

        step = self.epochs // 100

        for epoch in range(self.epochs):
            self.cache.clear()
            activation = np.zeros( self.hidden_units, )
            for timestep in range( self.input_size[0] ):
                state = np.dot(self.context_weights , activation  ) + np.dot(   self.input_weights, self.X[timestep]   )
                activation = self.sigmoid( state )
                self.cache["states"].append( state )
                self.cache["activations"].append(activation)
                self.cache["Losses"].append( self.loss( self.Y[timestep] , activation ) )
            self.weight_update()
            self.epoch_loss.append( np.sum( self.cache["Losses"] ) / self.input_size[0] / self.hidden_units )
            if epoch+1 == training_counter * step:
                print( "[", "=" *training_counter, "]" , f" the loss for epoch {1+epoch} is  {  self.epoch_loss[-1] }" )
                training_counter +=1
        plt.plot( range(self.epochs) , self.epoch_loss )
        plt.xlabel('X-axis ,  epochs')
        plt.ylabel('Y-axis ,  Loss')

