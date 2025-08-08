import numpy as np

class RNN_RTRL:
    def __init__(self,n_features,learning_rate=0.01,hidden_units=10):
        self.learning_rate= learning_rate
        self.hidden_units=hidden_units
        self.cache = {}
        limit = np.sqrt( 6 / ( n_features+self.hidden_units ))
        self.hidden_weights = np.random.uniform(-limit, limit, (self.hidden_units , self.hidden_units) )

        self.input_weights = np.random.uniform(-limit, limit , ( self.hidden_units, n_features) )

    def loss(self, predicted , target):
        return (1/2*self.hidden_units) * np.square(target - predicted)

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def train(self,X , target ):
        """trains real time recurrent logic"""
        self.cache.clear()


        self.X = X

        self.target = target

        #for simplicity we assume offline learning

        self.timesteps , n_features = self.X.shape

        # limit = np.sqrt( 6 / ( n_features+self.hidden_units ))

        """

        Glorot init Draws samples from a uniform distribution within `[-limit, limit]`, where
        `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input

        """


        #sensitivity matrices
        self.sensitivity_tensor = np.zeros(shape=(self.hidden_units ,self.hidden_units , self.hidden_units))

        self.hidden_state = np.zeros(self.hidden_units)

        self.activation = np.zeros(self.hidden_units)

        self.cache["activations"] = [self.activation ]
        self.cache["losses"] = []

        sigmoid_derivative = lambda activation: activation * (1 - activation)

        self.gradients = np.zeros( (self.hidden_units , self.hidden_units) )

        for timestep in range(self.timesteps):

            self.hidden_state = np.dot(self.input_weights , self.X[timestep] ) + np.dot(self.hidden_weights , self.activation)

            self.activation = self.sigmoid( self.hidden_state )

            #computing gradients
            error = self.target[timestep] - self.activation

            loss  = self.loss(self.activation, self.target[timestep] )

            self.cache["losses"].append( np.mean(loss) )

            for u in range(self.hidden_units):
                for v in range(self.hidden_units):
                    temp = self.cache["activations"][-1][u] + np.dot( self.hidden_weights , self.sensitivity_tensor[u][v] )

                    self.sensitivity_tensor[u][v] = np.multiply( sigmoid_derivative(self.activation) , temp)

                    self.gradients[u][v] += np.dot(error, self.sensitivity_tensor[u][v])

            self.hidden_weights += self.gradients  * self.learning_rate

            self.cache["activations"].append( self.activation )
            print( f"\t the loss for timestep {timestep} is: {np.mean(loss)}")