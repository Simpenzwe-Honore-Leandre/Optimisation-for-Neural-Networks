import numpy as np
import matplotlib.pyplot as plt


class LSTM:
    def __init__(self, learning_rate=5e-7, hidden_units=10 ):
        self.learning_rate  = learning_rate
        self.hidden_units   = hidden_units
        self.params         = {}
        self.losses         = []

    def sigmoid(self, X):
        return 1/ (1+ np.exp(-X))

    #This tang squashes the cell input between -2 and 2
    def hp_tang(self, X):
        return (4 * self.sigmoid( X ) - 2 )

    def loss( self , error):
        return 1/2 * ( error ) ** 2
    #hidden_units refer to memory cells
    def train(self,X,Y ):

        self.timesteps , n_features = X.shape

        self.X = X
        self.Y = Y
        _,*dimsout = Y.shape
        n_out = (self.hidden_units,) if len(self.Y.shape) ==1 else ( self.hidden_units,*dimsout )
        #weight inits
        limit = np.sqrt( 6 / ( n_features+self.hidden_units ))
        limitU = np.sqrt( 6 / ( self.hidden_units+self.hidden_units ))
        limitO = np.sqrt( 6 / ( 1 if len(self.Y.shape) ==1 else dimsout[0]+self.hidden_units ))
        """

        Glorot init Draws samples from a uniform distribution within `[-limit, limit]`, where
        `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input

        """
        #input weights
        self.params['Wf'] = np.random.uniform(0, limit, (self.hidden_units , n_features) )
        self.params['Wi'] = np.random.uniform(-limit, -0.0001, (self.hidden_units , n_features) )
        self.params['Wo'] = np.random.uniform(-limit, -0.0001, (self.hidden_units , n_features) )
        self.params['Wc'] = np.random.uniform(-limit, limit, (self.hidden_units , n_features) )
        self.params['Wk'] = np.random.uniform(-limitO, limitO, self.hidden_units  )
        #hidden weights
        self.params['Uf'] = np.random.uniform(0, limitU, (self.hidden_units , self.hidden_units) )
        self.params['Ui'] = np.random.uniform(-limitU, limitU, (self.hidden_units , self.hidden_units) )
        self.params['Uo'] = np.random.uniform(-limitU, limitU, (self.hidden_units , self.hidden_units) )
        self.params['Uc'] = np.random.uniform(-limitU, limitU, (self.hidden_units , self.hidden_units) )
        #bias
        self.params['bf'] = 1#np.random.uniform(0, limit, self.hidden_units )
        self.params['bi'] = np.random.uniform(-limitU, -0.0001, self.hidden_units )
        self.params['bo'] = np.random.uniform(-limitU, -0.0001, self.hidden_units )
        self.params['bc'] = np.random.uniform(-limitU, limitU, self.hidden_units )

        #Ht
        self.hidden_state          = np.zeros(self.hidden_units)

        #~Ct
        self.cell_input_activation = np.zeros(self.hidden_units)

        #Ot
        self.output_activation     = np.zeros(self.hidden_units)

        #Ft
        self.forget_activation     = np.zeros(self.hidden_units)

        #It
        self.input_activation      = np.zeros(self.hidden_units)

        #Ct
        self.cell_state            = np.zeros(self.hidden_units)

        self.final_output          = np.zeros(shape=tuple(n_out))

        sigmoid_derivative         = lambda x: x * (1 - x)

        hp_tang_derivative         = lambda x: 4 * self.sigmoid(x)

        loss_function              = lambda error: 1/2 * (error)**2

        self.error_signal_final_out = np.zeros(self.hidden_units)

        self.error_signal_out_gate  = np.zeros(self.hidden_units)

        cell_input_partials_Wc = cell_input_partials_Uc = cell_input_partials_Wi = cell_input_partials_Ui = cell_input_partials_Wf = cell_input_partials_Uf = 0

        #iterating over sequence
        for timestep in range(self.timesteps):
            # print(timestep)
            self.forget_activation     = self.sigmoid( np.dot(self.params['Wf'] , self.X[timestep] ) + np.dot( self.params['Uf'] , self.hidden_state ) + self.params['bf'] )

            self.input_activation      = self.sigmoid( np.dot(self.params['Wi'] , self.X[timestep] ) + np.dot( self.params['Ui'] , self.hidden_state ) + self.params['bi'] )

            self.output_activation     = self.sigmoid( np.dot(self.params['Wo'] , self.X[timestep] ) + np.dot( self.params['Uo'] , self.hidden_state ) + self.params['bo'] )

            self.cell_input_activation = self.hp_tang( np.dot(self.params['Wc'] , self.X[timestep] ) + np.dot( self.params['Uc'] , self.hidden_state ) + self.params['bc'] )

            prev_cell_state            = self.cell_state

            self.cell_state            = np.multiply( self.forget_activation , self.cell_state) + np.multiply( self.input_activation , self.cell_input_activation)

            prev_hidden_state          = self.hidden_state

            self.hidden_state          = np.multiply( self.output_activation , np.tanh( self.cell_state ) )

            self.final_output          = self.sigmoid( np.dot( self.params['Wk'] , self.hidden_state  ) )

            #BPTT for output units and output gates pass

            error = self.Y[timestep] -  self.final_output

            self.error_signal_out_units = sigmoid_derivative(self.final_output) * error

            self.error_signal_out_gate = sigmoid_derivative(self.output_activation) * (  np.multiply( self.hp_tang( self.cell_state ) , np.dot( self.params['Wk'] , self.error_signal_out_units  )  ) )

            #I need the previous hidden state

            output_unit_gradient = np.outer(  self.error_signal_out_units , prev_hidden_state )

            if len(self.Y.shape) ==1:
                output_unit_gradient = output_unit_gradient[0]

            self.params['Wk']  += self.learning_rate * output_unit_gradient

            #BPTT for output gate

            self.params['Uo']  += self.learning_rate * np.outer(  self.error_signal_out_gate , prev_hidden_state)

            self.params['Wo']  += self.learning_rate * np.outer(  self.error_signal_out_gate , self.X[timestep])


            #for input gates, cell and forget gates we use RTRl
            #before np.outer we elementwise multiply by internal state error.

            internal_state_error              = self.output_activation * 2 * self.sigmoid(self.cell_state) * np.dot( self.params['Wk'] , self.error_signal_out_units )

            cell_input_activation_partials_Wc = np.outer( internal_state_error * cell_input_partials_Wc * self.forget_activation + hp_tang_derivative( self.cell_input_activation ) * self.input_activation ,  self.X[timestep] )

            cell_input_activation_partials_Uc = np.outer( internal_state_error *cell_input_partials_Uc * self.forget_activation + hp_tang_derivative( self.cell_input_activation ) * self.input_activation , prev_hidden_state  )


            cell_igate_activation_partials_Wi = np.outer( internal_state_error *cell_input_partials_Wi * self.forget_activation + sigmoid_derivative(self.input_activation)  * self.cell_input_activation , self.X[timestep] )

            cell_igate_activation_partials_Ui = np.outer( internal_state_error *cell_input_partials_Ui * self.forget_activation + sigmoid_derivative(self.input_activation)  * self.cell_input_activation , prev_hidden_state )


            cell_forget_activation_partial_Wf = np.outer( internal_state_error *cell_input_partials_Wf * self.forget_activation + prev_cell_state * sigmoid_derivative(self.forget_activation) * self.input_activation , self.X[timestep] )

            cell_forget_activation_partial_Uf = np.outer( internal_state_error *cell_input_partials_Uf * self.forget_activation + prev_cell_state * sigmoid_derivative(self.forget_activation) , prev_hidden_state )

            #input weights
            self.params['Wf'] -= self.learning_rate * cell_forget_activation_partial_Wf
            self.params['Wi'] -= self.learning_rate * cell_igate_activation_partials_Wi
            self.params['Wc'] -= self.learning_rate * cell_input_activation_partials_Wc

            #hidden weights
            self.params['Uf'] -= self.learning_rate * cell_forget_activation_partial_Uf
            self.params['Ui'] -= self.learning_rate * cell_igate_activation_partials_Ui
            self.params['Uc'] -= self.learning_rate * cell_input_activation_partials_Uc
            self.losses.append( np.mean( self.loss( error ) ))

            #self.params['bf'] += self.learning_rate *  internal_state_error
            self.params['bi'] += self.learning_rate *  internal_state_error
            self.params['bo'] += self.learning_rate *  self.error_signal_out_gate
            self.params['bc'] += self.learning_rate *  internal_state_error

        plt.plot( range(len(self.losses)) , self.losses )
        plt.xlabel('timesteps')
        plt.ylabel('Loss')

