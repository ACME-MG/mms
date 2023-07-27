"""
 Title:         Deep neural network
 Description:   Contains the DNN for the Surrogate Model 
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # disable warnings
import tensorflow.keras as kr

# Constants
LEARNING_RATE = 0.1
MOMENTUM      = 0.1
HIDDEN_LAYERS = [32, 16, 8]
ACTIVATION    = "relu"

# DNN Class
class DNN:

    def __init__(self, input_size:int, output_size:int):
        """
        Deep neural network class for surrogate modelling
        
        Parameters:
        * `input_size`:  The number of inputs of the DNN
        * `output_size`: The number of outputs of the DNN
        """

        # Initialise model
        self.model = kr.Sequential()
        self.model.add(kr.layers.InputLayer(input_shape=(input_size,)))
        for i in range(len(HIDDEN_LAYERS)):
            self.model.add(kr.layers.Dense(units=HIDDEN_LAYERS[i]))
            self.model.add(kr.layers.Activation(ACTIVATION))
        self.model.add(kr.layers.Dense(units=output_size))
        self.model.add(kr.layers.Activation(ACTIVATION))
        
        # Define optimiser and compile
        # opt = kr.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
        optimiser = "adam"
        loss_function = "mse"
        metrics = ["accuracy"]
        self.model.compile(optimizer=optimiser, loss=loss_function, metrics=metrics)

    def train(self, x_train:list, y_train:list, epochs:int=100, batch_size:int=None, verbose:bool=False):
        """
        Trains the model
        
        Parameters:
        * `x_train`:    List of training inputs
        * `y_train`:    List of training outputs
        * `epochs`:     Number of epochs
        * `batch_size`: The size of the batches during training
        * `verbose`:    Whether or not to train with additional output from tensorflow
        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        verbose = 1 if verbose else 0
        batch_size = batch_size if batch_size != None else round(len(x_train)/2)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, x_test:list):
        """
        Makes predictions based on a fitted model
        
        Parameters:
        * `x_test`: List of inputs for prediction
        
        Returns the predictions based on the inputs
        """
        x_test = np.array(x_test)
        y_pred = self.model.predict(x_test)
        return y_pred