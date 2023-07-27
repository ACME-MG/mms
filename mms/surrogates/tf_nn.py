"""
 Title:         Neural network
 Description:   For building a neural network 
 Author:        Janzen Choi

"""

# Libraries
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # disable warnings
import numpy as np
import tensorflow.keras as kr
from mms.mapper import MapperDict
from mms.interface.converter import dict_to_grid, grid_to_dict
from mms.surrogates.__surrogate__ import __Surrogate__

# Neural class
class Surrogate(__Surrogate__):

    # Trains the model
    def train(self, input_dict:dict, output_dict:dict, epochs:int, batch_size:int, verbose:bool=False) -> None:
        
        # Initialise model
        input_size = self.get_input_size()
        output_size = self.get_output_size()
        self.model = get_model(input_size, output_size)
        
        # Compile model
        optimiser = kr.optimizers.Adam(learning_rate=1e-3)
        # optimiser = kr.optimizers.SGD(learning_rate=1e-3, clipnorm=1)
        self.model.compile(optimizer=optimiser, loss="mean_squared_error")
        
        # Create mappers
        self.input_mapper = MapperDict(input_dict)
        self.output_mapper = MapperDict(output_dict)
        
        # Map the data dictionaries
        norm_input_dict = self.input_mapper.map(input_dict)
        norm_output_dict = self.output_mapper.map(output_dict)
        
        # Convert the dictionaries to matrices
        norm_input_grid = np.array(dict_to_grid(norm_input_dict))
        norm_output_grid = np.array(dict_to_grid(norm_output_dict))
        
        # Start training
        verbose = 1 if verbose else 0
        self.history = self.model.fit(norm_input_grid, norm_output_grid, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
    # Returns the predictions
    def predict(self, input_dict:dict) -> dict:
        
        # Map and convert input data
        norm_input_dict = self.input_mapper.map(input_dict)
        norm_input_grid = np.array(dict_to_grid(norm_input_dict))
        
        # Gets the prediction
        norm_output_grid = self.model.predict(norm_input_grid)
        output_headers = self.output_mapper.get_headers()
        norm_output_dict = grid_to_dict(norm_output_grid, output_headers)
        
        # Unmap and return
        output_dict = self.output_mapper.unmap(norm_output_dict)
        return output_dict

# Returns the model
def get_model(input_size:int, output_size:int):
    model = kr.Sequential()
    model.add(kr.layers.InputLayer(input_shape=(input_size,)))
    model.add(kr.layers.Dense(units=128))
    model.add(kr.layers.Activation("relu"))
    model.add(kr.layers.Dense(units=64))
    model.add(kr.layers.Activation("relu"))
    model.add(kr.layers.Dense(units=32))
    model.add(kr.layers.Activation("relu"))
    model.add(kr.layers.Dense(units=output_size))
    return model