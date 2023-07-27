"""
 Title:         SMT RBF Surrogate
 Description:   Surrogate using the SMT tool box
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
from smt.surrogate_models import RBF
from mms.mapper import MapperDict
from mms.interface.converter import dict_to_grid, grid_to_dict
from mms.surrogates.__surrogate__ import __Surrogate__

# SMT RBF Class
class Surrogate(__Surrogate__):
    
    # Trains the model
    def train(self, input_dict:dict, output_dict:dict) -> None:
        
        # Initialise model
        input_size = self.get_input_size()
        self.model = RBF(d0=input_size)
        
        # Create mappers
        self.input_mapper = MapperDict(input_dict)
        self.output_mapper = MapperDict(output_dict)
        
        # Map the data dictionaries
        norm_input_dict = self.input_mapper.map(input_dict)
        norm_output_dict = self.output_mapper.map(output_dict)
        
        # Convert the dictionaries to matrices
        norm_input_grid = np.array(dict_to_grid(norm_input_dict))
        norm_output_grid = np.array(dict_to_grid(norm_output_dict))
        
        # Train
        self.model.set_training_values(norm_input_grid, norm_output_grid)
        self.model.train()
        
    # Returns the predictions
    def predict(self, input_dict:dict) -> dict:
    
        # Map and convert input data
        norm_input_dict = self.input_mapper.map(input_dict)
        norm_input_grid = np.array(dict_to_grid(norm_input_dict))
        
        # Gets the prediction
        norm_output_grid = self.model.predict_values(norm_input_grid)
        output_headers = self.output_mapper.get_headers()
        norm_output_dict = grid_to_dict(norm_output_grid, output_headers)
        
        # Unmap and return
        output_dict = self.output_mapper.unmap(norm_output_dict)
        return output_dict
