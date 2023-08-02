"""
 Title:         Material Modelling Surrogate Controller
 Description:   Connects the API to the rest of the modules
 Author:        Janzen Choi

"""

# Libraries
import numpy as np, random
from tabulate import tabulate
from mms.interface.converter import csv_to_dict, transpose
from mms.surrogates.__surrogate__ import get_surrogate
from mms.parameter import Parameter

# Controller class
class Controller:
    
    def __init__(self, output_path:str):
        """
        Initialises everything for the controller class
        
        Parameters:
        * `output_path`: The path to the outputs
        """
        self.output_path     = output_path
        self.raw_exp_dict    = {}
        self.input_exp_dict  = {}
        self.output_exp_dict = {}
        self.train_input     = []
        self.train_output    = []
        self.valid_input      = []
        self.valid_output     = []
    
    def read_data(self, data_path:str) -> None:
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_path`: The path to the CSV file storing the experimental data
        """
        self.raw_exp_dict = csv_to_dict(data_path)
        if list(self.raw_exp_dict.keys()) == []:
            raise ValueError("The CSV file contains no data!")
    
    def filter_data(self, param_name:str, value:float) -> None:
        """
        Removes data points with specific parameter values
        
        Parameters:
        * `param_name`: The name of the parameter
        * `value`:      The value of the parameter
        """
        
        # Create a new dictionary
        new_raw_exp_dict = {}
        for key in self.raw_exp_dict.keys():
            new_raw_exp_dict[key] = []
        
        # Populate new dictionary
        num_data = len(self.raw_exp_dict[list(self.raw_exp_dict.keys())[0]])
        for i in range(num_data):
            if self.raw_exp_dict[param_name][i] != value:
                for key in self.raw_exp_dict.keys():
                    new_raw_exp_dict[key].append(self.raw_exp_dict[key][i])
        
        # Replace dictionary
        self.raw_exp_dict = new_raw_exp_dict
    
    def __get_num_data__(self) -> int:
        """
        Gets the number of data (from the input dict)
        """
        return self.input_exp_dict[list(self.input_exp_dict.keys())[0]].get_num_data()
    
    def __get_param__(self, param_name:str, mappers:list=None, **kwargs) -> Parameter:
        """
        Adds a parameter as an input or output, internally
        
        Parameters:
        * `param_name`: The name of the output parameter
        * `mappers`: The ordered list of how the parameter will be mapped
        """
        if not param_name in self.raw_exp_dict.keys():
            raise ValueError(f"The parameter {param_name} does not exist in the CSV file!")
        value_list = self.raw_exp_dict[param_name]
        mappers = [] if mappers == None else mappers
        parameter = Parameter(param_name, value_list, mappers, **kwargs)
        return parameter
    
    def add_input(self, param_name:str, mappers:list=None, **kwargs) -> None:
        """
        Adds a parameter as an input
        
        Parameters:
        * `param_name`: The name of the input parameter
        * `mappers`: The ordered list of how the parameter will be mapped
        """
        parameter = self.__get_param__(param_name, mappers, **kwargs)
        self.input_exp_dict[param_name] = parameter
    
    def add_output(self, param_name:str, mappers:list=None, **kwargs) -> None:
        """
        Adds a parameter as an output
        
        Parameters:
        * `param_name`: The name of the output parameter
        * `mappers`: The ordered list of how the parameter will be mapped
        """
        parameter = self.__get_param__(param_name, mappers, **kwargs)
        self.output_exp_dict[param_name] = parameter
    
    def set_surrogate(self, surrogate_name:str, **kwargs) -> None:
        """
        Defines the surrogate model to be used
        
        Parameters:
        * `surrogate_name`: The name of the surrogate model
        """
        input_size = len(self.input_exp_dict.keys())
        output_size = len(self.output_exp_dict.keys())
        if input_size == 0:
            raise ValueError("There are no inputs defined!")
        if output_size == 0:
            raise ValueError("There are no outputs defined!")
        self.surrogate = get_surrogate(surrogate_name, input_size, output_size, self.output_path, **kwargs)
    
    def __get_data__(self, indexes:list, param_dict:dict, map:bool=True) -> list:
        """
        Gets the list of data points
        
        Parameters:
        * `indexes`:    The indexes of the data points
        * `param_dict`: Dictionary of parameters (i.e., input / output)
        * `map`:        Whether or not to map the data
        """
        mapped_grid = []
        for param_name in param_dict.keys():
            param = param_dict[param_name]
            value_list = param.get_and_remove_value_list(indexes)
            if map:
                value_list = param.map_values(value_list)
            mapped_grid.append(value_list)
        mapped_grid = transpose(mapped_grid)
        return mapped_grid
    
    def add_training_data(self, num_data:int) -> None:
        """
        Adds data for training
        
        Parameters:
        * `num_data`: The number of training data
        """
        random_indexes = random.sample(range(self.__get_num_data__()), num_data)
        self.train_input += self.__get_data__(random_indexes, self.input_exp_dict)
        self.train_output += self.__get_data__(random_indexes, self.output_exp_dict)
    
    def add_validation_data(self, num_data:int) -> None:
        """
        Adds data for validation
        
        Parameters:
        * `num_data`: The number of validation data
        """
        random_indexes = random.sample(range(self.__get_num_data__()), num_data)
        self.valid_input += self.__get_data__(random_indexes, self.input_exp_dict)
        self.valid_output += self.__get_data__(random_indexes, self.output_exp_dict)
        
    def train_surrogate(self, **kwargs) -> None:
        """
        Trains the model
        """
        self.surrogate.set_train_data(self.train_input, self.train_output)
        self.surrogate.set_valid_data(self.valid_input, self.valid_output)
        self.surrogate.train(**kwargs)

    def __unmap_output__(self, output_grid:list) -> None:
        """
        Unmaps the outputs
        
        Parameters:
        * `output_grid`: The mapped outputs
        """
        unmapped_output_grid = []
        output_grid = transpose(output_grid)
        output_key_list = list(self.output_exp_dict.keys())
        for i in range(len(output_grid)):
            param = self.output_exp_dict[output_key_list[i]]
            unmapped_output_list = param.unmap_values(output_grid[i])
            unmapped_output_grid.append(unmapped_output_list)
        unmapped_output_grid = transpose(unmapped_output_grid)
        return unmapped_output_grid

    def validate_surrogate(self, **kwargs) -> None:
        """
        Validates the model
        """

        # Get the outputs
        prd_output = self.surrogate.predict(**kwargs)
        prd_output = transpose(self.__unmap_output__(prd_output))
        valid_output = transpose(self.__unmap_output__(self.valid_output))
        
        # Assess the results
        for i in range(len(prd_output)):
            
            # Initialise error storage
            error_list = []
            summary_grid = [["index", "exp", "prd", "RE"]]
            for j in range(len(self.valid_output)):
                
                # Calculate error
                exp_value = valid_output[i][j]
                prd_value = prd_output[i][j]
                error = round(100 * abs(exp_value - prd_value) / exp_value, 2)
                error_list.append(error)
                summary_grid.append([j+1, "{:0.3}".format(exp_value), "{:0.3}".format(prd_value), f"{error}%"])
            
            # Print table and average error
            print(tabulate(summary_grid, headers="firstrow", tablefmt="grid"))
            avg_error = round(np.average(error_list), 2)
            header = list(self.output_exp_dict.keys())[i]
            print(f"Average error for {header} = {avg_error}%")
