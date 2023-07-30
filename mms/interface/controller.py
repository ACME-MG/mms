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
    
    def __init__(self):
        """
        Initialises everything for the controller class
        """
        self.raw_exp_dict = {}
        self.input_exp_dict = {}
        self.output_exp_dict = {}
    
    def read_data(self, data_path:str) -> None:
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_path`: The path to the CSV file storing the experimental data
        """
        self.raw_exp_dict = csv_to_dict(data_path)
        if list(self.raw_exp_dict.keys()) == []:
            raise ValueError("The CSV file contains no data!")
        self.total_data = len(self.raw_exp_dict[list(self.raw_exp_dict.keys())[0]])
    
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
        self.surrogate = get_surrogate(surrogate_name, input_size, output_size, **kwargs)
    
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
            value_list = param.get_value_list()
            value_list = [value_list[i] for i in indexes]
            if map:
                value_list = param.map_values(value_list)
            mapped_grid.append(value_list)
        mapped_grid = transpose(mapped_grid)
        return mapped_grid
    
    def train_surrogate(self, num_data:int, **kwargs) -> None:
        """
        Trains the model
        
        Parameters:
        * `num_data`: The number of training data
        """
        random_indexes = random.sample(range(self.total_data), num_data)
        input_grid = self.__get_data__(random_indexes, self.input_exp_dict)
        output_grid = self.__get_data__(random_indexes, self.output_exp_dict)
        self.surrogate.train(input_grid, output_grid, **kwargs)
    
    def test_surrogate(self, num_data:int, **kwargs) -> None:
        """
        Tests the model
        
        Parameters:
        * `num_data`: The number of training data
        """
        
        # Get the data
        random_indexes = random.sample(range(self.total_data), num_data)
        input_grid = self.__get_data__(random_indexes, self.input_exp_dict)
        output_grid = self.__get_data__(random_indexes, self.output_exp_dict, map=False)

        # Get the predictions
        prd_output_grid = self.surrogate.predict(input_grid)
        prd_output_grid = transpose(prd_output_grid)

        # Unmap the predictions
        unmapped_prd_output_grid = []
        output_key_list = list(self.output_exp_dict.keys())
        for i in range(len(prd_output_grid)):
            param = self.output_exp_dict[output_key_list[i]]
            prd_output_list = prd_output_grid[i]
            unmapped_prd_output_list = param.unmap_values(prd_output_list)
            unmapped_prd_output_grid.append(unmapped_prd_output_list)
        
        # Assess the results
        output_grid = transpose(output_grid)
        for i in range(len(output_grid)):
            
            # Initialise error storage
            error_list = []
            summary_grid = [["index", "exp", "prd", "RE"]]
            for j in range(num_data):
                
                # Calculate error
                exp_value = output_grid[i][j]
                prd_value = unmapped_prd_output_grid[i][j]
                error = round(100 * abs(exp_value - prd_value) / exp_value, 2)
                error_list.append(error)
                summary_grid.append([j+1, "{:0.3}".format(exp_value), "{:0.3}".format(prd_value), f"{error}%"])
            
            # Print table and average error
            print(tabulate(summary_grid, headers="firstrow", tablefmt="grid"))
            avg_error = round(np.average(error_list), 2)
            header = list(self.output_exp_dict.keys())[i]
            print(f"Average error for {header} = {avg_error}%")
