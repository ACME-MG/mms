"""
 Title:         Reader
 Description:   For reading experimental data 
 Author:        Janzen Choi

"""

# Libraries
import random
from mms.io.converter import csv_to_dict

# Reader class
class Reader:
    
    def __init__(self, data_path:str, input_names:list, output_names:list):
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_path`:    The path to the CSV file storing the experimental data
        * `input_names`:  The list of the input names
        * `output_names`: The list of the output names
        """
        
        # Gets the data
        data_dict = csv_to_dict(data_path)
        self.total_data = len(data_dict[list(data_dict.keys())[0]])
        
        # Extract inputs
        self.input_dict = {}
        for input_name in input_names:
            self.input_dict[input_name] = data_dict[input_name]
        
        # Extract outputs
        self.output_dict = {}
        for output_name in output_names:
            self.output_dict[output_name] = data_dict[output_name]
            
    def get_total_data(self) -> int:
        """
        Gets the total number of data points read in
        """
        return self.total_data

    def get_data(self, num_data:int) -> tuple:
        """
        Gets a number of input-output pairs specified by the user
        
        Parameters:
        * `num_data`: The number of input-output pairs
        
        Returns the a list of the inputs and a list of the outputs
        """
        
        # Check that the number of points requested is appropriate and get indexes
        if num_data > self.total_data:
            raise ValueError("The number of data points requested is greater than read in!")
        random_indexes = random.sample(range(self.total_data), num_data)
        
        # Get inputs
        input_dict = {}
        for header in self.input_dict.keys():
            input_dict[header] = [self.input_dict[header][i] for i in random_indexes]
        
        # Get outputs
        output_dict = {}
        for header in self.output_dict.keys():
            output_dict[header] = [self.output_dict[header][i] for i in random_indexes]
        
        # Return the data
        return input_dict, output_dict
