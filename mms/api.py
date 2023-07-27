"""
 Title:         Material Modelling Surrogate API
 Description:   API for calibrating creep models
 Author:        Janzen Choi

"""

# Libraries
import os, re, time, numpy as np
from tabulate import tabulate
from mms.interface.reader import Reader
from mms.surrogates.__surrogate__ import get_surrogate

# API Class
class API:

    def __init__(self, title:str="", input_path:str="./data", output_path:str="./results", output_here:bool=False):
        """
        Class to interact with the optimisation code
        
        Parameters:
        * `title`:       Title of the output folder
        * `input_path`:  Path to the input folder
        * `output_path`: Path to the output folder
        * `verbose`:     If true, outputs messages for each function call
        * `output_here`: If true, just dumps the output in ths executing directory
        """
        
        # Print starting message
        self.__print_index__ = 0
        time_str = time.strftime("%A, %D, %H:%M:%S", time.localtime())
        self.__print__(f"\n  Starting on {time_str}\n", add_index=False)
                
        # Get start time
        self.__start_time__ = time.time()
        time_stamp = time.strftime("%y%m%d%H%M%S", time.localtime(self.__start_time__))
        
        # Define input and output
        self.__input_path__ = input_path
        self.__get_input__  = lambda x : f"{self.__input_path__}/{x}"
        title = "" if title == "" else f"_{title}"
        title = re.sub(r"[^a-zA-Z0-9_]", "", title.replace(" ", "_"))
        self.__output_dir__ = "." if output_here else time_stamp
        self.__output_path__ = "." if output_here else f"{output_path}/{self.__output_dir__}{title}"
        self.__get_output__ = lambda x : f"{self.__output_path__}/{x}"
        
        # Create directories
        safe_mkdir(output_path)
        safe_mkdir(self.__output_path__)
    
    def __print__(self, message:str, add_index:bool=True) -> None:
        """
        Displays a message before running the command (for internal use only)
        
        Parameters:
        * `message`:   the message to be displayed
        * `add_index`: if true, adds a number at the start of the message
        * `sub_index`: if true, adds a number as a decimal
        """
        if add_index:
            self.__print_index__ += 1
            print(f"   {self.__print_index__})\t", end="")
        print(message)

    def __del__(self):
        """
        Prints out the final message (for internal use only)
        """
        time_str = time.strftime("%A, %D, %H:%M:%S", time.localtime())
        duration = round(time.time() - self.__start_time__)
        self.__print__(f"\n  Finished on {time_str} in {duration}s\n", add_index=False)

    def read_data(self, data_file:str, input_names:list, output_names:list) -> None:
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_file`:    The path to the CSV file storing the experimental data
        * `input_names`:  The list of the input names
        * `output_names`: The list of the output names
        """
        self.__print__(f"Reading experimental data from {data_file} ...")
        self.input_names = input_names
        self.output_names = output_names
        data_path = self.__get_input__(data_file)
        self.reader = Reader(data_path, input_names, output_names)
    
    def define_surrogate(self, surrogate_name:str, **kwargs) -> None:
        """
        Defines the surrogate model to be used
        
        Parameters:
        * `surrogate_name`: The name of the surrogate model
        """
        self.__print__(f"Defining the {surrogate_name} surrogate model ...")
        self.surrogate = get_surrogate(surrogate_name, len(self.input_names), len(self.output_names), **kwargs)
    
    def train(self, num_data:int, **kwargs) -> None:
        """
        Trains the model
        
        Parameters:
        * `num_data`: The number of training data
        """
        self.__print__(f"Training the model with {num_data} data points ...")
        input_dict, output_dict = self.reader.get_data(num_data)
        self.surrogate.train(input_dict, output_dict, **kwargs)

    def predict(self, num_data:int, **kwargs) -> None:
        """
        Tests the trained model
        
        Parameters:
        * `num_data`: The number of testing data
        """
        self.__print__(f"Using the model to predict {num_data} data points ...")
        input_dict, output_dict = self.reader.get_data(num_data)
        prd_output_dict = self.surrogate.predict(input_dict, **kwargs)
        for header in output_dict.keys():
            print_table(header, output_dict, prd_output_dict)

# Prints a table given the experimental and predicted values
def print_table(header:str, exp_dict:dict, prd_dict:dict):
    print(f"Summary for {header}")
    error_list = []
    summary_grid = [["index", "exp", "prd", "RE"]]
    for i in range(len(exp_dict[header])):
        exp_value = exp_dict[header][i]
        prd_value = prd_dict[header][i]
        error = round(100 * abs(exp_value - prd_value) / exp_value, 2)
        error_list.append(error)
        summary_grid.append([i+1, "{:0.3}".format(exp_value), "{:0.3}".format(prd_value), f"{error}%"])
    print(tabulate(summary_grid, headers="firstrow", tablefmt="grid"))
    avg_error = round(np.average(error_list), 2)
    print(f"Average error for {header} = {avg_error}%")

# For safely making a directory
def safe_mkdir(dir_path:str) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)