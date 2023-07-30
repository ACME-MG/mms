"""
 Title:         Material Modelling Surrogate API
 Description:   API for calibrating creep models
 Author:        Janzen Choi

"""

# Libraries
import os, re, time
from mms.interface.controller import Controller

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
        
        # Define controller
        self.__controller__ = Controller()
        
        # Create directories
        safe_mkdir(output_path)
        safe_mkdir(self.__output_path__)
    
    def __print__(self, message:str, add_index:bool=True) -> None:
        """
        Displays a message before running the command (for internal use only)
        
        Parameters:
        * `message`:   the message to be displayed
        * `add_index`: if true, adds a number at the start of the message
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

    def read_data(self, data_file:str) -> None:
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_file`:    The path to the CSV file storing the experimental data
        """
        self.__print__(f"Reading experimental data from {data_file} ...")
        data_path = self.__get_input__(data_file)
        self.__controller__.read_data(data_path)
    
    def add_input(self, param_name:str, mappers:list=None, **kwargs) -> None:
        """
        Adds a parameter as an input
        
        Parameters:
        * `param_name`: The name of the input parameter
        * `mappers`: The ordered list of how the parameter will be mapped
        """
        mapper_str = "" if mappers == None else f"({', '.join(mappers)})"
        self.__print__(f"Adding input {param_name} {mapper_str}")
        self.__controller__.add_input(param_name, mappers, **kwargs)
    
    def add_output(self, param_name:str, mappers:list=None, **kwargs) -> None:
        """
        Adds a parameter as an output
        
        Parameters:
        * `param_name`: The name of the output parameter
        * `mappers`: The ordered list of how the parameter will be mapped
        """
        mapper_str = "" if mappers == None else f"({', '.join(mappers)})"
        self.__print__(f"Adding output {param_name} {mapper_str}")
        self.__controller__.add_output(param_name, mappers, **kwargs)
    
    def set_surrogate(self, surrogate_name:str, **kwargs) -> None:
        """
        Defines the surrogate model to be used
        
        Parameters:
        * `surrogate_name`: The name of the surrogate model
        """
        self.__print__(f"Defining the {surrogate_name} surrogate model ...")
        self.__controller__.set_surrogate(surrogate_name, **kwargs)
    
    def train(self, num_data:int, **kwargs) -> None:
        """
        Trains the model
        
        Parameters:
        * `num_data`: The number of training data
        """
        self.__print__(f"Training the model with {num_data} data points ...")
        self.__controller__.train_surrogate(num_data, **kwargs)

    def test(self, num_data:int, **kwargs) -> None:
        """
        Tests the trained model
        
        Parameters:
        * `num_data`: The number of testing data
        """
        self.__print__(f"Using the model to predict {num_data} data points ...")
        self.__controller__.test_surrogate(num_data, **kwargs)

# For safely making a directory
def safe_mkdir(dir_path:str) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)