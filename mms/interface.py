"""
 Title:         Material Modelling Surrogate Interface
 Description:   Interface for calibrating creep models
 Author:        Janzen Choi

"""

# Libraries
import inspect, os, re, time
from mms.surrogator.controller import Controller

# Interface Class
class Interface:

    def __init__(self, title:str="", input_path:str="./data", output_path:str="./results",
                 verbose:bool=True, output_here:bool=False):
        """
        Class to interact with the optimisation code
        
        Parameters:
        * `title`:       Title of the output folder
        * `input_path`:  Path to the input folder
        * `output_path`: Path to the output folder
        * `verbose`:     If true, outputs messages for each function call
        * `output_here`: If true, just dumps the output in ths executing directory
        """
        
        # Initialise internal variables
        self.__controller__  = Controller()
        self.__print_index__ = 0
        self.__verbose__     = verbose
        
        # Starting code
        time_str = time.strftime("%A, %D, %H:%M:%S", time.localtime())
        self.__print__(f"\n  Starting on {time_str}\n", add_index=False)
        self.__start_time__ = time.time()
        time_stamp = time.strftime("%y%m%d%H%M%S", time.localtime(self.__start_time__))
        
        # Define input and output
        self.__input_path__ = input_path
        file_path = inspect.currentframe().f_back.f_code.co_filename
        self.__file_name__ = file_path.split("/")[-1].replace(".py","")
        title = f"_{self.__file_name__}" if title == "" else f"_{title}"
        title = re.sub(r"[^a-zA-Z0-9_]", "", title.replace(" ", "_"))
        self.__output_dir__ = "." if output_here else time_stamp
        self.__output_path__ = "." if output_here else f"{output_path}/{self.__output_dir__}{title}"
        
        # Define input / output functions
        self.__get_input__  = lambda x : f"{self.__input_path__}/{x}"
        self.__get_output__ = lambda x : f"{self.__output_path__}/{x}"
        
        # Create directories
        if not output_here:
            safe_mkdir(output_path)
            safe_mkdir(self.__output_path__)

    def read_data(self, data_file:str) -> None:
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_file`:    The path to the CSV file storing the experimental data
        """
        self.__print__(f"Reading experimental data from {data_file}")
        data_path = self.__get_input__(data_file)
        self.__controller__.read_data(data_path)
    
    def remove_data(self, param_name:str, value:float) -> None:
        """
        Removes data points with specific parameter values
        
        Parameters:
        * `param_name`: The name of the parameter
        * `value`:      The value of the parameter
        """
        self.__print__(f"Removing datapoints with {param_name} = {value}")
        self.__controller__.remove_data(param_name, value)
    
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
    
    def add_training_data(self, ratio:float=1.0) -> None:
        """
        Adds data for training
        
        Parameters:
        * `ratio`: The ratio of the training data (0-1)
        """
        self.__print__(f"Adding {round(ratio*100)}% of the data points to the training dataset")
        self.__controller__.add_training_data(ratio)
    
    def add_validation_data(self, ratio:float=None) -> None:
        """
        Adds data for validation
        
        Parameters:
        * `ratio`: The ratio of the validation data (0-1);
                   if unspecifies, then it selects the remainder
        """
        if ratio == None:
            self.__print__(f"Adding the remaining data points to the validation dataset")
        else:
            self.__print__(f"Adding {round(ratio*100)}% of the data points to the validation dataset")
        self.__controller__.add_validation_data(ratio)
    
    def define(self, surrogate_name:str, **kwargs) -> None:
        """
        Defines the surrogate

        Parameters:
        * `surrogate_name`: The name of the surrogate
        """
        self.__print__(f"Defining the '{surrogate_name}' surrogate model")
        self.__controller__.define_surrogate(surrogate_name, **kwargs)

    def train(self, **kwargs) -> None:
        """
        Trains the model
        """
        self.__print__(f"Training the surrogate model")
        self.__controller__.train_surrogate(**kwargs)

    def plot_loss_history(self, loss_file:str="loss_history") -> None:
        """
        Plots the loss history after training the surrogate model
        
        Parameters:
        * `loss_file`:  The file to display the loss
        """
        self.__print__(f"Plotting the loss history to {loss_file}")
        loss_path = self.__get_output__(loss_file)
        self.__controller__.plot_loss_history(loss_path)

    def save(self, model_file:str=None) -> None:
        """
        Saves the surrogate model
        
        Parameters:
        * `model_file`: The file to save the surrogate model
        """
        model_file = model_file if model_file != None else self.__file_name__
        self.__print__(f"Saving the surrogate model to '{model_file}'")
        model_path = self.__get_output__(model_file)
        self.__controller__.save(model_path)

    def export_maps(self, map_file:str=None) -> None:
        """
        Exports information about the maps

        Parameters:
        * `map_file`: The file to save the mapping information
        """
        map_file = map_file if map_file != None else self.__file_name__
        self.__print__(f"Saving the mapping information to '{map_file}'")
        map_path = self.__get_output__(map_file)
        self.__controller__.export_maps(map_path)

    def get_validation_data(self) -> None:
        """
        Uses a subset of training data as validation data;
        requires the surrogate model to define 'get_valid_indexes'
        """
        self.__print__("Getting a subset of training data to use as validation data")
        self.__controller__.get_validation_data()

    def print_validation(self, use_log:bool=False, print_table:bool=False) -> None:
        """
        Prints a summary of the validation data
        
        Parameters:
        * `use_log`:     Whether to use log when checking the relative error
        * `print_table`: Whether to print out the table
        """
        use_log_str = "with log " if use_log else ""
        self.__print__(f"Summarising the validation data {use_log_str}...")
        self.__controller__.print_validation(use_log, print_table)

    def plot_validation(self, headers:list, use_log:bool=False, plot_file:str="prd_plot", label:str="") -> None:
        """
        Creates plots of the validation predictions
        
        Parameters:
        * `headers`:   Headers for the outputs to plot
        * `use_log`:   Whether to use log when plotting the values
        * `plot_path`: The path name of the plot (without the extension)
        * `label`:     The label for the plot
        """
        self.__print__(f"Plotting the validation predictions to {plot_file}.png")
        plot_path = self.__get_output__(plot_file)
        self.__controller__.plot_validation(headers, use_log, plot_path, label)

    def export_validation(self, export_file:str="prd_data") -> None:
        """
        Exports the validation predictions to a CSV file
        
        Parameters:
        * `export_file`: The file name (without the extension)
        """
        self.__print__(f"Exporting the validation predictions to {export_file}")
        export_path = self.__get_output__(export_file)
        self.__controller__.export_validation(export_path)

    def __print__(self, message:str, add_index:bool=True) -> None:
        """
        Displays a message before running the command (for internal use only)
        
        Parameters:
        * `message`:   the message to be displayed
        * `add_index`: if true, adds a number at the start of the message
        """
        if not add_index:
            print(message)
        if not self.__verbose__ or not add_index:
            return
        self.__print_index__ += 1
        print(f"   {self.__print_index__})\t{message} ...")

    def __del__(self):
        """
        Prints out the final message (for internal use only)
        """
        time_str = time.strftime("%A, %D, %H:%M:%S", time.localtime())
        duration = round(time.time() - self.__start_time__)
        duration_h = duration // 3600
        duration_m = (duration - duration_h * 3600) // 60
        duration_s = duration - duration_h * 3600 - duration_m * 60
        duration_str_list = [
            f"{duration_h} hours" if duration_h > 0 else "",
            f"{duration_m} mins" if duration_m > 0 else "",
            f"{duration_s} seconds" if duration_s > 0 else ""
        ]
        duration_str = ", ".join([d for d in duration_str_list if d != ""])
        duration_str = f"in {duration_str}" if duration_str != "" else ""
        self.__print__(f"\n  Finished on {time_str} {duration_str}\n", add_index=False)

def safe_mkdir(dir_path:str) -> None:
    """
    For safely making a directory

    Parameters:
    * `dir_path`: The path to the directory
    """
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
