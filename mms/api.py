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
            message += " ..."
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
    
    def add_training_data(self, num_data:int) -> None:
        """
        Adds data for training
        
        Parameters:
        * `num_data`: The number of training data
        """
        self.__print__(f"Adding {num_data} data points to the training dataset")
        self.__controller__.add_training_data(num_data)
    
    def add_validation_data(self, num_data:int) -> None:
        """
        Adds data for validation
        
        Parameters:
        * `num_data`: The number of validation data
        """
        self.__print__(f"Adding {num_data} data points to the validation dataset")
        self.__controller__.add_validation_data(num_data)
    
    def train(self, epochs:int, batch_size:int, verbose:bool=False) -> None:
        """
        Trains the model
        
        Parameters:
        * `epochs`:     The number of epochs
        * `batch_size`: The size of each batch
        * `verbose`:    Whether to display training updates or not
        """
        self.__print__(f"Training the model with a batch size of {batch_size} for {epochs} epochs")
        self.__controller__.train_surrogate(epochs, batch_size, verbose)

    def plot_loss_history(self, loss_file:str="loss_history") -> None:
        """
        Plots the loss history after training the surrogate model
        
        Parameters:
        * `loss_file`:  The file to display the loss
        """
        self.__print__(f"Plotting the loss history to {loss_file}")
        loss_path = self.__get_output__(loss_file)
        self.__controller__.plot_loss_history(loss_path)

    def save(self, model_file:str="surrogate") -> None:
        """
        Saves the surrogate model
        
        Parameters:
        * `model_file`: The file to save the surrogate model
        """
        self.__print__(f"Saving the surrogate model to {model_file}")
        model_path = self.__get_output__(model_file)
        self.__controller__.plot_loss_history(model_path)

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

    def plot_validation(self, use_log:bool=False, plot_file:str="prd_plot") -> None:
        """
        Creates plots of the validation predictions
        
        Parameters:
        * `use_log`:    Whether to use log when plotting the values
        * `plot_file`:  The file name of the plot (without the extension)
        """
        self.__print__(f"Plotting the validation predictions to {plot_file}_*")
        plot_path = self.__get_output__(plot_file)
        self.__controller__.plot_validation(use_log, plot_path)

    def export_validation(self, export_file:str="prd_data") -> None:
        """
        Exports the validation predictions to a CSV file
        
        Parameters:
        * `export_file`: The file name (without the extension)
        """
        self.__print__(f"Exporting the validation predictions to {export_file}")
        export_path = self.__get_output__(export_file)
        self.__controller__.export_validation(export_path)

# For safely making a directory
def safe_mkdir(dir_path:str) -> None:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
