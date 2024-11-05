"""
 Title:         Material Modelling Surrogate Controller
 Description:   Connects the Interface to the rest of the modules
 Author:        Janzen Choi

"""

# Libraries
import math, numpy as np, random
import matplotlib.pyplot as plt
from tabulate import tabulate
from copy import deepcopy
from mms.helper.general import transpose
from mms.helper.io import get_file_path_exists, csv_to_dict, dict_to_csv
from mms.surrogates.__surrogate__ import create_surrogate
from mms.surrogator.parameter import Parameter

# Controller class
class Controller:
    
    def __init__(self):
        """
        Initialises everything for the controller class
        """
        self.raw_exp_dict    = {}
        self.input_exp_dict  = {}
        self.output_exp_dict = {}
        self.train_input     = []
        self.train_output    = []
        self.valid_input     = []
        self.valid_output    = []
    
    def read_data(self, data_path:str) -> None:
        """
        Reads experimental data from CSV files
        
        Parameters:
        * `data_path`: The path to the CSV file storing the experimental data
        """
        self.raw_exp_dict = csv_to_dict(data_path)
        if list(self.raw_exp_dict.keys()) == []:
            raise ValueError("The CSV file contains no data!")
    
    def remove_data(self, param_name:str, value:float) -> None:
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
    
    def add_training_data(self, ratio:int) -> None:
        """
        Adds data for training
        
        Parameters:
        * `ratio`: The ratio of the training data
        """
        num_data = round(self.__get_num_data__() * ratio)
        random_indexes = random.sample(range(self.__get_num_data__()), num_data)
        self.train_input += self.__get_data__(random_indexes, self.input_exp_dict)
        self.train_output += self.__get_data__(random_indexes, self.output_exp_dict)
    
    def add_validation_data(self, ratio:int=None) -> None:
        """
        Adds data for validation
        
        Parameters:
        * `ratio`: The ratio of the validation data
        """
        if ratio != None:
            num_data = round(self.__get_num_data__() * ratio)
            random_indexes = random.sample(range(self.__get_num_data__()), num_data)
        else:
            random_indexes = list(range(self.__get_num_data__()))
        self.valid_input += self.__get_data__(random_indexes, self.input_exp_dict)
        self.valid_output += self.__get_data__(random_indexes, self.output_exp_dict)
    
    def get_validation_data(self) -> None:
        """
        Gets validation data from the training data;
        requires the surrogate model to define 'get_valid_indexes'
        """
        valid_indexes = self.surrogate.get_valid_indexes()
        self.valid_input = [self.train_input[i] for i in range(len(self.train_input)) if i in valid_indexes]
        self.valid_output = [self.train_output[i] for i in range(len(self.train_output)) if i in valid_indexes]

    def define_surrogate(self, surrogate_name:str, device_name:str=None, **kwargs) -> None:
        """
        Defines the surrogate

        Parameters:
        * `surrogate_name`: The name of the surrogate
        * `device_name`:    The name of the device; uses default device if unspecified
        """
        
        # Get and check sizes
        input_size = len(self.input_exp_dict.keys())
        output_size = len(self.output_exp_dict.keys())
        if input_size == 0:
            raise ValueError("There are no inputs defined!")
        if output_size == 0:
            raise ValueError("There are no outputs defined!")
        
        # Define the surrogate
        self.surrogate = create_surrogate(surrogate_name, device_name, input_size, output_size,
                                          self.input_exp_dict, self.output_exp_dict, **kwargs)

    def train_surrogate(self, **kwargs) -> None:
        """
        Trains the surrogate model
        """
        self.surrogate.train(self.train_input, self.train_output,
                             self.valid_input, self.valid_output,
                             **kwargs)
    
    def plot_loss_history(self, loss_path:str) -> None:
        """
        Plots the loss history
        
        Parameters:
        * `loss_path`:  The path to output the loss plot
        """
        self.surrogate.plot_loss_history(loss_path)

    def save(self, model_path:str) -> None:
        """
        Saves the surrogate model
        
        Parameters:
        * `model_path`: The path to the surrogate model (excluding extension)
        """
        self.surrogate.save(model_path)

    def export_maps(self, map_path:str) -> None:
        """
        Exports information about the maps

        Parameters:
        * `map_path`: The path to save the mapping information
        """

        # Get parameter information
        param_names = list(self.input_exp_dict.keys()) + list(self.output_exp_dict.keys())
        param_list = list(self.input_exp_dict.values()) + list(self.output_exp_dict.values())

        # Get information about mappers
        combined_mapper_dict_list = []
        for param in param_list:
            combined_mapper_dict = {}
            for mapper in param.get_mappers():
                mapper_dict = mapper.get_info()
                combined_mapper_dict.update(mapper_dict)
            combined_mapper_dict_list.append(combined_mapper_dict)

        # Initialise summary
        param_types = ["input"]*len(self.input_exp_dict) + ["output"]*len(self.output_exp_dict)
        mapper_summary = {"param_type": param_types, "param_name": param_names}

        # Combine mapper information
        mapper_keys = [list(combined_mapper_dict.keys()) for combined_mapper_dict in combined_mapper_dict_list]
        mapper_keys = list(set([item for sublist in mapper_keys for item in sublist]))
        for mapper_key in mapper_keys:
            mapper_summary[mapper_key] = []
            for combined_mapper_dict in combined_mapper_dict_list:
                if mapper_key in combined_mapper_dict.keys():
                    mapper_summary[mapper_key].append(combined_mapper_dict[mapper_key])
                else:
                    mapper_summary[mapper_key].append("")

        # Write to CSV file
        dict_to_csv(mapper_summary, f"{map_path}.csv")

    def print_validation(self, use_log:bool=False, print_table:bool=False) -> None:
        """
        Prints a summary of the validation data
        
        Parameters:
        * `use_log`:     Whether to use log when checking the relative error
        * `print_table`: Whether to print out the table
        """
        
        # Initialise function to account for the use_log boolean
        transform = lambda x : math.log10(x) if use_log else x
        use_log_str = "logged " if use_log else ""

        # Get the outputs
        prd_output = self.surrogate.predict(self.valid_input)
        prd_output = transpose(self.__unmap_params__(prd_output, self.output_exp_dict))
        valid_output = transpose(self.__unmap_params__(self.valid_output, self.output_exp_dict))
        
        # Assess and print out the results
        print()
        for i in range(len(prd_output)):
            
            # Initialise error storage
            error_list = []
            summary_grid = [["index", "exp", "prd", "RE"]]
            average_exp = np.average([abs(transform(valid_output[i][j])) for j in range(len(self.valid_output))])
            
            # Iterate through predictions
            for j in range(len(self.valid_output)):
                
                # Calculate error
                exp_value = transform(valid_output[i][j])
                prd_value = transform(prd_output[i][j])
                error = round(100 * abs(exp_value - prd_value) / average_exp, 2)
                error_list.append(error)
                summary_grid.append([j+1, "{:0.3}".format(exp_value), "{:0.3}".format(prd_value), f"{error}%"])
            
            # Print table and average error
            if print_table:
                print(tabulate(summary_grid, headers="firstrow", tablefmt="grid"))
            avg_error = round(np.average(error_list), 2)
            header = list(self.output_exp_dict.keys())[i]
            print(f"Average {use_log_str}error for {header} = {avg_error}%")
        print()

    def plot_validation(self, headers:list, use_log:bool, plot_path:str, label:str="") -> None:
        """
        Creates plots of the validation predictions
        
        Parameters:
        * `headers`:    for the outputs to plot
        * `use_log`:    to use log when plotting the values
        * `plot_path`:  path name of the plot (without the extension)
        * `label`:      label for the plot
        """

        # Get all the training and validation data
        train_sim_dict, train_prd_dict = self.__get_predictions__(self.train_input, self.train_output)
        valid_sim_dict, valid_prd_dict = self.__get_predictions__(self.valid_input, self.valid_output)
        
        # Flatten data
        flatten = lambda nested_list : [item for sublist in nested_list for item in sublist]
        train_sim_list = flatten([train_sim_dict[header] for header in headers if header in train_sim_dict.keys()])
        train_prd_list = flatten([train_prd_dict[header] for header in headers if header in train_prd_dict.keys()])
        valid_sim_list = flatten([valid_sim_dict[header] for header in headers if header in valid_sim_dict.keys()])
        valid_prd_list = flatten([valid_prd_dict[header] for header in headers if header in valid_prd_dict.keys()])

        # Plot the training and validation data
        plt.figure(figsize=(5,5))
        plt.scatter(train_sim_list, train_prd_list, c="green")
        plt.scatter(valid_sim_list, valid_prd_list, c="red")
        
        # Define limits and plot the line
        all_data = train_sim_list + train_prd_list + valid_sim_list + valid_prd_list
        min_limit = min(all_data)
        max_limit = max(all_data)
        line_list = [min_limit, max_limit]
        plt.plot(line_list, line_list, linestyle="--", c="black")

        # Add 'non-conservative' region
        triangle_vertices = np.array([[min_limit, min_limit], [max_limit, min_limit], [max_limit, max_limit]])
        plt.gca().fill(triangle_vertices[:, 0], triangle_vertices[:, 1], color="gray", alpha=0.3)
        plt.text(max_limit-0.48*(max_limit-min_limit), min_limit+0.05*(max_limit-min_limit), "Non-conservative", fontsize=10, color="black")
        
        # Define legend
        handles = [plt.scatter([], [], color="green", label="Training"), plt.scatter([], [], color="red", label="Validation")]
        legend = plt.legend(handles=handles, framealpha=1, edgecolor="black", fancybox=True, facecolor="white", fontsize=12, loc="upper left")
        plt.gca().add_artist(legend)

        # Format scale, limits, and labels
        if use_log:
            plt.xscale("log")
            plt.yscale("log")
        plt.xlim(min_limit, max_limit)
        plt.ylim(min_limit, max_limit)
        plt.xlabel(f"Simulated {label}", fontsize=15)
        plt.ylabel(f"Predicted {label}", fontsize=15)
        
        # Format position and add grid
        plt.gca().set_position([0.17, 0.12, 0.75, 0.75])
        plt.gca().grid(which="major", axis="both", color="SlateGray", linewidth=1, linestyle=":")
        
        # Save and clear plot
        plot_path = get_file_path_exists(plot_path, "png")
        plt.savefig(plot_path)
        plt.clf()

    def export_validation(self, export_path:str="prd_data") -> None:
        """
        Exports the validation predictions to a CSV file
        
        Parameters:
        * `export_path`: The path to the export file (without the extension)
        """
        
        # Unmap inputs and convert to dictionaries
        input_headers = list(self.input_exp_dict.keys())
        valid_input = transpose(self.__unmap_params__(self.valid_input, self.input_exp_dict))
        valid_input_dict = {key: value for key, value in zip(input_headers, valid_input)}
        
        # Get the outputs
        prd_output = self.surrogate.predict(self.valid_input)
        prd_output = transpose(self.__unmap_params__(prd_output, self.output_exp_dict))
        valid_output = transpose(self.__unmap_params__(self.valid_output, self.output_exp_dict))
        
        # Convert outputs to dictionaries
        output_headers = list(self.output_exp_dict.keys())
        valid_output_dict = {f"valid_{key}": value for key, value in zip(output_headers, valid_output)}
        prd_output_dict = {f"prd_{key}": value for key, value in zip(output_headers, prd_output)}
        
        # Prepare the export dictionary
        export_dict = deepcopy(valid_input_dict)
        export_dict.update(valid_output_dict)
        export_dict.update(prd_output_dict)
        
        # Write dictionary to CSV path
        dict_to_csv(export_dict, f"{export_path}.csv")

    def __get_predictions__(self, input_list:list, output_list:list) -> tuple:
        """
        Gets the predictions given a list of inputs and outputs

        Parameters:
        * `input_list`:  List of inputs
        * `output_list`: List of outputs

        Returns the dictionaries of simulated (training / validation) and predicted (surrogate) data
        """

        # Get the outputs
        prd_output = self.surrogate.predict(input_list)
        prd_output = transpose(self.__unmap_params__(prd_output, self.output_exp_dict))
        sim_output = transpose(self.__unmap_params__(output_list, self.output_exp_dict))
        
        # Convert outputs to dictionaries
        output_headers = list(self.output_exp_dict.keys())
        sim_output_dict = {key: value for key, value in zip(output_headers, sim_output)}
        prd_output_dict = {key: value for key, value in zip(output_headers, prd_output)}
        
        # Returns the data
        return sim_output_dict, prd_output_dict

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
    
    def __unmap_params__(self, param_grid:list, param_dict:dict) -> None:
        """
        Unmaps the parameters (i.e., inputs/outputs)
        
        Parameters:
        * `param_grid`: The mapped values as a list of lists
        * `param_dict`: The dictionary of parameters
        """
        unmapped_param_grid = []
        param_grid = transpose(param_grid)
        key_list = list(param_dict.keys())
        for i in range(len(param_grid)):
            param = param_dict[key_list[i]]
            unmapped_param_list = param.unmap_values(param_grid[i])
            unmapped_param_grid.append(unmapped_param_list)
        unmapped_param_grid = transpose(unmapped_param_grid)
        return unmapped_param_grid
