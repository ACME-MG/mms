"""
 Title:         Material Modelling Surrogate
 Description:   For surrogate modelling creep models
 Author:        Janzen Choi

"""

# Libraries
import warnings; warnings.filterwarnings("ignore")
import torch, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import math, random, numpy as np
from copy import deepcopy

# Set tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

def main():
    """
    The main function
    """
    
    # Convert CSV file to dictionary
    data_dict = csv_to_dict("gb_data_gary.csv")
    
    # Only use realisation = 0
    data_dict = filter_data(data_dict, "realisation", 1)
    data_dict = filter_data(data_dict, "realisation", 2)
    data_dict = filter_data(data_dict, "realisation", 3)
    data_dict = filter_data(data_dict, "realisation", 4)
    
    # Adds the inputs
    input_list = [
        Parameter(data_dict, "D",           [LogMapper(), LinearMapper()]),
        Parameter(data_dict, "FN",          [LogMapper(), LinearMapper()]),
        Parameter(data_dict, "temperature", [LinearMapper()]),
        Parameter(data_dict, "stress",      [LinearMapper()]),
    ]
    
    # Adds the outputs
    output_list = [
        Parameter(data_dict, "time_secondary",   [LogMapper(), LinearMapper()]),
        # Parameter(data_dict, "strain_secondary", [LogMapper(), LinearMapper()]),
    ]
    
    # Gets the training and validation data
    train_inputs, train_outputs = get_data(1000, input_list, output_list)
    valid_inputs, valid_outputs = get_data(100, input_list, output_list)
    
    # Create surrogate model
    surrogate = Neural(len(input_list), len(output_list))
    
    # Train and plot results
    surrogate.train(train_inputs, train_outputs, valid_inputs, valid_outputs, epochs=5000, batch_size=32)
    surrogate.plot_loss_history("loss_plot.png")
    surrogate.print_validation(valid_inputs, valid_outputs, output_list)
    surrogate.plot_validation(valid_inputs, valid_outputs, output_list, "prd_plot.png")

# Parameter class
class Parameter:
    
    def __init__(self, data_dict:dict, param_name:str, mapper_list:list):
        """
        Constructor for the parameter object
        
        Parameters:
        * `data_dict`:   The dictionary storing the parameters
        * `param_name`:  The name of the parameter
        * `mapper_list`: The ordered list of the parameter mappers
        """
        
        # Initialise arguments
        self.param_name = param_name
        self.value_list = data_dict[param_name]
        self.mapper_list = mapper_list
        
        # Initialise mappers
        value_list = data_dict[param_name]
        for mapper in self.mapper_list:
            mapper.initialise(value_list)
            value_list = deepcopy(value_list)
            value_list = [mapper.map(value) for value in value_list]

    def map_values(self, value_list:list) -> list:
        """
        Maps a list of values
        
        Parameters:
        * `value_list`: The list of values to be mapped
        
        Returns the list of mapped values
        """
        value_list = deepcopy(value_list)
        for mapper in self.mapper_list:
            value_list = [mapper.map(value) for value in value_list]
        return value_list

    def unmap_values(self, value_list:list) -> list:
        """
        Unmaps a list of values
        
        Parameters:
        * `value_list`: The list of values to be unmapped
        
        Returns the list of unmapped values
        """
        value_list = deepcopy(value_list)
        for mapper in self.mapper_list[::-1]: # iterate reverse order
            value_list = [mapper.unmap(value) for value in value_list]
        return value_list

    def get_and_remove_value_list(self, indexes:list) -> list:
        """
        Gets the values of the parameter at specific indexes, then removes them
        """
        extracted_value_list = [self.value_list[i] for i in indexes]
        self.value_list = [self.value_list[i] for i in range(len(self.value_list)) if not i in indexes]
        return extracted_value_list

# Mapper class
class LinearMapper:

    def initialise(self, value_list:list) -> None:
        """
        Initialises the linear mapper
        
        Parameters:
        * `value_list`:  The list of values
        """
        self.in_l_bound = min(value_list)
        self.in_u_bound = max(value_list)
        self.out_l_bound = 0
        self.out_u_bound = 1
        self.distinct = self.in_l_bound == self.in_u_bound or self.out_l_bound == self.out_u_bound

    def map(self, value:float) -> float:
        """
        Maps a value
        
        Parameters:
        * `value`: The value to be mapped
        
        Returns the mapped value
        """
        if self.distinct:
            return value
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.in_l_bound) * factor + self.out_l_bound

    def unmap(self, value:float) -> float:
        """
        Unmaps a value
        
        Parameters:
        * `value`: The value to be unmapped
        
        Returns the unmapped value
        """
        if self.distinct:
            return value
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.out_l_bound) / factor + self.in_l_bound

# Log Mapper class
class LogMapper:

    def initialise(self, value_list:list) -> None:
        """
        Initialises the log mapper
        
        Parameters:
        * `value_list`:  The list of values
        """
        self.value_list = value_list # unused

    def map(self, value:float) -> float:
        """
        Maps a value
        
        Parameters:
        * `value`: The value to be mapped
        
        Returns the mapped value
        """
        if value < 0:
            return 0
        return math.log10(value)

    def unmap(self, value:float) -> float:
        """
        Unmaps a value
        
        Parameters:
        * `value`: The value to be unmapped
        
        Returns the unmapped value
        """
        return math.pow(10, value)

# Neural class
class Neural:
    
    def __init__(self, input_size:int, output_size:int):
        """
        Constructor for the neural network surrogate model
        
        Parameters:
        * `input_size`:  The number of inputs
        * `output_size`: The number of outputs
        """
        self.input_size = input_size
        self.output_size = output_size
    
    def train(self, train_input:list, train_output:list, valid_input:list, valid_output:list, epochs:int, batch_size:int) -> None:
        """
        Trains the model
        
        Parameters:
        * `train_input`:  Input data for training
        * `train_output`: Output data for training
        * `valid_input`:  Input data for validation
        * `valid_output`: Output data for validation
        * `epochs`: The number of epochs
        * `batch_size`: The size of each batch
        """
        
        # Initialise model
        self.model = CustomModel(self.input_size, self.output_size, [32, 16, 8])
        parameters = self.model.parameters()
        
        # Get the data and convert to tensors
        train_input_tensor = torch.tensor(train_input)
        train_output_tensor = torch.tensor(train_output)
        valid_input_tensor = torch.tensor(valid_input)
        valid_output_tensor = torch.tensor(valid_output)
        
        # Define optimisation
        learning_rate = 1e-3
        weight_decay  = 1e-7
        loss_function = torch.nn.MSELoss()
        optimiser     = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        
        # Initialise everything before training
        dataset = CustomDataset(train_input_tensor, train_output_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.train_loss_list = []
        self.valid_loss_list = []
        
        # Start training
        # torch.autograd.set_detect_anomaly(True) # for debugging
        for _ in range(epochs):
            
            # Completes training
            for batch_inputs, batch_outputs in dataloader:
                optimiser.zero_grad()
                outputs = self.model(batch_inputs)
                loss = loss_function(outputs, batch_outputs)
                loss.backward()
                optimiser.step()
            
            # Gets the training loss
            with torch.no_grad():
                prd_train_output_tensor = self.model(train_input_tensor)
                train_loss = loss_function(prd_train_output_tensor, train_output_tensor)
                self.train_loss_list.append(train_loss.item())
            
            # Gets the validation loss
            with torch.no_grad():
                prd_valid_output_tensor = self.model(valid_input_tensor)
                valid_loss = loss_function(prd_valid_output_tensor, valid_output_tensor)
                self.valid_loss_list.append(valid_loss.item())
    
    def plot_loss_history(self, loss_path:str):
        """
        Make plot of training and validation losses
        
        Parameters:
        * `loss_path`:   The path to output the loss
        """
        plt.title("Log_10 Loss vs Epochs")
        plt.xlabel("epochs")
        plt.ylabel("MSE Loss")
        plt.plot(list(range(len(self.train_loss_list))), self.train_loss_list, c="blue", label="training")
        plt.plot(list(range(len(self.valid_loss_list))), self.valid_loss_list, c="red", label="validation")
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{loss_path}.png")
        plt.clf()

    def predict(self, input_grid:list=None) -> list:
        """
        Returns a prediction
        
        Parameters:
        * `input_grid`: Input list of lists
        """
        input_tensor = torch.tensor(input_grid)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        output_grid = output_tensor.tolist()
        return output_grid

    def print_validation(self, valid_inputs:list, valid_outputs:list, output_list:list) -> None:
        """
        Prints a summary of the validation data
        
        Parameters:
        * `valid_inputs`:  The input values
        * `valid_outputs`: The output values
        * `output_list`:   The list of output parameters
        """

        # Get the outputs
        prd_outputs = self.predict(valid_inputs)
        prd_outputs = transpose(unmap_output(prd_outputs, output_list))
        valid_outputs = transpose(unmap_output(valid_outputs, output_list))
        
        # Assess the results
        for i in range(len(prd_outputs)):
            
            # Initialise error storage
            error_list = []
            for j in range(len(valid_outputs[i])):
                exp_value = math.log10(valid_outputs[i][j])
                prd_value = math.log10(prd_outputs[i][j])
                error = round(100 * abs(exp_value - prd_value) / abs(exp_value), 2)
                error_list.append(error)
            
            # Print average error
            avg_error = round(np.average(error_list), 2)
            print(f"Average log error = {avg_error}%")

    def plot_validation(self, valid_input:list, valid_output:list, output_list:list, plot_path:str="prd_plot") -> None:
        """
        Creates plots of the validation predictions
        
        Parameters:
        * `valid_input`:  Input data for validation
        * `valid_output`: Output data for validation
        * `output_list`:  The list of output parameters
        * `plot_path`:    The path name of the plot (without the extension)
        """
        
        # Get the outputs
        prd_output = self.predict(valid_input)
        prd_output = transpose(unmap_output(prd_output, output_list))
        valid_output = transpose(unmap_output(valid_output, output_list))
        
        # Create plots
        for i in range(len(prd_output)):
            
            # Get data points
            valid_list = valid_output[i]
            prd_list   = prd_output[i]
            line_list  = [min(valid_list + prd_list), max(valid_list + prd_list)]
            
            # Scale log
            plt.xscale("log")
            plt.yscale("log")
            
            # Plot the values with line
            plt.plot(line_list, line_list, linestyle='--', c="red")
            plt.scatter(valid_list, prd_list, c="grey")
            
            # Format, save, and clear plot
            plt.xlabel("validation")
            plt.ylabel("prediction")
            plt.gca().set_aspect("equal")
            plt.savefig(plot_path)
            plt.clf()

# Custom PyTorch model
class CustomModel(torch.nn.Module):
    
    def __init__(self, input_size:int, output_size:int, hidden_sizes:list):
        """
        Defines the structure of the neural network
        
        Parameters:
        * `input_size`: The number of inputs
        * `output_size`: The number of outputs
        * `hidden_sizes`: List of number of nodes for the hidden layers
        """
        super(CustomModel, self).__init__()
        self.input_layer   = torch.nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = [torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                              for i in range(len(hidden_sizes)-1)]
        self.output_layer  = torch.nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass
        
        Parameters:
        * `input_tensor`: PyTorch tensor for neural network input
        
        Returns the output of the network as a tensor
        """
        output_tensor = torch.relu(self.input_layer(input_tensor))
        for layer in self.hidden_layers:
            output_tensor = torch.relu(layer(output_tensor))
        output_tensor = self.output_layer(output_tensor)
        return output_tensor

# Custom Dataset
class CustomDataset(Dataset):
    
    def __init__(self, input_tensor:torch.Tensor, output_tensor:torch.Tensor):
        """
        Defines the input and output tensors
        
        Parameters:
        * `input_tensor`: PyTorch tensor for neural network input
        * `output_tensor`: PyTorch tensor for neural network output
        """
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def __len__(self) -> int:
        """
        Gets the length of the dataset
        """
        return len(self.input_tensor)

    def __getitem__(self, index:int) -> tuple:
        """
        Gets the item from the dataset with an index
        
        Parameters:
        * `index`: The index of the item
        """
        return self.input_tensor[index], self.output_tensor[index]

def unmap_output(output_grid:list, output_list:list) -> list:
    """
    Unmaps the outputs
    
    Parameters:
    * `output_grid`: The mapped outputs
    * `output_list`: The list of output parameters
    """
    unmapped_output_grid = []
    output_grid = transpose(output_grid)
    for i in range(len(output_grid)):
        unmapped_output_list = output_list[i].unmap_values(output_grid[i])
        unmapped_output_grid.append(unmapped_output_list)
    unmapped_output_grid = transpose(unmapped_output_grid)
    return unmapped_output_grid

def transpose(list_of_lists:list) -> list:
    """
    Transposes a 2D list of lists
    
    Parameters:
    * `list_of_lists`: A list of lists (i.e., a 2D grid)
    
    Returns the transposed list of lists
    """
    transposed = np.array(list_of_lists).T.tolist()
    return transposed

def get_data(num_data:int, input_list:list, output_list:list) -> tuple:
    """
    Gets data
    
    Parameters:
    * `num_data`:    The number of data desired
    * `input_list`:  List of input parameters
    * `output_list`: List of output parameters
    """
    total_data = len(input_list[0].value_list) # spaghetti
    random_indexes = random.sample(range(total_data), num_data)
    input_data = __get_data__(random_indexes, input_list)
    output_data = __get_data__(random_indexes, output_list)
    return input_data, output_data

def __get_data__(indexes:list, param_list:list) -> list:
    """
    Gets the list of data points
    
    Parameters:
    * `indexes`:    The indexes of the data points
    * `param_list`: List of parameters (i.e., input / output)
    """
    mapped_grid = []
    for param in param_list:
        value_list = param.get_and_remove_value_list(indexes)
        value_list = param.map_values(value_list)
        mapped_grid.append(value_list)
    mapped_grid = transpose(mapped_grid)
    return mapped_grid

def filter_data(data_dict:dict, param_name:str, value:float) -> dict:
    """
    Removes data points with specific parameter values
    
    Parameters:
    * `data_dict`:  The dictionary whose values are being filtered 
    * `param_name`: The name of the parameter
    * `value`:      The value of the parameter
    """
    
    # Create a new dictionary
    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = []
    
    # Populate new dictionary
    num_data = len(data_dict[list(data_dict.keys())[0]])
    for i in range(num_data):
        if data_dict[param_name][i] != value:
            for key in data_dict.keys():
                new_data_dict[key].append(data_dict[key][i])

    # Return
    return new_data_dict

def csv_to_dict(csv_path:str, delimeter:str=",") -> dict:
    """
    Converts a CSV file into a dictionary
    
    Parameters:
    * `csv_path`:  The path to the CSV file
    * `delimeter`: The separating character
    
    Returns the dictionary
    """

    # Read all data from CSV (assume that file is not too big)
    csv_fh = open(csv_path, "r")
    csv_lines = csv_fh.readlines()
    csv_fh.close()

    # Initialisation for conversion
    csv_dict = {}
    headers = csv_lines[0].replace("\n", "").split(delimeter)
    csv_lines = csv_lines[1:]
    for header in headers:
        csv_dict[header] = []

    # Start conversion to dict
    for csv_line in csv_lines:
        csv_line_list = csv_line.replace("\n", "").split(delimeter)
        for i in range(len(headers)):
            value = csv_line_list[i]
            if value == "":
                continue
            try:
                value = float(value)
            except:
                pass
            csv_dict[headers[i]].append(value)
    
    # Convert single item lists to items and things multi-item lists
    for header in headers:
        if len(csv_dict[header]) == 1:
            csv_dict[header] = csv_dict[header][0]
        else:
            csv_dict[header] = csv_dict[header]
    
    # Return
    return csv_dict

if __name__ == "__main__":
    """
    Calls the main function
    """
    main()