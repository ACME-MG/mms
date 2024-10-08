"""
 Title:         Surrogate Template for Pytorch Neural Networks
 Description:   Contains the basic structure for a surrogate class
 Author:        Janzen Choi

"""

# Libraries
import importlib, os, pathlib, sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# Default settings for Pytorch
import warnings; warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.DoubleTensor)

# The Surrogate Template Class
class __Surrogate__:

    def __init__(self, name:str, input_dict, output_dict):
        """
        Class for defining a surrogate
        
        Parameters:
        * `name`:        The name of the surrogate model
        * `input_dict`:  The dictionary containing the mapping of the inputs
        * `output_dict`: The dictionary  containing the mapping of the outputs
        """
        self.name = name
        self.results = {"train_loss": [], "valid_loss": []}
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        print(f"\n  Running Torch with '{device_name}' ...\n")
        self.input_dict = input_dict
        self.output_dict = output_dict

    def get_name(self) -> str:
        """
        Gets the name of the msurrogate
        """
        return self.name

    def get_device(self) -> torch.device:
        """
        Returns the Pytorch device
        """
        return self.device

    def add_train_loss(self, train_loss:float) -> None:
        """
        Adds a training loss value to the list

        Parameters:
        * `train_loss`: The training loss value
        """
        self.results["train_loss"].append(train_loss)

    def add_valid_loss(self, valid_loss:float) -> None:
        """
        Adds a validation loss value to the list

        Parameters:
        * `valid_loss`: The validation loss value
        """
        self.results["valid_loss"].append(valid_loss)

    def initialise(self, input_size:int, output_size:int, **kwargs) -> None:
        """
        Runs at the start, once; must be implemented;
        must be overriden
        
        Parameters:
        * `input_size`:  The number of inputs
        * `output_size`: The number of outputs
        """
        raise NotImplementedError
    
    def train(self, train_input:list, train_output:list, valid_input:list, valid_output:list) -> None:
        """
        Trains the model;
        must be overriden
        
        Parameters:
        * `train_input`:  Input data for training
        * `train_output`: Output data for training
        * `valid_input`:  Input data for validation
        * `valid_output`: Output data for validation
        """
        raise NotImplementedError

    def plot_loss_history(self, loss_path:str):
        """
        Make plot of training and validation losses
        
        Parameters:
        * `loss_path`:   The path to output the loss
        """
        plt.title("Log_10 Loss vs Epochs")
        plt.figure(figsize=(6, 6))
        plt.xlabel("Number of epochs", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.plot(list(range(len(self.results["train_loss"]))), self.results["train_loss"], c="blue", label="Training")
        plt.plot(list(range(len(self.results["valid_loss"]))), self.results["valid_loss"], c="red", label="Validation")
        plt.yscale("log")
        plt.legend(framealpha=1, edgecolor="black", fancybox=True, facecolor="white")
        plt.savefig(f"{loss_path}.png")
        plt.clf()

    def get_valid_indexes(self) -> list:
        """
        Returns a list of the validation indexes;
        called if no validation data is supplied
        """
        raise NotImplementedError

    def unmap_input(self, input_tensor:torch.tensor) -> torch.tensor:
        """
        Unmaps the input tensor
        
        Parameters:
        * `input_tensor`: The tensor of inputs
        
        Returns the unmapped input tensor
        """
        return unmap_params(self.input_dict, input_tensor)

    def unmap_output(self, output_tensor:torch.tensor) -> torch.tensor:
        """
        Unmaps the output tensor
        
        Parameters:
        * `output_tensor`: The tensor of outputs
        
        Returns the unmapped output tensor
        """
        return unmap_params(self.output_dict, output_tensor)

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

    def save(self, model_path:str) -> None:
        """
        Saves the surrogate model
        
        Parameters:
        * `model_path`: The path to the surrogate model (excluding extension)
        """
        torch.save(self.model, f"{model_path}.pt")

def unmap_params(param_dict:dict, param_tensor:torch.tensor) -> torch.tensor:
    """
    Unmaps the parameter tensor;
    Only supports linear (for now)
    
    Parameters:
    * `param_dict`:   The dictionary containing the parameter objects
    * `param_tensor`: The tensor containing the output values
    
    Returns the unmapped tensor
    """

    # Iterate through outputs
    key_list = list(param_dict.keys())
    for i, key in enumerate(key_list):
        param = param_dict[key]
        
        # Iterate through mappers in reverse order (for unmapping)
        mappers = param.get_mappers()
        for mapper in mappers[::-1]:
            
            # Get mapper information
            mapper_name = mapper.get_name()
            mapper_info = mapper.get_info()
            
            # Conduct linear unmapping using tensor maths
            if mapper_name == "linear":
                in_l_bound  = mapper_info["in_l_bound"]
                in_u_bound  = mapper_info["in_u_bound"]
                out_l_bound = mapper_info["out_l_bound"]
                out_u_bound = mapper_info["out_u_bound"]
                if in_l_bound == in_u_bound or out_l_bound == out_u_bound:
                    continue
                param_tensor[:,i].sub_(out_l_bound)
                param_tensor[:,i].div_((out_u_bound-out_l_bound)/(in_u_bound-in_l_bound))
                param_tensor[:,i].add_(in_l_bound)

    # Return unmapped tensor
    return param_tensor    
    
# Simple PyTorch model
class SimpleModel(torch.nn.Module):
    
    def __init__(self, input_size:int, output_size:int, hidden_sizes:list):
        """
        Defines the structure of the neural network
        
        Parameters:
        * `input_size`: The number of inputs
        * `output_size`: The number of outputs
        * `hidden_sizes`: List of number of nodes for the hidden layers
        """
        super(SimpleModel, self).__init__()
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

# Simple Dataset
class SimpleDataset(Dataset):
    
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

def create_surrogate(surrogate_name:str, input_size:int, output_size:int,
                     input_dict:dict, output_dict:dict, **kwargs) -> __Surrogate__:
    """
    Creates and returns a surrogate

    Parameters:
    * `surrogate_name`: The name of the surrogate
    * `input_size`:     The number of input variables
    * `output_size`:    The number of output variables
    * `input_dict`:     The dictionary containing the mapping of the inputs
    * `output_dict`:    The dictionary  containing the mapping of the outputs

    Returns the surrogate
    """

    # Get available surrogates in current folder
    surrogates_dir = pathlib.Path(__file__).parent.resolve()
    files = os.listdir(surrogates_dir)
    files = [file.replace(".py", "") for file in files]
    files = [file for file in files if not file in ["__surrogate__", "__pycache__"]]
    
    # Raise error if surrogate name not in available surrogates
    if not surrogate_name in files:
        raise NotImplementedError(f"The surrogate '{surrogate_name}' has not been implemented")

    # Prepare dynamic import
    module_path = f"{surrogates_dir}/{surrogate_name}.py"
    spec = importlib.util.spec_from_file_location("surrogate_file", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Initialise and return the surrogate
    from surrogate_file import Surrogate
    surrogate = Surrogate(surrogate_name, input_dict, output_dict)
    surrogate.initialise(input_size, output_size, **kwargs)
    return surrogate
