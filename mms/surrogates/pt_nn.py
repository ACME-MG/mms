"""
 Title:         Neural network
 Description:   For building a neural network 
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mms.mapper import MapperDict
from mms.interface.converter import dict_to_grid, grid_to_dict
from mms.surrogates.__surrogate__ import __Surrogate__

# Set tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

# Neural class
class Surrogate(__Surrogate__):
    
    # Trains the model
    def train(self, input_dict:dict, output_dict:dict, epochs:int, batch_size:int) -> None:
        
        # Initialise model
        input_size = self.get_input_size()
        output_size = self.get_output_size()
        self.model = CustomModel(input_size, output_size, [64, 32, 16])
        
        # Initialise other
        parameters = self.model.parameters()
        self.criterion = torch.nn.MSELoss()
        self.optimiser = optim.Adam(parameters, lr=0.001)
        
        # Create mappers
        self.input_mapper = MapperDict(input_dict)
        self.output_mapper = MapperDict(output_dict)
        
        # Map the data dictionaries
        norm_input_dict = self.input_mapper.map(input_dict)
        norm_output_dict = self.output_mapper.map(output_dict)
        
        # Convert the dictionaries to grids
        norm_input_grid = dict_to_grid(norm_input_dict)
        norm_output_grid = dict_to_grid(norm_output_dict)
        
        # Convert grids to tensors
        norm_input_tensor = torch.tensor(norm_input_grid)
        norm_output_tensor = torch.tensor(norm_output_grid)
        
        # Convert to data loader
        dataset = CustomDataset(norm_input_tensor, norm_output_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Start training    
        for _ in range(epochs):
            total_loss = 0
            for batch_inputs, batch_outputs in dataloader:
                self.optimiser.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_outputs)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()
        
    # Returns a prediction
    def predict(self, input_dict:dict) -> dict:
        
        # Map and convert input data
        norm_input_dict = self.input_mapper.map(input_dict)
        norm_input_grid = np.array(dict_to_grid(norm_input_dict))
        norm_input_tensor = torch.tensor(norm_input_grid)
        
        # Get the prediction
        with torch.no_grad():
            norm_output_tensor = self.model(norm_input_tensor)

        # Convert to dictionary
        norm_output_grid = norm_output_tensor.tolist()
        output_headers = self.output_mapper.get_headers()
        norm_output_dict = grid_to_dict(norm_output_grid, output_headers)
        
        # Unmap and return
        output_dict = self.output_mapper.unmap(norm_output_dict)
        return output_dict

# Custom PyTorch model
class CustomModel(torch.nn.Module):
    
    # Constructor
    def __init__(self, input_size:int, output_size:int, hidden_sizes:list):
        super(CustomModel, self).__init__()
        self.input_layer   = torch.nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = [torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                              for i in range(len(hidden_sizes)-1)]
        self.output_layer  = torch.nn.Linear(hidden_sizes[-1], output_size)
    
    # Runs the forward pass
    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        output_tensor = torch.relu(self.input_layer(input_tensor))
        for layer in self.hidden_layers:
            output_tensor = torch.relu(layer(output_tensor))
        output_tensor = self.output_layer(output_tensor)
        return output_tensor

# Custom Dataset
class CustomDataset(Dataset):
    
    # Constructor
    def __init__(self, input_tensor:torch.Tensor, output_tensor:torch.Tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    # Gets the length of the dataset
    def __len__(self) -> int:
        return len(self.input_tensor)

    # Gets the item from the dataset with an index
    def __getitem__(self, index:int) -> tuple:
        return self.input_tensor[index], self.output_tensor[index]
