"""
 Title:         Neural network
 Description:   For building a neural network 
 Author:        Janzen Choi

"""

# Libraries
import warnings; warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mms.surrogates.__surrogate__ import __Surrogate__

# Set tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

# Neural class
class Surrogate(__Surrogate__):
    
    # Trains the model
    def train(self, input_grid:dict, output_grid:list, epochs:int, batch_size:int) -> None:
        
        # Initialise model
        input_size = self.get_input_size()
        output_size = self.get_output_size()
        self.model = CustomModel(input_size, output_size, [32, 16, 8])
        parameters = self.model.parameters()
        
        # Define optimisation
        loss_fuction = MeanRelativeErrorLoss()
        optimiser = optim.Adam(parameters, lr=0.001)
        
        # Convert data to data loader
        input_tensor = torch.tensor(input_grid)
        output_tensor = torch.tensor(output_grid)
        dataset = CustomDataset(input_tensor, output_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Start training    
        # torch.autograd.set_detect_anomaly(True) # for debugging
        for _ in range(epochs):
            total_loss = 0
            for batch_inputs, batch_outputs in dataloader:
                optimiser.zero_grad()
                outputs = self.model(batch_inputs)
                loss = loss_fuction(outputs, batch_outputs)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
        
    # Returns a prediction
    def predict(self, input_grid:list) -> list:
        input_tensor = torch.tensor(input_grid)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        output_grid = output_tensor.tolist()
        print(output_grid)
        return output_grid

# Relative error function
def get_relative_error(prd_value, exp_value):
    abs_diff = torch.abs(prd_value - exp_value)
    relative_error = abs_diff / torch.abs(exp_value)
    return relative_error

# Relative error class
class MeanRelativeErrorLoss(torch.nn.Module):
    
    # Constructor
    def __init__(self):
        super(MeanRelativeErrorLoss, self).__init__()

    # Calculates the relative error
    def forward(self, prd_value, exp_value):
        relative_errors = get_relative_error(prd_value, exp_value)
        mean_relative_error = torch.mean(relative_errors)
        return mean_relative_error

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
