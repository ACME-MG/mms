"""
 Title:         Neural network
 Description:   For building a neural network 
 Author:        Janzen Choi

"""

# Libraries
import warnings; warnings.filterwarnings("ignore")
import math
import torch, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from mms.surrogates.__surrogate__ import __Surrogate__

# Set tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

# Neural class
class Surrogate(__Surrogate__):
    
    # Trains the model
    def train(self, epochs:int, batch_size:int) -> None:
        
        # Initialise model
        input_size = self.get_input_size()
        output_size = self.get_output_size()
        self.model = CustomModel(input_size, output_size, [32, 16, 8])
        parameters = self.model.parameters()
        
        # Get the data and convert to tensors
        train_input, train_output = self.get_train_data()
        valid_input, valid_output = self.get_valid_data()
        train_input_tensor = torch.tensor(train_input)
        train_output_tensor = torch.tensor(train_output)
        valid_input_tensor = torch.tensor(valid_input)
        valid_output_tensor = torch.tensor(valid_output)
        
        # Define optimisation
        learning_rate = 1e-3
        weight_decay  = 0 # 1e-5
        loss_function = torch.nn.MSELoss()
        optimiser     = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        
        # Initialise everything before training
        dataset = CustomDataset(train_input_tensor, train_output_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss_list = []
        valid_loss_list = []
        
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
                train_loss_list.append(math.log(train_loss.item()))
            
            # Gets the validation loss
            with torch.no_grad():
                prd_valid_output_tensor = self.model(valid_input_tensor)
                valid_loss = loss_function(prd_valid_output_tensor, valid_output_tensor)
                valid_loss_list.append(math.log(valid_loss.item()))
            
        # Make plot
        plt.title("Log Loss vs Epochs")
        plt.xlabel("epochs")
        plt.ylabel("log(loss)")
        plt.plot(list(range(epochs)), train_loss_list, c="blue", label="training")
        plt.plot(list(range(epochs)), valid_loss_list, c="red", label="validation")
        plt.legend()
        plt.savefig(f"{self.get_output_path()}/loss.png")

    # Returns a prediction based on the validation data only
    def predict(self) -> list:
        valid_input, _ = self.get_valid_data()
        input_tensor = torch.tensor(valid_input)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        output_grid = output_tensor.tolist()
        return output_grid

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
