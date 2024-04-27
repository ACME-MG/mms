"""
 Title:         Neural network
 Description:   For building a neural network 
 Author:        Janzen Choi

"""

# Libraries
import warnings; warnings.filterwarnings("ignore")
import torch, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Set tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

# Constants
HIDDEN_LAYER_SIZES  = [256, 128, 64, 32]
START_LEARNING_RATE = 1E-2 # 1E-3
MIN_LEARNING_RATE   = 1E-7 # 1E-6
WEIGHT_DECAY        = 1E-7
REDUCTION_FACTOR    = 0.5
PATIENCE_AMOUNT     = 100

# Neural class
class Neural:
    
    def __init__(self, input_size:int, output_size:int):
        """
        Constructor for the neural network surrogate model
        
        Parameters:
        * `input_size`:  The number of inputs
        * `output_size`: The number of outputs
        """
        
        # Initialise internal variables
        self.input_size  = input_size
        self.output_size = output_size
        self.results     = {"train_loss": [], "valid_loss": []}
        
        # Initialise model
        self.model = CustomModel(self.input_size, self.output_size, HIDDEN_LAYER_SIZES)
        parameters = self.model.parameters()
        
        # Define optimisation objects
        self.loss_function = torch.nn.MSELoss()
        self.optimiser     = optim.Adam(parameters, lr=START_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, "min", factor=REDUCTION_FACTOR, patience=PATIENCE_AMOUNT)
    
    def train(self, train_input:list, train_output:list, valid_input:list, valid_output:list, epochs:int, batch_size:int, verbose:bool=False) -> None:
        """
        Trains the model
        
        Parameters:
        * `train_input`:  Input data for training
        * `train_output`: Output data for training
        * `valid_input`:  Input data for validation
        * `valid_output`: Output data for validation
        * `epochs`:       The number of epochs
        * `batch_size`:   The size of each batch
        * `verbose`:      Whether to display updates or not
        """
        
        # Get the data and convert to tensors
        train_input_tensor  = torch.tensor(train_input)
        train_output_tensor = torch.tensor(train_output)
        valid_input_tensor  = torch.tensor(valid_input)
        valid_output_tensor = torch.tensor(valid_output)
        
        # Initialise everything before training
        dataset = CustomDataset(train_input_tensor, train_output_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Start training
        print()
        for epoch in range(epochs):
            
            # Completes training
            for batch_inputs, batch_outputs in data_loader:
                self.optimiser.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_function(outputs, batch_outputs)
                loss.backward()
                self.optimiser.step()
            
            # Gets the training loss
            with torch.no_grad():
                prd_train_output_tensor = self.model(train_input_tensor)
                train_loss = self.loss_function(prd_train_output_tensor, train_output_tensor)
            
            # Gets the validation loss
            with torch.no_grad():
                prd_valid_output_tensor = self.model(valid_input_tensor)
                valid_loss = self.loss_function(prd_valid_output_tensor, valid_output_tensor)
    
            # Update results dictionary
            self.results["train_loss"].append(train_loss.item())
            self.results["valid_loss"].append(valid_loss.item())
    
            # Print update if desired
            if verbose:
                print("Epoch={}, \tTrainLoss={:0.3}, \tValidLoss={:0.3}".format(
                    epoch+1, train_loss.item(), valid_loss.item()
                ))
    
            # Updates the state
            self.scheduler.step(valid_loss)
            curr_learning_rate = self.optimiser.param_groups[0]["lr"]
            if curr_learning_rate < MIN_LEARNING_RATE:
                break
    
    def plot_loss_history(self, loss_path:str):
        """
        Make plot of training and validation losses
        
        Parameters:
        * `loss_path`:   The path to output the loss
        """
        plt.title("Log_10 Loss vs Epochs")
        plt.figure(figsize=(6, 6))
        plt.xlabel("Number of epochs", fontsize=15)
        plt.ylabel("MSE Loss", fontsize=15)
        plt.plot(list(range(len(self.results["train_loss"]))), self.results["train_loss"], c="blue", label="training")
        plt.plot(list(range(len(self.results["valid_loss"]))), self.results["valid_loss"], c="red", label="validation")
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

    def save(self, model_path:str) -> None:
        """
        Saves the surrogate model
        
        Parameters:
        * `model_path`: The path to the surrogate model (excluding extension)
        """
        torch.save(self.model, f"{model_path}.pt")

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
