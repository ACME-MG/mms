"""
 Title:         Simple neural network
 Description:   A simple neural network 
 Author:        Janzen Choi

"""

# Libraries
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from mms.surrogates.__surrogate__ import __Surrogate__, SimpleModel, SimpleDataset

# Constants
HIDDEN_LAYER_SIZES  = [256, 128, 64, 32]
START_LEARNING_RATE = 1E-2 # 1E-3
MIN_LEARNING_RATE   = 1E-7 # 1E-6
WEIGHT_DECAY        = 1E-7
REDUCTION_FACTOR    = 0.5
PATIENCE_AMOUNT     = 100

# Surrogate class
class Surrogate(__Surrogate__):
    
    def initialise(self, input_size:int, output_size:int, epochs:int=1000, batch_size:int=32, verbose:bool=False):
        """
        Defines parameters for the surrogate model
        
        Parameters:
        * `input_size`:  The number of inputs
        * `output_size`: The number of outputs
        * `epochs`:       The number of epochs
        * `batch_size`:   The size of each batch
        * `verbose`:      Whether to display updates or not
        """
        
        # Initialise input variables
        self.input_size  = input_size
        self.output_size = output_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.verbose     = verbose
        
        # Initialise internal varaibles
        self.model = SimpleModel(self.input_size, self.output_size, HIDDEN_LAYER_SIZES)
        parameters = self.model.parameters()
        
        # Define optimisation objects
        self.loss_function = torch.nn.MSELoss()
        self.optimiser     = optim.Adam(parameters, lr=START_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, "min", factor=REDUCTION_FACTOR, patience=PATIENCE_AMOUNT)

    def train(self, train_input:list, train_output:list, valid_input:list, valid_output:list) -> None:
        """
        Trains the model
        
        Parameters:
        * `train_input`:  Input data for training
        * `train_output`: Output data for training
        * `valid_input`:  Input data for validation
        * `valid_output`: Output data for validation
        """
        
        # Get the data and convert to tensors
        train_input_tensor  = torch.tensor(train_input)
        train_output_tensor = torch.tensor(train_output)
        valid_input_tensor  = torch.tensor(valid_input)
        valid_output_tensor = torch.tensor(valid_output)
        
        # Initialise everything before training
        dataset = SimpleDataset(train_input_tensor, train_output_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Start training
        print()
        for epoch in range(self.epochs):
            
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
            self.add_train_loss(train_loss.item())
            self.add_valid_loss(valid_loss.item())
    
            # Print update if desired
            if self.verbose and epoch % ((self.epochs)//10) == 0:
                print("Epoch={}, \tTrainLoss={:0.3}, \tValidLoss={:0.3}".format(epoch, train_loss.item(), valid_loss.item()))
    
            # Updates the state
            self.scheduler.step(valid_loss)
            curr_learning_rate = self.optimiser.param_groups[0]["lr"]
            if curr_learning_rate < MIN_LEARNING_RATE:
                break
