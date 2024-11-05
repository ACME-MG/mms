"""
 Title:         KFold Neural network with Misorientation
 Description:   Neural network with regularisation, KFold cross validation, and misorientation 
 Author:        Janzen Choi

"""

# Libraries
import torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from mms.surrogates.__surrogate__ import __Surrogate__, SimpleModel, SimpleDataset
from mms.helper.general import integer_to_ordinal
from mms.maths.torch_maths import get_torch_misorientation

# Constants
HIDDEN_LAYER_SIZES  = [128, 256, 512]
START_LEARNING_RATE = 1E-3 # 1E-2
MIN_LEARNING_RATE   = 1E-7 # 1E-6
WEIGHT_DECAY        = 1E-7
REDUCTION_FACTOR    = 0.5
PATIENCE_AMOUNT     = 100

# Surrogate class
class Surrogate(__Surrogate__):
    
    def initialise(self, input_size:int, output_size:int, num_splits:int=5, epochs:int=1000,
                   batch_size:int=32, verbose:bool=False, crystal_type:str="cubic", mori_weight:float=1.0):
        """
        Defines the parameters of neural network surrogate model
        
        Parameters:
        * `input_size`:   The number of inputs
        * `output_size`:  The number of outputs
        * `num_splits`:   The number of kfold splits (k)
        * `epochs`:       The number of epochs
        * `batch_size`:   The size of each batch
        * `verbose`:      Whether to display updates or not
        * `crystal_type`: Type of crystal structure to calculate misorientation
        * `mori_weight`:  How much to weigh the misorientation losses
        """

        # Initialise input variables
        self.input_size   = input_size
        self.output_size  = output_size
        self.num_splits   = num_splits
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.crystal_type = crystal_type
        self.mori_weight  = mori_weight

        # Initialise internal variables
        self.device = self.get_device()
        self.kfold  = KFold(n_splits=self.num_splits, shuffle=True)
        self.model_list = []

    def loss_function(self, target:torch.tensor, output:torch.tensor) -> torch.tensor:
        """
        Calculates the loss between the target and output values;
        Calculates the first output using MSE and the remaining outputs
        in groups of three using misorientation
        
        Parameters:
        * `target`: The target values
        * `output`: The output values
        
        Returns the loss values
        """
        
        # Calculate the error for stress ([0])
        target_stress = target[:,0]
        output_stress = output[:,0]
        stress_loss = torch.mean((target_stress-output_stress)**2)
        
        # Conduct unmapping for misorientation calculations
        unmapped_output = self.unmap_output(output)
        unmapped_target = self.unmap_output(target)
        
        # Calculate the misorientation for orientations ([3*n+1,3*n+4])
        mean_misorientation_list = []
        num_angles = int((output.shape[1]-1)//3)
        for angle_index in range(num_angles):
            
            # Get specific angles
            target_angles = unmapped_target[:,3*angle_index+1:3*angle_index+4]
            output_angles = unmapped_output[:,3*angle_index+1:3*angle_index+4]
            
            # Get misorientations for each angle pair
            misorientation_list = []
            for target_angle, output_angle in zip(target_angles, output_angles):
                misorientation = get_torch_misorientation(target_angle, output_angle, self.crystal_type)
                misorientation_list.append(misorientation)
            
            # Average and add to misorientation
            misorientation_list = torch.tensor(misorientation_list, dtype=torch.float32)
            misorientation_mean = torch.mean(misorientation_list)
            mean_misorientation_list.append(misorientation_mean)
        
        # Combine losses and return
        mean_misorientation_list = torch.tensor(mean_misorientation_list, dtype=torch.float32)
        misorientation_loss = torch.mean(mean_misorientation_list)*num_angles
        total_loss = stress_loss + misorientation_loss*self.mori_weight
        return total_loss
        
    def train(self, train_input:list, train_output:list, valid_input:list, valid_output:list) -> None:
        """
        Trains the model
        
        Parameters:
        * `train_input`:  Input data for training
        * `train_output`: Output data for training
        * `valid_input`:  Input data for validation (unused)
        * `valid_output`: Output data for validation (unused)
        """

        # Iterate through folds
        train_dataset = SimpleDataset(torch.tensor(train_input), torch.tensor(train_output))
        for fold, (train_indexes, valid_indexes) in enumerate(self.kfold.split(train_dataset)):

            # Print message
            print(f"\nConducting {fold+1}/{self.num_splits} folds\n")
            
            # Define data loaders
            train_loader = DataLoader(
                dataset    = train_dataset,
                batch_size = self.batch_size,
                sampler    = torch.utils.data.SubsetRandomSampler(train_indexes),
            )
            valid_loader = DataLoader(
                dataset    = train_dataset,
                batch_size = self.batch_size,
                sampler    = torch.utils.data.SubsetRandomSampler(valid_indexes),
            )

            # Initialise model, optimiser, and scheduler
            model = SimpleModel(self.input_size, self.output_size, HIDDEN_LAYER_SIZES, self.device)
            optimiser = torch.optim.Adam(model.parameters(), lr=START_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min", factor=REDUCTION_FACTOR, patience=PATIENCE_AMOUNT)
            
            # Train the model
            valid_loss_list = []
            for epoch in range(1,self.epochs+1):
                
                # Train and get loss
                self.train_fold(model, optimiser, train_loader)
                train_loss = self.get_loss(model, train_loader)
                valid_loss = self.get_loss(model, valid_loader)
                self.add_train_loss(train_loss)
                self.add_valid_loss(valid_loss)
                valid_loss_list.append(valid_loss)

                # Display loss
                if self.verbose and epoch % (self.epochs//10) == 0:
                    print("Epoch={}, \tTrainLoss={:0.3}, \tValidLoss={:0.3}".format(epoch, train_loss, valid_loss))

                # Update scheduler
                scheduler.step(valid_loss)
                curr_learning_rate = optimiser.param_groups[0]["lr"]
                if curr_learning_rate < MIN_LEARNING_RATE:
                    break

            # Add model and losses
            self.model_list.append({
                "model": model,
                "loss":  np.average(valid_loss_list),
                "valid": valid_indexes,
            })
        
        # Select model based on minimality of losses
        loss_list = [model["loss"] for model in self.model_list]
        min_loss = min(loss_list)
        min_index = loss_list.index(min_loss)
        self.model = self.model_list[min_index]["model"]
        self.valid_indexes = self.model_list[min_index]["valid"]
        print(f"\nSelected the {integer_to_ordinal(min_index+1)} model\n")

    def get_valid_indexes(self) -> list:
        """
        Returns a list of the validation indexes
        """
        return self.valid_indexes

    def train_fold(self, model:torch.nn.Module, optimiser:torch.optim.Optimizer, data_loader:DataLoader) -> None:
        """
        Trains the model for a single fold

        Parameters:
        * `model`:       The model to run
        * `data_loader`: The loader for the training data
        """
        model.train()
        for _, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimiser.zero_grad()
            output = model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            optimiser.step()

    def get_loss(self, model:torch.nn.Module, data_loader:DataLoader) -> float:
        """
        Gets the loss from the model

        Parameters:
        * `model`:       The model to run
        * `data_loader`: The loader for the training data

        Returns the loss value
        """
        loss_values = []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss_value = self.loss_function(target, output).item()
                loss_values.append(loss_value)
        return np.average(loss_values)
