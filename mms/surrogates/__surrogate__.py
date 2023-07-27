"""
 Title:         Surrogate Template
 Description:   Contains the basic structure for a surrogate class
 Author:        Janzen Choi

"""

# Libraries
import importlib, os, pathlib, sys

# Surrogate class
class __Surrogate__:

    def __init__(self, name:str, input_size:int, output_size:int):
        """
        Template for a surrogate model
        
        Parameters:
        * `name`: The name of the surrogate model
        * `input_size`: The number of inputs
        * `output_size`: The number of outputs
        """
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
    
    def get_name(self) -> str:
        """
        Gets the name of the surrogate model
        """
        return self.name
    
    def get_input_size(self) -> int:
        """
        Gets the number of inputs
        """
        return self.input_size
    
    def get_output_size(self) -> int:
        """
        Gets the number of outputs
        """
        return self.output_size

    def initialise(self, **kwargs) -> None:
        """
        Initialises the model (optional)
        """
        pass

    def train(self, input_dict:dict, output_dict:dict, **kwargs) -> None:
        """
        Trains the model
        
        Parameters:
        * `input_dict`:  Dictionary of lists representing the input data
        * `output_dict`: Dictionary of lists representing the output data
        * `epochs`:      Number of epochs
        * `batch_size`:  The size of the batch
        * `verbose`:     Whether or not to train with additional output from tensorflow
        """
        raise NotImplementedError

    def predict(self, input_dict:dict, **kwargs) -> dict:
        """
        Makes predictions based on a fitted model
        
        Parameters:
        * `input_dict`:  Dictionary of lists representing the input data
        
        Returns the predictions based on the inputs
        """
        raise NotImplementedError

def get_surrogate(surrogate_name:str, input_size:int, output_size:int, **kwargs) -> __Surrogate__:
    """
    Creates and returns a surrogate model
    
    Parameters:
    * `surrogate_name`: The name of the surrogate model
    * `input_size`: The number of inputs
    * `output_size`: The number of outputs
        
    Returns the surrogate model object
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
    surrogate = Surrogate(surrogate_name, input_size, output_size)
    surrogate.initialise(**kwargs)
    return surrogate