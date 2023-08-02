"""
 Title:         Surrogate Template
 Description:   Contains the basic structure for a surrogate class
 Author:        Janzen Choi

"""

# Libraries
import importlib, os, pathlib, sys

# Surrogate class
class __Surrogate__:

    def __init__(self, name:str, input_size:int, output_size:int, output_path:str):
        """
        Template for a surrogate model
        
        Parameters:
        * `name`: The name of the surrogate model
        * `input_size`: The number of inputs
        * `output_size`: The number of outputs
        * `output_path`: The path to the outputs
        """
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.output_path = output_path
    
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

    def get_output_path(self) -> str:
        """
        Gets the path to the outputs
        """
        return self.output_path

    def initialise(self, **kwargs) -> None:
        """
        Initialises the model (optional)
        """
        pass

    def set_train_data(self, train_input:list, train_output:list) -> None:
        """
        Initialises the training data
        
        Parameters:
        * `train_input`:  Input data for training
        * `train_output`: Output data for training
        """
        self.train_input  = train_input
        self.train_output = train_output

    def get_train_data(self) -> tuple:
        """
        Returns the training data
        """
        return self.train_input, self.train_output

    def set_valid_data(self, valid_input:list, valid_output:list) -> None:
        """
        Initialises the validation data
        
        Parameters:
        * `valid_input`:  Input data for validation
        * `valid_output`: Output data for validation
        """
        self.valid_input  = valid_input
        self.valid_output = valid_output

    def get_valid_data(self) -> tuple:
        """
        Returns the validation data
        """
        return self.valid_input, self.valid_output

    def train(self, **kwargs) -> None:
        """
        Trains the model
        """
        raise NotImplementedError

    def test(self, **kwargs) -> None:
        """
        Tests the model
        """
        raise NotImplementedError

def get_surrogate(surrogate_name:str, input_size:int, output_size:int, output_path:str, **kwargs) -> __Surrogate__:
    """
    Creates and returns a surrogate model
    
    Parameters:
    * `surrogate_name`: The name of the surrogate model
    * `input_size`: The number of inputs
    * `output_size`: The number of outputs
    * `output_path`: The path to the outputs
        
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
    surrogate = Surrogate(surrogate_name, input_size, output_size, output_path)
    surrogate.initialise(**kwargs)
    return surrogate