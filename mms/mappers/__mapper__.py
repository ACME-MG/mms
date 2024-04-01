"""
 Title:         Mapper
 Description:   For mapping and unmapping data 
 Author:        Janzen Choi

"""

# Libraries
import importlib, os, pathlib, sys

# Mapper class
class __Mapper__:
    
    def __init__(self, name:str, value_list:list):
        """
        Template for mapper class
        
        Parameters:
        * `name`: The name of the mapper
        * `value_list`: A list of the values to be mapped and unmapped
        """
        self.name = name
        self.value_list = value_list
    
    def get_name(self) -> str:
        """
        Gets the name of the mapper
        """
        return self.name

    def get_value_list(self) -> list:
        """
        Gets the list of values
        """
        return self.value_list

    def initialise(self, **kwargs) -> float:
        """
        For initialising the mapper object with key word arguments
        """
        pass

    def map(self, value:float) -> float:
        """
        Maps a value
        
        Parameters:
        * `value`: The value to be mapped
        
        Returns the mapped value
        """
        raise NotImplementedError
    
    def unmap(self, value:float) -> float:
        """
        Unmaps a value
        
        Parameters:
        * `value`: The value to be unmapped
        
        Returns the unmapped value
        """
        raise NotImplementedError

    def get_info(self) -> dict:
        """
        Returns information about how the mapping is being done
        """
        raise NotImplementedError

def get_mapper(mapper_name:str, value_list:list, **kwargs) -> __Mapper__:
    """
    Creates and returns a mapper
    
    Parameters:
    * `mapper_name`: The name of the mapper model
    * `value_list`: The list of values
        
    Returns the mapper model object
    """

    # Get available mappers in current folder
    mappers_dir = pathlib.Path(__file__).parent.resolve()
    files = os.listdir(mappers_dir)
    files = [file.replace(".py", "") for file in files]
    files = [file for file in files if not file in ["__mapper__", "__pycache__"]]
    
    # Raise error if mapper name not in available mappers
    if not mapper_name in files:
        raise NotImplementedError(f"The mapper '{mapper_name}' has not been implemented")

    # Prepare dynamic import
    module_path = f"{mappers_dir}/{mapper_name}.py"
    spec = importlib.util.spec_from_file_location("mapper_file", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Initialise and return the mapper
    from mapper_file import Mapper
    mapper = Mapper(mapper_name, value_list)
    mapper.initialise(**kwargs)
    return mapper