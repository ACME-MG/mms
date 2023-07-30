"""
 Title:         Parameter
 Description:   Stores information about a parameter (input / output) 
 Author:        Janzen Choi

"""

# Libraries
from copy import deepcopy
from mms.mappers.__mapper__ import __Mapper__
from mms.mappers.__mapper__ import get_mapper

# Parameter class
class Parameter:
    
    def __init__(self, param_name:str, value_list:list, mapper_name_list:list, **kwargs):
        """
        Constructor for the parameter object
        
        Parameters:
        * `param_name`: The name of the parameter
        * `value_list`: The list of values
        * `mapper_name_list`: The ordered list of how the parameter will be mapped
        """
        self.param_name = param_name
        self.value_list = value_list
        self.mapper_list = []
        for mapper_name in mapper_name_list:
            mapper = get_mapper(mapper_name, value_list, **kwargs)
            value_list = deepcopy(value_list)
            value_list = [mapper.map(value) for value in value_list]
            self.mapper_list.append(mapper)

    def get_param_name(self) -> str:
        """
        Gets the name of the parameter
        """
        return self.param_name
        
    def get_value_list(self) -> list:
        """
        Gets the values of the parameter
        """
        return self.value_list
    
    def get_mappers(self) -> list:
        """
        Gets the mappers of the parameter
        """
        return self.mapper_list
    
    def map_values(self, value_list:list) -> list:
        """
        Maps a list of values
        
        Parameters:
        * `value_list`: The list of values to be mapped
        
        Returns the list of mapped values
        """
        value_list = deepcopy(value_list)
        for mapper in self.mapper_list:
            value_list = [mapper.map(value) for value in value_list]
        return value_list

    def unmap_values(self, value_list:list) -> list:
        """
        Unmaps a list of values
        
        Parameters:
        * `value_list`: The list of values to be unmapped
        
        Returns the list of unmapped values
        """
        value_list = deepcopy(value_list)
        for mapper in self.mapper_list[::-1]: # iterate reverse order
            value_list = [mapper.unmap(value) for value in value_list]
        return value_list
