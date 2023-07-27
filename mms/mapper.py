"""
 Title:         Mapper
 Description:   For mapping data 
 Author:        Janzen Choi

"""

# Libraries
import numpy as np

# Mapper class
class Mapper:
    
    def __init__(self, data_list:list):
        """
        Mapper class for mapping data
        
        Parameters:
        * `data_list`: List of values to define normalisation
        """
        self.value_mean = np.mean(data_list)
        self.value_std  = np.std(data_list)
        print(self.value_mean)
        print(self.value_std)
    
    def __init__(self, in_l_bound:float=0, in_u_bound:float=1, out_l_bound:float=0, out_u_bound:float=1):
        """
        Mapper class for mapping data
        
        Parameters:
        * `in_l_bound`:  Lower bound of input
        * `out_l_bound`: Lower bound of output
        * `in_u_bound`:  Upper bound of input
        * `out_u_bound`: Upper bound of output
        """
        self.in_l_bound = in_l_bound
        self.in_u_bound = in_u_bound
        self.out_l_bound = out_l_bound
        self.out_u_bound = out_u_bound
        self.distinct = in_l_bound == in_u_bound or out_l_bound == out_u_bound
    
    def map(self, value:float) -> float:
        """
        Maps a value (works for lists of values)
        
        Parameters:
        * `value`: The value to be mapped
        
        Returns the mapped value
        """
        if self.distinct:
            return value
        if isinstance(value, list):
            return [self.map(v) for v in value]
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.in_l_bound) * factor + self.out_l_bound

    def unmap(self, value:float) -> float:
        """
        Unmaps a value (works for lists of values)
        
        Parameters:
        * `value`: The value to be unmapped
        
        Returns the unmapped value
        """
        if self.distinct:
            return value
        if isinstance(value, list):
            return [self.unmap(v) for v in value]
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.out_l_bound) / factor + self.in_l_bound

# MapperDict class
class MapperDict:
    
    def __init__(self, data_dict:dict):
        """
        MapperDict class for mapping dictionaries of data lists
        
        Parameters:
        * `data_dict`: Dictionary of lists of values to define normalisation
        """
        self.headers = list(data_dict.keys())
        self.mapper_dict = {}
        for header in self.headers:
            mapper = Mapper(min(data_dict[header]), max(data_dict[header]), 0, 1)
            self.mapper_dict[header] = mapper
    
    def get_headers(self):
        """
        Returns the headers of the dictionary
        """
        return self.headers
    
    def map(self, data_dict:dict) -> dict:
        """
        maps a dictionary of lists of data
        
        Parameters:
        * `data_dict`: Dictionary of lists of unmapd values to be mapd
        
        Returns the mapd dictionary
        """
        new_data_dict = {}
        for header in self.headers:
            new_data_dict[header] = self.mapper_dict[header].map(data_dict[header])
        return new_data_dict

    def unmap(self, data_dict:dict) -> dict:
        """
        Unmaps a dictionary of lists of data
        
        Parameters:
        * `data_dict`: Dictionary of lists of mapd values to be unmapd
        
        Returns the unmapd dictionary
        """
        new_data_dict = {}
        for header in self.headers:
            new_data_dict[header] = self.mapper_dict[header].unmap(data_dict[header])
        return new_data_dict
        