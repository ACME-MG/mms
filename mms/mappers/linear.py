"""
 Title:         Linear Mapper
 Description:   For mapping and unmapping data linearly 
 Author:        Janzen Choi

"""

# Libraries
from mms.mappers.__mapper__ import __Mapper__

# Mapper class
class Mapper(__Mapper__):

    # Initialise
    def initialise(self, out_l_bound:float=0, out_u_bound:float=1) -> None:
        
        # Get in bounds
        value_list = self.get_value_list()
        self.in_l_bound = min(value_list)
        self.in_u_bound = max(value_list)
        
        # Define out bounds
        self.out_l_bound = out_l_bound
        self.out_u_bound = out_u_bound
        self.distinct = self.in_l_bound == self.in_u_bound or self.out_l_bound == self.out_u_bound

    # Linearly maps a value
    def map(self, value:float) -> float:
        if self.distinct:
            return value
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.in_l_bound) * factor + self.out_l_bound

    # Linearly unmaps a value
    def unmap(self, value:float) -> float:
        if self.distinct:
            return value
        factor = (self.out_u_bound - self.out_l_bound) / (self.in_u_bound - self.in_l_bound)
        return (value - self.out_l_bound) / factor + self.in_l_bound

    # Returns a dictionary of information about the mapping
    def get_info(self) -> dict:
        return {
            "in_l_bound":  self.in_l_bound,
            "in_u_bound":  self.in_u_bound,
            "out_l_bound": self.out_l_bound,
            "out_u_bound": self.out_u_bound,
        }
