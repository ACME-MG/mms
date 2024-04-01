"""
 Title:         Logarithmic Mapper
 Description:   For mapping and unmapping data logarithmically 
 Author:        Janzen Choi

"""

# Libraries
import math
from mms.mappers.__mapper__ import __Mapper__

# Mapper class
class Mapper(__Mapper__):

    # Initialise
    def initialise(self, base:float=10) -> None:
        self.base = base

    # Logarithmically maps a value
    def map(self, value:float) -> float:
        if value <= 0:
            return -10
        return math.log(value) / math.log(self.base)

    # Logarithmically unmaps a value
    def unmap(self, value:float) -> float:
        return math.pow(self.base, value)
    
    # Returns a dictionary of information about the mapping
    def get_info(self) -> dict:
        return {"base": self.base}