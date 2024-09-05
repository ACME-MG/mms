"""
 Title:         Optimiser
 Description:   Optimiser for the PyTorch surrogate model
 Author:        Janzen Choi

"""

# Libraries
import math, numpy as np, torch
import sys; sys.path += ["/home/janzen/code/mms"]
from mms.helper.io import csv_to_dict
from copy import deepcopy
from torch.func import jacrev, vmap

# Constants
DIRECTORY = "results/240830134116_617_s3"
SUR_PATH = f"{DIRECTORY}/sm.pt"
MAP_PATH = f"{DIRECTORY}/map.csv"
EXP_PATH = "data/617_s3_exp.csv"

# Main function
def main() -> None:
    
    exp_dict = csv_to_dict(EXP_PATH)
    strain_list = exp_dict["strain_intervals"]
    param_list = [825, 2, 112, 15]
    mapper = Mapper(MAP_PATH)
    
    input_list = mapper.map_input(param_list + [strain_list[5]])
    input_tensor = torch.tensor(input_list)
    
    surrogate = Surrogate(SUR_PATH)
    output_tensor = surrogate.forward(input_tensor)
    output_list = mapper.unmap_output(output_tensor.tolist())
    print(output_list)

# Class for the surrogate
class Surrogate(torch.nn.Module):
    
    def __init__(self, sm_path:str):
        """
        Constructor for the surrogate
        
        Parameters:
        * `sm_path`: The path to the saved surrogate 
        """
        super().__init__()
        self.model = torch.load(sm_path)
        self.model.eval()

    def forward(self, x) -> torch.Tensor:
        """
        Gets the response of the model from the parameters
        
        Parameters:
        * `x`: The parameters (tau_s, b, tau_0, n, strain)
        
        Returns the response as a tensor
        """
        return self.model(x.double())

# Mapper class for mapping inputs and unmapping outputs
class Mapper:
    
    def __init__(self, map_path:str):
        """
        Constructor for the mapper class
        
        Parameters:
        * `map_path`: Path to the map
        """
        map_dict = csv_to_dict(map_path)
        self.input_map_dict = {}
        self.output_map_dict = {}
        num_inputs = map_dict["param_type"].count("input")
        for key in map_dict.keys():
            self.input_map_dict[key] = map_dict[key][:num_inputs]
            self.output_map_dict[key] = map_dict[key][num_inputs:]

    def map_input(self, input_list:list) -> list:
        """
        Maps the raw input for the surrogate model
        
        Parameters:
        * `input_list`: List of unmapped input values
        
        Returns the mapped input values
        """
        mapped_input_list = []
        for i in range(len(input_list)):
            try:
                mapped_input = math.log(input_list[i]) / math.log(self.input_map_dict["base"][i])
                mapped_input = linear_map(
                    value = mapped_input,
                    in_l  = self.input_map_dict["in_l_bound"][i],
                    in_u  = self.input_map_dict["in_u_bound"][i],
                    out_l = self.input_map_dict["out_l_bound"][i],
                    out_u = self.input_map_dict["out_u_bound"][i],
                )
            except ValueError:
                return None
            mapped_input_list.append(mapped_input)
        return mapped_input_list
    
    def unmap_output(self, output_list:list) -> list:
        """
        Unmaps the output from the surrogate model
        
        Parameters:
        * `output_list`: List of mapped output values
        
        Returns the unmapped output values
        """
        unmapped_output_list = []
        for i in range(len(output_list)):
            try:
                unmapped_output = linear_unmap(
                    value = output_list[i],
                    in_l  = self.output_map_dict["in_l_bound"][i],
                    in_u  = self.output_map_dict["in_u_bound"][i],
                    out_l = self.output_map_dict["out_l_bound"][i],
                    out_u = self.output_map_dict["out_u_bound"][i],
                )
                unmapped_output = math.pow(self.output_map_dict["base"][i], unmapped_output)
            except ValueError:
                return None
            unmapped_output_list.append(unmapped_output)
        return unmapped_output_list

def linear_map(value:float, in_l:float, in_u:float, out_l:float, out_u:float) -> float:
    """
    Linearly maps a value

    Parameters:
    * `value`:  The value to be mapped
    * `in_l`:   The lower bound of the input
    * `in_u`:   The upper bound of the input
    * `out_l`:  The lower bound of the output
    * `out_u`:  The upper bound of the output

    Returns the mapped value
    """
    if in_l == in_u or out_l == out_u:
        return value
    factor = (out_u - out_l) / (in_u - in_l)
    return (value - in_l) * factor + out_l

def linear_unmap(value:float, in_l:float, in_u:float, out_l:float, out_u:float) -> float:
    """
    Linearly unmaps a value

    Parameters:
    * `value`:  The value to be unmapped
    * `in_l`:   The lower bound of the input
    * `in_u`:   The upper bound of the input
    * `out_l`:  The lower bound of the output
    * `out_u`:  The upper bound of the output

    Returns the unmapped value
    """
    if in_l == in_u or out_l == out_u:
        return value
    factor = (out_u - out_l) / (in_u - in_l)
    return (value - out_l) / factor + in_l

# Main function caller
if __name__ == "__main__":
    main()
