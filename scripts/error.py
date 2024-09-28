"""
 Title:         Error
 Description:   Calculates the surrogate's errors
 Author:        Janzen Choi

"""

# Libraries
import math, numpy as np, torch
import matplotlib.pyplot as plt
import sys; sys.path += ["/home/janzen/code/mms"]
from mms.helper.io import csv_to_dict
from mms.helper.general import transpose, sort_dict

# Paths
DIRECTORY = "results/240905132323_617_s3 (all grains)"
# DIRECTORY = "results/240920161852_617_s3"
SUR_PATH = f"{DIRECTORY}/sm.pt"
MAP_PATH = f"{DIRECTORY}/map.csv"
# EXP_PATH = "data/617_s3_exp.csv"
SUM_PATH = "data/617_s3_sampled.csv"

# Main function
def main() -> None:
    
    # Initialise everything
    surrogate = Surrogate(SUR_PATH)
    mapper    = Mapper(MAP_PATH)
    sum_dict  = csv_to_dict(SUM_PATH)
    
    # Define inputs / outputs
    input_fields  = ["cp_tau_s", "cp_b", "cp_tau_0", "cp_n", "average_strain"]
    inputs_list   = transpose([sum_dict[field] for field in input_fields])
    output_fields = mapper.output_map_dict["param_name"]
    outputs_list  = transpose([sum_dict[field] for field in output_fields])

    # Calculate errors for input-output pairs 
    err_dict = dict(zip(output_fields, [[] for _ in range(len(output_fields))]))
    for inputs, exp_outputs in zip(inputs_list, outputs_list):
        prd_outputs = evaluate_sm(inputs, surrogate, mapper)
        for field, exp_output, prd_output in zip(output_fields, exp_outputs, prd_outputs):
            err_dict[field].append(abs((exp_output-prd_output)/exp_output))
    
    # Reduce errors based on grains
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in output_fields if "phi_1" in key]
    err_red_dict = {}
    for grain_id in grain_ids:
        err_phi_1 = err_dict[f"g{grain_id}_phi_1"]
        err_Phi   = err_dict[f"g{grain_id}_Phi"]
        err_phi_2 = err_dict[f"g{grain_id}_phi_2"]
        err_red_dict[grain_id] = np.average(err_phi_1 + err_Phi + err_phi_2)*100
    
    # Print error summary
    err_red_dict = sort_dict(err_red_dict)
    # for grain_id in err_red_dict.keys():
    #     print(f"Grain {grain_id}:\t{round_sf(err_red_dict[grain_id], 3)}%")
    
    # Plot errors
    plt.figure(figsize=(5,5))
    plt.gca().set_position([0.17, 0.12, 0.75, 0.75])
    plt.gca().grid(which="major", axis="both", color="SlateGray", linewidth=1, linestyle=":")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.yscale("log")
    plt.hist(list(err_red_dict.values()), bins=100, color="blue", alpha=0.7, edgecolor="black")
    plt.savefig("plot_err.png")
    
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
            mapped_input = math.log(input_list[i]) / math.log(self.input_map_dict["base"][i])
            mapped_input = linear_map(
                value = mapped_input,
                in_l  = self.input_map_dict["in_l_bound"][i],
                in_u  = self.input_map_dict["in_u_bound"][i],
                out_l = self.input_map_dict["out_l_bound"][i],
                out_u = self.input_map_dict["out_u_bound"][i],
            )
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
            unmapped_output = linear_unmap(
                value = output_list[i],
                in_l  = self.output_map_dict["in_l_bound"][i],
                in_u  = self.output_map_dict["in_u_bound"][i],
                out_l = self.output_map_dict["out_l_bound"][i],
                out_u = self.output_map_dict["out_u_bound"][i],
            )
            unmapped_output = math.pow(self.output_map_dict["base"][i], unmapped_output)
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

def evaluate_sm(input_list:list, surrogate:Surrogate, mapper:Mapper) -> list:
    """
    Quickly evaluates the surrogate model given a list of input values
    
    Parameters:
    * `input_list`:  List of input values
    * `surrogate`:   Surrogate object
    * `mapper`:      Mapper object
    
    Returns the evaluation as a list
    """
    input_list = mapper.map_input(input_list)
    input_tensor = torch.tensor(input_list)
    output_tensor = surrogate.forward(input_tensor)
    output_list = mapper.unmap_output(output_tensor.tolist())
    return output_list

# Main function caller
if __name__ == "__main__":
    main()
