"""
 Title:         Optimiser
 Description:   Optimiser for the PyTorch surrogate model
 Author:        Janzen Choi

"""

# Libraries
import itertools, math, time, torch
import matplotlib.pyplot as plt
import sys; sys.path += ["/home/janzen/code/mms"]
from mms.helper.io import csv_to_dict
from mms.helper.general import transpose
from mms.analyser.plotter import save_plot, define_legend, Plotter
from mms.analyser.pole_figure import get_lattice, IPF

# Constants
DIRECTORY = "results/240830134116_617_s3"
SUR_PATH = f"{DIRECTORY}/sm.pt"
MAP_PATH = f"{DIRECTORY}/map.csv"
EXP_PATH = "data/617_s3_exp.csv"

# Main function
def main() -> None:
    
    # Start timer
    start_time = time.time()
    
    # Read experimental data
    exp_dict = csv_to_dict(EXP_PATH)
    
    # Initialise surrogate and mapper
    surrogate = Surrogate(SUR_PATH)
    mapper = Mapper(MAP_PATH)
    
    # Get grain IDs and experimental trajectories
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in mapper.output_map_dict["param_name"] if "phi_1" in key]
    # grain_ids = [164, 173, 265, 213, 207]
    get_trajectories = lambda dict : [transpose([dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]) for grain_id in grain_ids]
    exp_trajectories = get_trajectories(exp_dict)
    
    # Initialise IPF
    ipf = IPF(get_lattice("fcc"))
    direction = [1,0,0]
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2})
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015})
    ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2})
    for exp_trajectory, grain_id in zip(exp_trajectories, grain_ids):
        ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})
    
    # Define parameters
    params_dict = {
        "tau_sat": [200, 400, 800, 1600],
        "b":       [1, 2, 4, 8, 16],
        "tau_0":   [100, 200, 400, 800],
        "n":       [1, 2, 4, 8, 16],
    }
    param_grid = get_combinations(params_dict)
    
    # Announce completion of setup
    duration = round(time.time() - start_time, 2)
    print(f"Completed surrogate setup ({duration}s)")
    
    # Iterate through parameters
    for i, param_list in enumerate(param_grid):
        
        # Evaluate the surrogate
        start_time = time.time()
        sim_dict = evaluate_sm(param_list, surrogate, mapper)
        for key in sim_dict.keys():
            sim_dict[key] = [exp_dict[key][0]] + sim_dict[key]
        sim_trajectories = get_trajectories(sim_dict)

        # Plot reorientation trajectories
        colour_list = ["red", "blue", "green", "orange", "purple"]
        for sim_trajectory, colour in zip(sim_trajectories, colour_list):
            ipf.plot_ipf_trajectory([sim_trajectory], direction, "plot", {"color": colour, "linewidth": 1, "zorder": 3})
            ipf.plot_ipf_trajectory([sim_trajectory], direction, "arrow", {"color": colour, "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
            ipf.plot_ipf_trajectory([[sim_trajectory[0]]], direction, "scatter", {"color": colour, "s": 6**2, "zorder": 3})
        # ipf.plot_ipf_trajectory(sim_trajectories, direction, "plot", {"color": "red", "linewidth": 1, "zorder": 3})
        # ipf.plot_ipf_trajectory(sim_trajectories, direction, "arrow", {"color": "red", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
        # ipf.plot_ipf_trajectory([[st[0] for st in sim_trajectories]], direction, "scatter", {"color": "red", "s": 6**2, "zorder": 3})
        if i % 10 == 0:
            ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2, "zorder": 4})
            ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015, "zorder": 4})
            ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2, "zorder": 4})
            for exp_trajectory, grain_id in zip(exp_trajectories, grain_ids):
                ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id, "zorder": 4})
            define_legend(["silver", "black"], ["Experimental", "Surrogate"], ["scatter", "scatter"])
            plt.savefig("plot_rt.png")

        # Print progress
        duration = round(time.time() - start_time, 2)
        print(f"Evaluated {i+1}/{len(param_grid)}\t{param_list}\t({duration}s)")

def get_combinations(params_dict:dict) -> list:
    """
    Returns a list of possible combinations of a set of parameters
    
    Parameters:
    * `params_dict`: Dictionary of parameter lists

    Returns the list of parameter combinations
    """
    param_list = list(params_dict.values())
    combinations = list(itertools.product(*param_list))
    combinations = [list(c) for c in combinations]
    return combinations

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

def evaluate_sm(param_list:list, surrogate:Surrogate, mapper:Mapper) -> dict:
    """
    Quickly evaluates the surrogate model given a list of parameter values
    
    Parameters:
    * `param_list`:  List of parameter values
    * `surrogate`:   Surrogate object
    * `mapper`:      Mapper object
    
    Returns the evaluation as a dictionary
    """
    
    # Define strain
    strain_list = [0.01*i for i in range(1,31)]
    
    # Gets the outputs
    output_grid = []
    for strain in strain_list:
        input_list = mapper.map_input(param_list + [strain])
        input_tensor = torch.tensor(input_list)
        output_tensor = surrogate.forward(input_tensor)
        output_list = mapper.unmap_output(output_tensor.tolist())
        output_grid.append(output_list)
    output_grid = transpose(output_grid)
    
    # Converts the outputs into a dictionary
    sim_dict = {}
    for i, key in enumerate(mapper.output_map_dict["param_name"]):
        sim_dict[key] = output_grid[i]
    
    # Add stress and strain and return
    sim_dict["strain"] = strain_list
    sim_dict["stress"] = sim_dict.pop("average_stress")
    return sim_dict

# Main function caller
if __name__ == "__main__":
    main()
