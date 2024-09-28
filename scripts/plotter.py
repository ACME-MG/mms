"""
 Title:         Assess
 Description:   Plots the reorientaiton trajectories
 Author:        Janzen Choi

"""

# Libraries
import math, torch
import sys; sys.path += ["/home/janzen/code/mms"]
from mms.helper.io import csv_to_dict
from mms.helper.general import transpose
from mms.analyser.plotter import save_plot, define_legend, Plotter
from mms.analyser.pole_figure import get_lattice, IPF

# Paths
# DIRECTORY = "results/240830134116_617_s3"
DIRECTORY = "results/240905132323_617_s3"
SUR_PATH = f"{DIRECTORY}/sm.pt"
MAP_PATH = f"{DIRECTORY}/map.csv"
EXP_PATH = "data/617_s3_exp.csv"

# Other parameters
PARAM_LIST    = [741.2, 0.68945, 146.68, 15.709]
# PARAM_LIST    = [799.61, 0.63363, 149.7, 14.604]
CAL_GRAIN_IDS = [125, 159, 189, 101, 249]
VAL_GRAIN_IDS = [207, 57, 231, 155, 271]

# Main function
def main() -> None:
    
    # Initialise everything
    surrogate = Surrogate(SUR_PATH)
    mapper    = Mapper(MAP_PATH)
    exp_dict  = csv_to_dict(EXP_PATH)
    
    # Evaluate surrogate
    sur_dict = evaluate_sm(PARAM_LIST, surrogate, mapper)
    
    # Get trajectories
    get_trajectories = lambda dict, grain_ids : [transpose([dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]) for grain_id in grain_ids]
    exp_trajectories = get_trajectories(exp_dict, CAL_GRAIN_IDS + VAL_GRAIN_IDS)
    cal_trajectories = get_trajectories(sur_dict, CAL_GRAIN_IDS)
    val_trajectories = get_trajectories(sur_dict, VAL_GRAIN_IDS)
    
    # Initialise IPF
    ipf = IPF(get_lattice("fcc"))
    direction = [1,0,0]
    
    # Plot experimental trajectories
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2})
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015})
    ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2})
    for exp_trajectory, grain_id in zip(exp_trajectories, CAL_GRAIN_IDS + VAL_GRAIN_IDS):
        ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})
    
    # Plot calibration trajectories
    ipf.plot_ipf_trajectory(cal_trajectories, direction, "plot", {"color": "green", "linewidth": 1, "zorder": 3})
    ipf.plot_ipf_trajectory(cal_trajectories, direction, "arrow", {"color": "green", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
    ipf.plot_ipf_trajectory([[ct[0] for ct in cal_trajectories]], direction, "scatter", {"color": "green", "s": 6**2, "zorder": 3})
    
    # Plot valibration trajectories
    ipf.plot_ipf_trajectory(val_trajectories, direction, "plot", {"color": "red", "linewidth": 1, "zorder": 3})
    ipf.plot_ipf_trajectory(val_trajectories, direction, "arrow", {"color": "red", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
    ipf.plot_ipf_trajectory([[vt[0] for vt in val_trajectories]], direction, "scatter", {"color": "red", "s": 6**2, "zorder": 3})
    
    # Format and save IPF
    define_legend(["silver", "green", "red"], ["Experimental", "Calibration", "Validation"], ["scatter", "scatter", "scatter"])
    save_plot("plot_rt.png")
    
    # Plot stress-strain response
    plotter = Plotter("strain", "stress", "mm/mm", "MPa")
    plotter.prep_plot()
    plotter.scat_plot(exp_dict, "silver", "Experimental")
    plotter.line_plot(sur_dict, "green", "Calibration")
    plotter.set_legend()
    save_plot("plot_ss.png")
    
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
