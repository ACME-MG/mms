"""
 Title:         The PyTorch Surrogate for the VoceSlipHardening + AsaroInelasticity + Damage Model
 Description:   Tests the surrogate
 Author:        Janzen Choi

"""

# Libraries
import torch, math
import matplotlib.pyplot as plt
from copy import deepcopy
import sys; sys.path += [".."]
from mms.helper.general import transpose
from mms.analyser.pole_figure import get_lattice, IPF
from mms.analyser.plotter import define_legend, save_plot

# Constants
SUR_PATH = "results/240827094818_617_s3/sm.pt"
MAP_PATH = "results/240827094818_617_s3/map.csv"
EXP_PATH = "data/617_s3_exp.csv"
# SIM_PATH = "data/617_s3_summary.csv"
SIM_PATH = "data/summary.csv"

def csv_to_dict(csv_path:str, delimeter:str=",") -> dict:
    """
    Converts a CSV file into a dictionary
    
    Parameters:
    * `csv_path`:  The path to the CSV file
    * `delimeter`: The separating character
    
    Returns the dictionary
    """

    # Read all data from CSV (assume that file is not too big)
    csv_fh = open(csv_path, "r", encoding="utf-8-sig")
    csv_lines = csv_fh.readlines()
    csv_fh.close()

    # Initialisation for conversion
    csv_dict = {}
    headers = csv_lines[0].replace("\n", "").split(delimeter)
    csv_lines = csv_lines[1:]
    for header in headers:
        csv_dict[header] = []

    # Start conversion to dict
    for csv_line in csv_lines:
        csv_line_list = csv_line.replace("\n", "").split(delimeter)
        for i in range(len(headers)):
            value = csv_line_list[i]
            if value == "":
                continue
            try:
                value = float(value)
            except:
                pass
            csv_dict[headers[i]].append(value)
    
    # Convert single item lists to items and things multi-item lists
    for header in headers:
        if len(csv_dict[header]) == 1:
            csv_dict[header] = csv_dict[header][0]
    
    # Return
    return csv_dict

def linear(value:float, map:dict, mapper, index:int) -> float:
    """
    Linearly maps or unmaps a value

    Parameters:
    * `value`:  The value to be mapped / unmapped
    * `map`:    The mapping information
    * `mapper`: The mapping function handler
    * `index`:  The index of the map
    """
    return mapper(
        value = value,
        in_l  = map["in_l_bound"][index],
        in_u  = map["in_u_bound"][index],
        out_l = map["out_l_bound"][index],
        out_u = map["out_u_bound"][index],
    )

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

def get_sm_info(sm_path:str, map_path:str) -> tuple:
    """
    Loads the model and maps given two paths

    Parameters:
    * `sm_path`:    Path to the surrogate model
    * `map_path`:   Path to the map

    Returns the surrogate model, the input map, and the output map
    """

    # Load surrogate model
    model = torch.load(sm_path)
    model.eval()
    
    # Load maps
    input_map_dict, output_map_dict = {}, {}
    map_dict = csv_to_dict(map_path)
    num_inputs = map_dict["param_type"].count("input")
    for key in map_dict.keys():
        input_map_dict[key] = map_dict[key][:num_inputs]
        output_map_dict[key] = map_dict[key][num_inputs:]

    # Return everything
    return model, input_map_dict, output_map_dict

# Class for the model
class Model:
    
    def __init__(self, sm_path:str, map_path:str, exp_path:str) -> None:
        """
        Constructor for the surrogate model

        Parameters:
        * `sm_path`:  The path to the surrogate model
        * `map_path`: The path to the mapping methods
        * `exp_path`: The path to the experimental data
        """
        
        # Get model and maps
        self.model, self.input_map, self.output_map = get_sm_info(sm_path, map_path)
        
        # Extract experimental information
        exp_dict = csv_to_dict(exp_path)
        # import numpy as np
        # self.response_dict = {"strain": np.linspace(0,1.0,100)}
        self.response_dict = {"strain": exp_dict["strain_intervals"]}
        self.response_dict["strain_intervals"] = self.response_dict["strain"]
        for param_name in self.output_map["param_name"]:
            is_orientation = "phi_1" in param_name or "Phi" in param_name or "phi_2" in param_name
            initial_value = exp_dict[param_name][0] if is_orientation else 0.0
            self.response_dict[param_name] = [initial_value]

    def get_response(self, tau_sat:float, b:float, tau_0:float, n:float) -> dict:
        """
        Gets the response of the model from the parameters

        Parameters:
        * `tau_sat`: VoceSlipHardening parameter
        * `b`:       VoceSlipHardening parameter
        * `tau_0`:   VoceSlipHardening parameter
        * `n`:       AsaroInelasticity parameter
        
        Returns the response as a dictionary
        """

        # Initialise
        param_list = [tau_sat, b, tau_0, n]
        response_dict = deepcopy(self.response_dict)
        
        # Get outputs and combine
        for strain in response_dict["strain"][1:]:
            output_dict = self.get_output(param_list + [strain])
            if output_dict == None:
                return
            for key in output_dict.keys():
                response_dict[key].append(output_dict[key])
        
        # Adjust and return
        response_dict["stress"] = response_dict.pop("average_stress")
        return response_dict

    def get_output(self, input_list:list) -> dict:
        """
        Gets the outputs of the surrogate model

        Parameters:
        * `input_list`: The list of raw input values

        Returns the outputs
        """
        
        # Process inputs
        processed_input_list = []
        for i in range(len(input_list)):
            try:
                input_value = math.log(input_list[i]) / math.log(self.input_map["base"][i])
                input_value = linear(input_value, self.input_map, linear_map, i)
            except ValueError:
                return None
            processed_input_list.append(input_value)
        
        # Get raw outputs and process
        input_tensor = torch.tensor(processed_input_list)
        with torch.no_grad():
            output_list = self.model(input_tensor).tolist()
        for i in range(len(output_list)):
            try:
                output_list[i] = linear(output_list[i], self.output_map, linear_unmap, i)
                output_list[i] = math.pow(self.output_map["base"][i], output_list[i])
            except ValueError:
                return None
        
        # Return the dictionary of outputs
        output_dict = dict(zip(self.output_map["param_name"], output_list))
        return output_dict

# Define model parameters
tau_sat  = 825
b        = 2
tau_0    = 112
n        = 15

# Get all results
sim_dict = csv_to_dict(SIM_PATH)
model = Model(SUR_PATH, MAP_PATH, EXP_PATH)
res_dict = model.get_response(tau_sat, b, tau_0, n)

# Reformat reorientation trajectories
grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in res_dict.keys() if "phi_1" in key]
grain_ids = [23, 53, 71, 79, 238]
get_trajectories = lambda dict : [transpose([dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]) for grain_id in grain_ids]
sim_trajectories = get_trajectories(sim_dict)
res_trajectories = get_trajectories(res_dict)

# Plot stress-strain curve
plt.figure(figsize=(5,5))
plt.gca().set_position([0.17, 0.12, 0.75, 0.75])
plt.gca().grid(which="major", axis="both", color="SlateGray", linewidth=1, linestyle=":")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Stress (MPa)")
plt.plot(sim_dict["average_strain"], sim_dict["average_stress"], color="blue", label="CPFE")
plt.plot(res_dict["strain"], res_dict["stress"], color="red", label="Surrogate")
plt.legend(framealpha=1, edgecolor="black", fancybox=True, facecolor="white")
save_plot("plot_ss.png")

# Plot reorientation trajectories
ipf = IPF(get_lattice("fcc"))
direction = [1,0,0]
ipf.plot_ipf_trajectory(sim_trajectories, direction, "plot", {"color": "blue", "linewidth": 2})
ipf.plot_ipf_trajectory(sim_trajectories, direction, "arrow", {"color": "blue", "head_width": 0.01, "head_length": 0.015})
ipf.plot_ipf_trajectory([[st[0]] for st in sim_trajectories], direction, "scatter", {"color": "blue", "s": 8**2})
for sim_trajectory, grain_id in zip(sim_trajectories, grain_ids):
    ipf.plot_ipf_trajectory([[sim_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})
ipf.plot_ipf_trajectory(res_trajectories, direction, "plot", {"color": "red", "linewidth": 1, "zorder": 3})
ipf.plot_ipf_trajectory(res_trajectories, direction, "arrow", {"color": "red", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
ipf.plot_ipf_trajectory([[rt[0]] for rt in res_trajectories], direction, "scatter", {"color": "red", "s": 6**2, "zorder": 3})
define_legend(["blue", "red"], ["CPFE", "Surrogate"], ["line", "line"])
save_plot("plot_rt.png")
