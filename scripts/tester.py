"""
 Title:         The PyTorch Surrogate for the VoceSlipHardening + AsaroInelasticity + Damage Model
 Description:   Tests the surrogate
 Author:        Janzen Choi

"""

# Libraries
import torch, math, os
from copy import deepcopy
import sys; sys.path += [".."]
from mms.helper.general import transpose
from mms.analyser.pole_figure import get_lattice, IPF
from mms.analyser.plotter import define_legend, save_plot, Plotter

# Constants
# DIRECTORY = "240830134116_617_s3"
DIRECTORY = sorted([item for item in os.listdir("results") if os.path.isdir(os.path.join("results", item))])[-1]
SUR_PATH = f"results/{DIRECTORY}/sm.pt"
MAP_PATH = f"results/{DIRECTORY}/map.csv"
EXP_PATH = "data/617_s3_exp.csv"
SIM_PATH = "data/617_s3_summary.csv"

# Define model parameters
# TAU_S, B, TAU_0, N = 825, 2, 112, 15
TAU_S, B, TAU_0, N = 799.61, 0.63363, 149.7, 14.604
# grain_ids = [125, 159, 189, 101, 249]
grain_ids = [61, 71, 74, 82, 130, 152, 155]

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

    def get_response(self, tau_s:float, b:float, tau_0:float, n:float) -> dict:
        """
        Gets the response of the model from the parameters

        Parameters:
        * `tau_s`: VoceSlipHardening parameter
        * `b`:     VoceSlipHardening parameter
        * `tau_0`: VoceSlipHardening parameter
        * `n`:     AsaroInelasticity parameter
        
        Returns the response as a dictionary
        """

        # Initialise
        param_list = [tau_s, b, tau_0, n]
        response_dict = deepcopy(self.response_dict)
        
        # Get outputs and combine
        for strain in response_dict["strain"][1:]:
            output_dict = self.get_output(param_list + [strain])
            if output_dict == None:
                return
            for key in output_dict.keys():
                response_dict[key].append(output_dict[key])
        
        # Adjust and return
        if "average_stress" in response_dict.keys():
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

# Get all results
sim_dict = csv_to_dict(SIM_PATH)
model = Model(SUR_PATH, MAP_PATH, EXP_PATH)
res_dict = model.get_response(TAU_S, B, TAU_0, N)
exp_dict = csv_to_dict(EXP_PATH)

# Reformat reorientation trajectories
# grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in res_dict.keys() if "phi_1" in key]
# grain_ids = [141, 103, 230, 186, 93, 66, 183, 106, 192, 254, 241, 242, 95, 188, 228, 180, 99, 112, 243, 113, 46, 129, 207, 256, 49, 89, 197, 86, 134, 244, 136, 119, 128, 101, 47, 68, 303, 38, 110, 56, 155, 152, 169, 73, 74, 53, 214, 269, 88, 163, 235, 71, 42, 211, 32, 282, 48, 265, 156, 279, 257, 81, 84, 237, 204, 283, 107, 224, 239, 284, 149, 266, 231, 278, 58, 57, 80, 131, 216, 33, 83, 104, 158, 233, 41, 137, 82, 179, 182, 154, 258, 7, 167, 286, 59, 54, 21, 36, 193, 22, 274, 306, 187, 246, 191, 45, 255, 126, 190, 111, 252, 8, 299, 249, 226, 130, 273, 75, 304, 312, 97, 87, 271, 245, 76, 276, 50, 100, 292, 260, 289, 215, 151, 19, 248, 122, 209, 184, 309, 177, 25, 159, 24, 262, 30, 108, 240, 146, 217, 79, 250, 125, 77, 67, 234, 302, 206, 117, 295, 115, 272, 51, 114, 189, 91, 208, 102, 142, 118, 175, 109, 285, 90, 35, 143, 44, 212, 13, 300, 293, 196, 40, 263, 210, 27, 287, 236, 229, 12, 20, 205, 39, 14, 147, 308, 259, 251, 202, 16, 280, 315, 153, 277, 264, 150, 140, 70, 43, 275, 29, 268, 11, 37, 28, 166, 200, 34, 161, 198, 174, 85, 121, 247, 96, 60, 203, 55, 9, 281, 238, 253, 178, 120, 64, 31, 261, 123, 185, 133, 173, 218, 144, 10, 294, 135, 148, 23, 232, 199, 213, 61, 69, 176, 78, 164, 132, 181, 288, 220, 201, 52, 165, 63, 195, 219, 313, 172, 225, 72, 116, 227, 26, 291, 92, 270, 157, 194, 138, 223, 222, 221, 105, 162, 139]
# grain_ids = grain_ids[:30]
get_trajectories = lambda dict : [transpose([dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]) for grain_id in grain_ids]
sim_trajectories = get_trajectories(sim_dict)
res_trajectories = get_trajectories(res_dict)
exp_trajectories = get_trajectories(exp_dict)

# Initialise IPF
ipf = IPF(get_lattice("fcc"))
direction = [1,0,0]

# Plot experimental reorientation trajectories
ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2})
ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015})
ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2})
for exp_trajectory, grain_id in zip(exp_trajectories, grain_ids):
    ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})

# # Plot simulated reorientation trajectories
# ipf.plot_ipf_trajectory(sim_trajectories, direction, "plot", {"color": "blue", "linewidth": 1, "zorder": 3})
# ipf.plot_ipf_trajectory(sim_trajectories, direction, "arrow", {"color": "blue", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
# ipf.plot_ipf_trajectory([[st[0]] for st in sim_trajectories], direction, "scatter", {"color": "blue", "s": 6**2, "zorder": 3})

# Plot surrogate reorientation trajectories
ipf.plot_ipf_trajectory(res_trajectories, direction, "plot", {"color": "red", "linewidth": 1, "zorder": 3})
ipf.plot_ipf_trajectory(res_trajectories, direction, "arrow", {"color": "red", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
ipf.plot_ipf_trajectory([[rt[0]] for rt in res_trajectories], direction, "scatter", {"color": "red", "s": 6**2, "zorder": 3})

# Save IPF
# define_legend(["silver", "blue", "red"], ["Experimental", "CPFE", "Surrogate"], ["scatter", "line", "line"])
define_legend(["silver", "red"], ["Experimental", "Surrogate"], ["scatter", "line"])
save_plot("plot_rt.png")

# Plot stress-strain curve
if "stress" in res_dict.keys():
    plotter = Plotter("strain", "stress", "mm/mm", "MPa")
    plotter.prep_plot()
    plotter.scat_plot(exp_dict, "silver", "Experimental")
    plotter.line_plot({"strain": sim_dict["average_strain"], "stress": sim_dict["average_stress"]}, "blue", "CPFE")
    plotter.line_plot(res_dict, "red", "Surrogate")
    plotter.set_legend()
    save_plot("plot_ss.png")
