"""
 Title:         Summariser with Surrogate Modeller
 Description:   Summarises sampled data and trains surrogate models
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
import math, numpy as np, os, re
from mms.interface import Interface
from mms.helper.interpolator import Interpolator
from mms.helper.general import transpose, round_sf
from mms.helper.io import csv_to_dict, dict_to_csv, safe_mkdir

# Simulation paths
RESULTS_DIR = "/mnt/c/Users/janzen/OneDrive - UNSW/PhD/results/moose_sim"
SAMPLED_PATHS = [f"{RESULTS_DIR}/{sp}" for sp in [
    "2024-11-10 (617_s3_40um_lh2_sm8)",
    "2024-11-11 (617_s3_40um_lh2_sm16)",
    "2024-11-13 (617_s3_40um_lh2_sm24)",
    "2024-11-04 (617_s3_40um_lh2_sm32)",
    "2024-11-17 (617_s3_40um_lh2_sm40)",
    "2024-12-08 (617_s3_40um_lh2_sm48)"
]]
NUM_STRAINS = [8, 16, 24, 32, 40, 48]

# Constants
PARAMS = [f"cp_lh_{i}" for i in range(2)] + ["cp_tau_0", "cp_n", "cp_gamma_0"]
# PARAMS = ["cp_tau_s", "cp_b", "cp_tau_0", "cp_n", "cp_gamma_0"]
STRAIN_FIELD = "average_strain"
STRESS_FIELD = "average_stress"
MAX_STRAIN = 1.0
GRAIN_IDS = [59, 63, 86, 237, 303] # calibration

def main():
    """
    Main function
    """

    # Iterate through samples
    for sampled_path in SAMPLED_PATHS:
        num_params = int(sampled_path.split("/")[-1].split("sm")[-1].replace(")",""))

        # Iterate through number of strains
        for num_strain in NUM_STRAINS:

            # Create results folder
            results_path = f"results/617_s3_lh2_sm{num_params}_ns{num_strain}"
            safe_mkdir(f"results/617_s3_lh2_sm{num_params}_ns{num_strain}")

            # Summarise and train
            summary_dict = summarise(sampled_path, num_strain)
            train(summary_dict, results_path, PARAMS, GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)
            
def summarise(sampled_path:str, num_strain:int) -> dict:
    """
    Summarises a collection of simulation results

    Parameters:
    * `sampled_path`: Path to the sampled simulation results
    * `num_strain`:   Number of strains to summarise with

    Returns the summarised results as a dictionary
    """
    
    # Identify paths to simulations
    dir_path_list = [f"{sampled_path}/{dir_path}" for dir_path in os.listdir(sampled_path)
                    if os.path.exists(f"{sampled_path}/{dir_path}/summary.csv")]

    # Read all summary files
    summary_path_list = [f"{dir_path}/summary.csv" for dir_path in dir_path_list]
    summary_dict_list = [csv_to_dict(summary_path) for summary_path in summary_path_list]
    param_dict_list = [get_param_dict(f"{dir_path}/params.txt") for dir_path in dir_path_list]

    # Process the dictionaries
    processed_dict_list = [process_data_dict(summary_dict, num_strain) for summary_dict in summary_dict_list]
    key_list = list(param_dict_list[0].keys()) + list(processed_dict_list[0].keys())
    super_processed_dict = dict(zip(key_list, [[] for _ in range(len(key_list))]))

    # Iterate through the results
    for processed_dict, param_dict in zip(processed_dict_list, param_dict_list):
        num_values = len(list(processed_dict.values())[0])
        for key in param_dict:
            super_processed_dict[key] += [round_sf(param_dict[key], 5)]*num_values
        for key in processed_dict:
            super_processed_dict[key] += round_sf(processed_dict[key], 5)
    
    # Return summary
    return super_processed_dict

def train(train_dict:dict, train_path:str, param_names:list, grain_ids:list,
          strain_field:str, stress_field:str):
    """
    Trains a surrogate model
    
    Parameters:
    * `train_dict`:   Dictionary containing training data
    * `train_path`:   Path to store the training results
    * `param_names`:  List of parameter names
    * `grain_ids`:    List of grain IDs to conduct the training
    * `strain_field`: Name of the field for the strain data
    * `stress_field`: Name of the field for the stress data
    """

    # Initialise interface
    itf = Interface(input_path=".", output_here=True, verbose=True)
    itf.__output_path__ = train_path
    itf.__get_output__ = lambda x : f"{itf.__output_path__}/{x}"

    # Read data
    sampled_data_path = f"{train_path}/sampled_data.csv"
    dict_to_csv(train_dict, sampled_data_path)
    itf.read_data(sampled_data_path)

    # Identify and scale inputs
    input_list = param_names + [strain_field]
    for input in input_list:
        itf.add_input(input, ["log", "linear"])
    
    # Identify and scale output
    output_list = [stress_field] + [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
    for output in output_list:
        itf.add_output(output, ["log", "linear"])

    # Train surrogate model
    itf.define_surrogate("kfold_2", num_splits=5, epochs=1000, batch_size=32, verbose=True)
    itf.add_training_data()
    itf.train()
    itf.plot_loss_history()

    # Save surrogate model and mapping
    itf.save("sm")
    itf.export_maps("map")

    # Validate the trained model
    itf.get_validation_data()
    itf.print_validation(use_log=True, print_table=False)
    itf.plot_validation(
        headers   = [stress_field],
        label     = "Stress (MPa)",
        use_log   = False,
        plot_file = "stress"
    )
    itf.plot_validation(
        headers   = [f"g{grain_id}_{phi}" for grain_id in grain_ids for phi in ["phi_1", "Phi", "phi_2"]],
        label     = "Orientation (rads)",
        use_log   = False,
        plot_file = "all_phi"
    )

def get_param_dict(params_path:str) -> dict:
    """
    Gets the parameters from a file
    
    Parameters:
    * `params_path`: The path to the parameters file
    
    Returns a dictionary of the parameter values
    """
    with open(params_path, "r") as fh:
        line_list = [line.replace("\n", "").replace(":", "") for line in fh.readlines()]
        param_dict = {line.split(" ")[0]: float(line.split(" ")[1]) for line in line_list
                      if line.split(" ")[0] in PARAMS}
        return param_dict

def get_trajectories(data_dict:dict) -> dict:
    """
    Gets the reorientation trajectories

    Parameters:
    * `data_dict`: The data dictionary
    
    Return trajectories as a dictoinary of lists of euler angles
    """
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in data_dict.keys() if "_phi_1" in key]
    trajectories = {}
    for grain_id in grain_ids:
        trajectory = [data_dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]
        trajectory = transpose(trajectory)
        trajectories[grain_id] = trajectory
    return trajectories

def convert_grain_ids(data_dict:dict, grain_map_path:str) -> dict:
    """
    Converts the grain IDs of a dictionary
    
    Parameters:
    * `data_dict`:      The dictionary
    * `grain_map_path`: The path to the grain map
    
    Returns the dictionary with renamed keys
    """
    
    # Initialise conversion
    grain_map = csv_to_dict(grain_map_path)
    new_data_dict = {}
    mesh_to_ebsd = dict(zip(grain_map["mesh_id"], grain_map["ebsd_id"]))

    # Iterate through keys
    for key in data_dict:
        if bool(re.match(r'^g\d+.*$', key)):
            key_list = key.split("_")
            mesh_id = int(key_list[0].replace("g",""))
            new_key = f"g{int(mesh_to_ebsd[mesh_id])}_{'_'.join(key_list[1:])}"
            new_data_dict[new_key] = data_dict[key]
        else:
            new_data_dict[key] = data_dict[key]
    
    # Return
    return new_data_dict

def fix_angle(angle:float, l_bound:float=0.0, u_bound:float=2*math.pi) -> float:
    """
    Fixes an angle between two bounds
    
    Parameters:
    * `angle`: The angle (rads)

    Returns the fixed angle
    """
    if abs(angle-l_bound) < 1e-4 or abs(angle-u_bound) < 1e-4:
        return angle
    range = u_bound - l_bound
    if l_bound < angle and angle < u_bound:
        return angle
    elif angle < l_bound:
        return fix_angle(angle+range, l_bound, u_bound)
    else:
        return fix_angle(angle-range, l_bound, u_bound)

def process_data_dict(data_dict:dict, num_strains:int=NUM_STRAINS) -> dict:
    """
    Processes the data in the data dictionary;
    normalises strain difference
    
    Parameters:
    * `data_dict`:   The dictionary of data
    * `num_strains`: The number of strains to intervaluate

    Returns the processed dictionary
    """
    
    # Prepare old and new strain values
    strain_list = data_dict[STRAIN_FIELD]
    max_strain = max([s for s in data_dict[STRAIN_FIELD] if s < MAX_STRAIN])
    new_strain_list = list(np.linspace(0, max_strain, num_strains+1)[1:])
    
    # Prepare fields
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in data_dict.keys() if "_phi_1" in key]
    euler_fields = [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
    field_list = [STRESS_FIELD] + euler_fields
    
    # Interpolate for each field and return
    processed_dict = {STRAIN_FIELD: new_strain_list}
    for field in field_list:
        field_itp = Interpolator(strain_list, data_dict[field], len(strain_list))
        new_list = field_itp.evaluate(new_strain_list)
        processed_dict[field] = new_list
    
    # Fix the domain of the euler-bunge angles
    for field in euler_fields:
        processed_dict[field] = [fix_angle(phi) for phi in processed_dict[field]]
    
    # Return processed dictionary
    return processed_dict

# Main function
if __name__ == "__main__":
    main()

