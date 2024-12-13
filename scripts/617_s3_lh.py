"""
 Title:         617_s3_bulk
 Description:   Trains a surrogate model for approximating bulk responses
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from mms.interface import Interface
from mms.helper.io import csv_to_dict

# Constants
NUM_LH = 2
# STRAIN_FIELD = "average_grain_strain"
# STRESS_FIELD = "average_grain_stress"
STRAIN_FIELD = "average_strain"
STRESS_FIELD = "average_stress"

# Read data
itf = Interface(f"617_s3_lh{NUM_LH}")
data_file = f"617_s3_40um_lh{NUM_LH}_sampled.csv"
itf.read_data(data_file)

# Define grain IDs
data_dict = csv_to_dict(f"data/{data_file}")
grain_ids = [int(key.replace("_phi_1","").replace("g","")) for key in data_dict.keys() if "_phi_1" in key]
grain_ids = [207, 79, 164, 167, 309]

# Define input and output fields
input_list = [f"cp_lh_{i}" for i in range(NUM_LH)] + ["cp_tau_0", "cp_n", "cp_gamma_0", STRAIN_FIELD]
grain_output_list = [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
output_list = [STRESS_FIELD] + grain_output_list

# Scale input and outputs
for input in input_list:
    itf.add_input(input, ["log", "linear"])
for output in output_list:
    itf.add_output(output, ["log", "linear"])
    # itf.add_output(output, ["linear"])

# Train surrogate model
# itf.define_surrogate("kfold_2", num_splits=5, epochs=1000, batch_size=32, verbose=True)
itf.define_surrogate("kfold_2", num_splits=5, epochs=2000, batch_size=64, verbose=True)
# itf.define_surrogate("kfold_geodesic", "cpu", num_splits=5, epochs=2000, batch_size=64, verbose=True, geodesic_weight=200)
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
    headers   = [STRESS_FIELD],
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
itf.export_validation()
