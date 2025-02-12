"""
 Title:         617_s3_bulk
 Description:   Trains a surrogate model for approximating bulk responses
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from mms.interface import Interface
from mms.helper.io import csv_to_dict

# Read data
itf = Interface()
data_file = "617_s3_sampled_lhs.csv"
itf.read_data(data_file)

# Define grain IDs
# grain_ids = [98, 123, 140, 196] # bad
# grain_ids = [164, 173, 265, 213, 207]
# grain_ids = [12, 16, 28, 44, 45, 67, 68, 72, 76, 78, 79, 80, 82, 87, 90, 110, 120, 147, 155, 158, 164, 179, 192, 207, 216, 242, 250, 265, 269, 271, 283, 299, 302]
data_dict = csv_to_dict(f"data/{data_file}")
grain_ids = [int(key.replace("_phi_1","").replace("g","")) for key in data_dict.keys() if "_phi_1" in key]
# for grain_id in [98, 123, 140, 196]:
#     grain_ids.remove(grain_id)
# grain_ids = grain_ids[:10]

# Define input and output fields
input_list = ["cp_tau_s", "cp_b", "cp_tau_0", "cp_n", "average_strain"]
bulk_output_list = ["average_stress"]
grain_output_list = [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
output_list = bulk_output_list + grain_output_list

# Scale input and outputs
for input in input_list:
    itf.add_input(input, ["log", "linear"])
for output in output_list:
    # itf.add_output(output, ["log", "linear"])
    itf.add_output(output, ["linear"])

# Train surrogate model
# itf.define_surrogate("kfold", num_splits=5, epochs=2000, batch_size=32, verbose=True)
itf.define_surrogate("kfold_2", num_splits=5, epochs=2000, batch_size=64, verbose=True)
# itf.define_surrogate("kfold_2", num_splits=5, epochs=500, batch_size=64, verbose=True)
# itf.define_surrogate("kfold_geodesic", num_splits=5, epochs=2000, batch_size=64, verbose=True, geodesic_weight=200)
itf.add_training_data()
itf.train()
itf.plot_loss_history()

# Validate the trained model
itf.get_validation_data()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(
    headers   = ["average_stress"],
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

# Save surrogate model and mapping
itf.save("sm")
itf.export_maps("map")
