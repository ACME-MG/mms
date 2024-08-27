"""
 Title:         617_s3_bulk
 Description:   Trains a surrogate model for approximating bulk responses
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from mms.interface import Interface

# Read data
itf = Interface()
itf.read_data("617_s3_sampled.csv")

# Define grain IDs
grain_ids = [15, 23, 45, 80, 101]

# Define input and output fields
input_list = ["cp_tau_s", "cp_b", "cp_tau_0", "cp_n", "average_strain"]
bulk_output_list = ["average_stress", "average_elastic"]
grain_output_list = [f"g{grain_id}_{field}" for grain_id in grain_ids
                     for field in ["stress", "elastic", "phi_1", "Phi", "phi_2"]]
output_list = bulk_output_list + grain_output_list

# Scale input and outputs
for input in input_list:
    itf.add_input(input, ["log", "linear"])
for output in output_list:
    itf.add_output(output, ["log", "linear"])

# Train surrogate model
itf.define("kfold", num_splits=5, epochs=1000, batch_size=32, verbose=True)
itf.add_training_data()
itf.train()
itf.plot_loss_history()

# Validate the trained model
itf.get_validation_data()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(["average_stress"], label="Stress (MPa)", use_log=False)
itf.plot_validation(["average_elastic"], label="Elastic Strain (mm/mm)", use_log=False)
itf.plot_validation([f"g{grain_id}_stress" for grain_id in grain_ids], label="Stress (MPa)", use_log=False)
itf.plot_validation([f"g{grain_id}_elastic" for grain_id in grain_ids], label="Elastic Strain (mm/mm)", use_log=False)
itf.plot_validation([f"g{grain_id}_{field}" for grain_id in grain_ids
                     for field in ["phi_1", "Phi", "phi_2"]], label="Orientation (rads)", use_log=False)
itf.export_validation()

# Save surrogate model and mapping
itf.save("sm")
itf.export_maps("map")
