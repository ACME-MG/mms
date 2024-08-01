import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface(f"617_s1_phi")
itf.read_data(f"617_s1_phi.csv")

input_list = ["tau_sat", "b", "tau_0", "gamma_0", "n"]
grain_indexes = [56, 346, 463, 568, 650] # [75, 189, 314, 338, 463]
output_list = [f"g{i}_{strain_interval}_{label}" for i in grain_indexes for label in ["phi_1", "Phi", "phi_2"]
               for strain_interval in ["0p2", "0p4", "0p6", "0p8", "1p0"]]
for input in input_list:
    itf.add_input(input, ["log", "linear"])
for output in output_list:
    itf.add_output(output, ["log", "linear"])

# itf.define("simple", epochs=1000, batch_size=32, verbose=True)
# itf.add_training_data(0.9)
# itf.add_validation_data(0.1)
itf.define("kfold", num_splits=5, epochs=1000, batch_size=32, verbose=True)
itf.add_training_data()
itf.train()
itf.plot_loss_history()

itf.get_validation_data()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation()
itf.export_validation()

itf.save(f"617_s1_phi")
itf.export_maps(f"617_s1_phi")
