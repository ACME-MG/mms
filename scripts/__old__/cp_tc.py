import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface("617_s1_tc")
itf.read_data("617_s1_tc.csv")

input_list = ["tau_sat", "b", "tau_0", "gamma_0", "n"]
output_list = ["x_end"] + [f"y_{i+1}" for i in range(30)]
for input in input_list:
    itf.add_input(input, ["log", "linear"])
for output in output_list:
    itf.add_output(output, ["log", "linear"])

itf.define("kfold", num_splits=5, epochs=1000, batch_size=32, verbose=True)
itf.add_training_data()
itf.train()
itf.plot_loss_history()

itf.get_validation_data()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation()
itf.export_validation()

itf.save("617_s1_tc")
itf.export_maps("617_s1_tc")
