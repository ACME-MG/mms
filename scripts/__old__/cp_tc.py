import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface("tc")
itf.read_data("tc.csv")
itf.define("simple", epochs=50, batch_size=32, verbose=True)

input_list = ["tau_sat", "b", "tau_0", "gamma_0", "n"]
output_list = ["x_end"] + [f"y_{i+1}" for i in range(30)]

for input in input_list:
    itf.add_input(input, ["log", "linear"])

for output in output_list:
    itf.add_output(output, ["log", "linear"])

itf.add_training_data(0.9)
itf.add_validation_data()

itf.train()
itf.plot_loss_history()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(use_log=True)
itf.export_validation()

itf.save("tc")
itf.export_maps("tc")
