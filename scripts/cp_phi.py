import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface(f"phi")
itf.read_data(f"phi.csv")

for input in ["tau_sat", "b", "tau_0", "gamma_0", "n"]:
    itf.add_input(input, ["log", "linear"])

for output in [f"g{i+1}_{label}_{pos}" for i in range(5) for label in ["phi_1", "Phi", "phi_2"] for pos in ["start", "end"]]:
    itf.add_output(output, ["log", "linear"])

itf.add_training_data(0.9)
itf.add_validation_data()

itf.train(epochs=1000, batch_size=32, verbose=True)

itf.plot_loss_history()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(use_log=True)
itf.export_validation()

itf.save(f"phi")
itf.export_maps(f"phi")
