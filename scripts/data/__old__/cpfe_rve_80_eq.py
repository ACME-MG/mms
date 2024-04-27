import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface("cpfe_rve_80_eq")
itf.read_data("cpfe_rve_80_eq.csv")

itf.add_input("tau_s",   ["log", "linear"])
itf.add_input("b",       ["log", "linear"])
itf.add_input("tau_0",   ["log", "linear"])
itf.add_input("gamma_0", ["log", "linear"])
itf.add_input("n",       ["log", "linear"])

for i in range(50):
    itf.add_output(f"y_{i+1}", ["log", "linear"])

itf.add_training_data(450)
itf.add_validation_data(50)

itf.train(epochs=10000, batch_size=32, verbose=True)

itf.plot_loss_history()
itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(use_log=True)
itf.export_validation()

itf.save("cpfe_rve_80_eq")
itf.export_maps("cpfe_rve_80_eq")
