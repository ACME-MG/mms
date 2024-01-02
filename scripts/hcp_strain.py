import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface("hcp_strain_375")
itf.read_data("hcp_strain_375.csv")

itf.add_input("VSH_TAU_0",   ["log", "linear"])
itf.add_input("VSH_TAU_SAT", ["log", "linear"])
itf.add_input("VSH_B",       ["log", "linear"])
itf.add_input("AI_N",        ["log", "linear"])
itf.add_input("AI_GAMMA0",   ["log", "linear"])

itf.add_output("x_end", ["log", "linear"])
itf.add_output("y_1", ["log", "linear"])
itf.add_output("y_2", ["log", "linear"])
itf.add_output("y_3", ["log", "linear"])
itf.add_output("y_4", ["log", "linear"])
itf.add_output("y_5", ["log", "linear"])
itf.add_output("y_6", ["log", "linear"])
itf.add_output("y_7", ["log", "linear"])
itf.add_output("y_8", ["log", "linear"])
itf.add_output("y_9", ["log", "linear"])
itf.add_output("y_10", ["log", "linear"])
itf.add_output("y_11", ["log", "linear"])
itf.add_output("y_12", ["log", "linear"])
itf.add_output("y_13", ["log", "linear"])
itf.add_output("y_14", ["log", "linear"])
itf.add_output("y_15", ["log", "linear"])
itf.add_output("y_16", ["log", "linear"])
itf.add_output("y_17", ["log", "linear"])
itf.add_output("y_18", ["log", "linear"])
itf.add_output("y_19", ["log", "linear"])
itf.add_output("y_20", ["log", "linear"])

itf.add_training_data(338)
itf.add_validation_data(37)

itf.train(epochs=10000, batch_size=32, verbose=True)
itf.plot_loss_history()

itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(use_log=True)
itf.export_validation()

itf.save()
