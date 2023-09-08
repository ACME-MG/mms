import sys; sys.path += [".."]
from mms.api import API

api = API("hcp_strain_375")
api.read_data("hcp_strain_375.csv")

api.add_input("VSH_TAU_0",   ["log", "linear"])
api.add_input("VSH_TAU_SAT", ["log", "linear"])
api.add_input("VSH_B",       ["log", "linear"])
api.add_input("AI_N",        ["log", "linear"])
api.add_input("AI_GAMMA0",   ["log", "linear"])

api.add_output("x_end", ["log", "linear"])
api.add_output("y_1", ["log", "linear"])
api.add_output("y_2", ["log", "linear"])
api.add_output("y_3", ["log", "linear"])
api.add_output("y_4", ["log", "linear"])
api.add_output("y_5", ["log", "linear"])
api.add_output("y_6", ["log", "linear"])
api.add_output("y_7", ["log", "linear"])
api.add_output("y_8", ["log", "linear"])
api.add_output("y_9", ["log", "linear"])
api.add_output("y_10", ["log", "linear"])
api.add_output("y_11", ["log", "linear"])
api.add_output("y_12", ["log", "linear"])
api.add_output("y_13", ["log", "linear"])
api.add_output("y_14", ["log", "linear"])
api.add_output("y_15", ["log", "linear"])
api.add_output("y_16", ["log", "linear"])
api.add_output("y_17", ["log", "linear"])
api.add_output("y_18", ["log", "linear"])
api.add_output("y_19", ["log", "linear"])
api.add_output("y_20", ["log", "linear"])

api.add_training_data(338)
api.add_validation_data(37)

api.train(epochs=10000, batch_size=32, verbose=True)
api.plot_loss_history()

api.print_validation(use_log=True, print_table=False)
api.plot_validation(use_log=True)
api.export_validation()

api.save()
