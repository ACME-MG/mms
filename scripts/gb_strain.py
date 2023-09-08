import sys; sys.path += [".."]
from mms.api import API

api = API("gb_strain")
api.read_data("gb_strain.csv")

api.remove_data("realisation", 1)
api.remove_data("realisation", 2)
api.remove_data("realisation", 3)
api.remove_data("realisation", 4)

api.add_input("D", ["log", "linear"])
api.add_input("FN", ["log", "linear"])

api.add_input("temperature", ["linear"])
api.add_input("stress", ["linear"])

api.add_output("time_failure", ["log", "linear"])
api.add_output("strain_1", ["log", "linear"])
api.add_output("strain_2", ["log", "linear"])
api.add_output("strain_3", ["log", "linear"])
api.add_output("strain_4", ["log", "linear"])
api.add_output("strain_5", ["log", "linear"])
api.add_output("strain_6", ["log", "linear"])
api.add_output("strain_7", ["log", "linear"])
api.add_output("strain_8", ["log", "linear"])
api.add_output("strain_9", ["log", "linear"])
api.add_output("strain_10", ["log", "linear"])

api.add_training_data(1000)
api.add_validation_data(100)

api.train(epochs=5000, batch_size=32)
api.plot_loss_history()

api.print_validation(use_log=True, print_table=False)
api.plot_validation(use_log=True)
