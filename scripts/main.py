import sys; sys.path += [".."]
from mms.api import API

api = API(output_here=True)

api.read_data("gb_data.csv")

api.filter_data("realisation", 1)
api.filter_data("realisation", 2)
api.filter_data("realisation", 3)
api.filter_data("realisation", 4)

api.add_input("D", ["log", "linear"])
api.add_input("FN", ["log", "linear"])

api.add_input("temperature", ["linear"])
api.add_input("stress", ["linear"])

api.add_output("time_to_tertiary", ["log", "linear"])
# api.add_output("time_to_failure", ["log", "linear"])
# api.add_output("strain_to_failure", ["log", "linear"])

api.set_surrogate("pt_nn")
api.add_training_data(1000)
api.add_validation_data(100)

api.train(epochs=1000, batch_size=32)
api.test()
