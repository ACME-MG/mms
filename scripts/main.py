import sys; sys.path += [".."]
from mms.api import API

api = API(output_here=True)

api.read_data("gb_data.csv")

api.add_input("D0", ["linear"])
api.add_input("QD", ["linear"])
api.add_input("FN0", ["linear"])
api.add_input("QFN", ["linear"])
# api.add_input("T0", ["linear"])
# api.add_input("gamma", ["linear"])
api.add_input("Nc", ["linear"])
api.add_input("temperature", ["linear"])
api.add_input("stress", ["linear"])

api.add_output("time_to_tertiary", ["log", "linear"])

api.set_surrogate("pt_nn")
api.train(1000, epochs=1000, batch_size=64)

api.test(100)
