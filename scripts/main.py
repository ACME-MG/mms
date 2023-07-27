import sys; sys.path += [".."]
from mms.api import API

api = API(output_here=True)

input_list  = ["D0", "QD", "FN0", "QFN", "T0", "gamma", "Nc", "temperature", "stress"]
output_list = ["time_to_tertiary"]

api.read_data("gb_data.csv", input_list, output_list)

api.define_surrogate("pt_nn")
api.train(4000, epochs=1000, batch_size=16)

api.predict(100)
