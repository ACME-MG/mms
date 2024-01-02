import sys; sys.path += [".."]
from mms.interface import Interface

itf = Interface("gb_critical")
itf.read_data("gb_critical.csv")

itf.remove_data("realisation", 1)
itf.remove_data("realisation", 2)
itf.remove_data("realisation", 3)
itf.remove_data("realisation", 4)

itf.add_input("D", ["log", "linear"])
itf.add_input("FN", ["log", "linear"])

itf.add_input("temperature", ["linear"])
itf.add_input("stress", ["linear"])

itf.add_output("time_primary", ["log", "linear"])
itf.add_output("strain_primary", ["log", "linear"])

itf.add_output("time_secondary", ["log", "linear"])
itf.add_output("strain_secondary", ["log", "linear"])

itf.add_output("time_tertiary", ["log", "linear"])
itf.add_output("strain_tertiary", ["log", "linear"])

itf.add_output("time_failure", ["log", "linear"])
itf.add_output("strain_failure", ["log", "linear"])

# itf.add_output("time_failure", ["log", "linear"])
# itf.add_output("strain_1", ["log", "linear"])
# itf.add_output("strain_2", ["log", "linear"])
# itf.add_output("strain_3", ["log", "linear"])
# itf.add_output("strain_4", ["log", "linear"])
# itf.add_output("strain_5", ["log", "linear"])
# itf.add_output("strain_6", ["log", "linear"])
# itf.add_output("strain_7", ["log", "linear"])
# itf.add_output("strain_8", ["log", "linear"])
# itf.add_output("strain_9", ["log", "linear"])
# itf.add_output("strain_10", ["log", "linear"])

itf.add_training_data(1000)
itf.add_validation_data(100)

itf.train(epochs=5000, batch_size=32)
itf.plot_loss_history()

itf.print_validation(use_log=True, print_table=False)
itf.plot_validation(use_log=True)
