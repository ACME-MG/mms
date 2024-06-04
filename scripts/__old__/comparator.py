"""
 Title:         Comparison plotter
 Description:   Creates 1:1 plots for looking at the performance of the surrogate model
 Author:        Janzen Choi

"""

# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
FILE_PATH      = "prd_data.csv"
LABEL_FONTSIZE = 16
OTHER_FONTSIZE = 13
MARKER_SIZE    = 5
LINEWIDTH      = 1

# Tries to float cast a value
def try_float_cast(value:str) -> float:
    try:
        return float(value)
    except:
        return value

# Converts a header file into a dict of lists
def csv_to_dict(csv_path:str, delimeter:str=",") -> dict:

    # Read all data from CSV (assume that file is not too big)
    csv_fh = open(csv_path, "r")
    csv_lines = csv_fh.readlines()
    csv_fh.close()

    # Initialisation for conversion
    csv_dict = {}
    headers = csv_lines[0].replace("\n", "").split(delimeter)
    csv_lines = csv_lines[1:]
    for header in headers:
        csv_dict[header] = []

    # Start conversion to dict
    for csv_line in csv_lines:
        csv_line_list = csv_line.replace("\n", "").split(delimeter)
        for i in range(len(headers)):
            value = csv_line_list[i]
            if value == "":
                continue
            value = try_float_cast(value)
            csv_dict[headers[i]].append(value)
    
    # Convert single item lists to items and things multi-item lists
    for header in headers:
        if len(csv_dict[header]) == 1:
            csv_dict[header] = csv_dict[header][0]
    
    # Return
    return csv_dict

# Returns simerimental and predicted data from a CSV based on prefixes
def extract_data(data_dict:dict, sim_prefix:str, prd_prefix:str) -> tuple:
    sim_list, prd_list = [], []
    for field in data_dict.keys():
        if field.startswith(sim_prefix):
            sim_list += data_dict[field]
        elif field.startswith(prd_prefix):
            prd_list += data_dict[field]
    return sim_list, prd_list

# Prepare and plot
fig, ax = plt.subplots()
ax.set_aspect("equal", "box")

# Get simerimental and predicted data
data_dict = csv_to_dict(FILE_PATH)
# data_info = r"$\epsilon$ (mm/mm)"
# limits = (0, 0.25)
# sim_list, prd_list = extract_data(data_dict, "valid_y", "prd_y")
# colour = "green"
data_info = r"$t_f$ (h)"
limits = (0, 12000)
sim_list, prd_list = extract_data(data_dict, "valid_x", "prd_x")
sim_list = [t/3600 for t in sim_list]
prd_list = [t/3600 for t in prd_list]
colour = "red"
ax.ticklabel_format(axis="x", style="sci", scilimits=(3,3))
ax.ticklabel_format(axis="y", style="sci", scilimits=(3,3))
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True

# Set labels and plot line
plt.xlabel(f"Surrogate model {data_info}", fontsize=LABEL_FONTSIZE)
plt.ylabel(f"CPFE model {data_info}", fontsize=LABEL_FONTSIZE)
plt.plot(limits, limits, linestyle="--", color="black", zorder=1)

# Plot data
plt.scatter(prd_list, sim_list, zorder=2, color=colour, linewidth=LINEWIDTH, s=MARKER_SIZE**2)

# Add 'conservative' region
triangle_vertices = np.array([[limits[0], limits[0]], [limits[1], limits[0]], [limits[1], limits[1]]])
ax.fill(triangle_vertices[:, 0], triangle_vertices[:, 1], color="gray", alpha=0.3)
plt.text(limits[1]-0.48*(limits[1]-limits[0]), limits[0]+0.05*(limits[1]-limits[0]), "Non-conservative", fontsize=OTHER_FONTSIZE, color="black")

# Format figure size
plt.gca().set_position([0.17, 0.12, 0.75, 0.75])
plt.gca().grid(which="major", axis="both", color="SlateGray", linewidth=1, linestyle=":")
plt.gcf().set_size_inches(5, 5)

# Format limits and ticks
plt.xlim(limits)
plt.ylim(limits)
plt.xticks(fontsize=OTHER_FONTSIZE)
plt.yticks(fontsize=OTHER_FONTSIZE)

# Save
plt.savefig("plot.png")
