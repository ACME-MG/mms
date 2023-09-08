# Libraries
import numpy as np
import matplotlib.pyplot as plt

def csv_to_dict(csv_path:str, delimeter:str=",") -> dict:
    """
    Converts a CSV file into a dictionary
    
    Parameters:
    * `csv_path`:  The path to the CSV file
    * `delimeter`: The separating character
    
    Returns the dictionary
    """

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
            try:
                value = float(value)
            except:
                pass
            csv_dict[headers[i]].append(value)
    
    # Convert single item lists to items and things multi-item lists
    for header in headers:
        if len(csv_dict[header]) == 1:
            csv_dict[header] = csv_dict[header][0]
        else:
            csv_dict[header] = csv_dict[header]
    
    # Return
    return csv_dict

def get_x_list(num_stress:int, x_end:float):
    return list(np.linspace(0, x_end, num_stress+1))
    # start_num_strains = 5
    # end_num_strains = num_stress - start_num_strains
    # x_list  = [x_end/10/start_num_strains*(i+1) for i in range(start_num_strains)]
    # x_list += [x_end/end_num_strains*(i+1) for i in range(start_num_strains, num_stress)]
    # return [0] + x_list

# Read results
result_dict = csv_to_dict("prd_data.csv")
result_headers = list(result_dict.keys())
num_data = len(result_dict[result_headers[0]])

# Get input, valid, and predicted parameters
input_params = ["evp_s0", "evp_R", "evp_d", "evp_n", "evp_eta"]
valid_params = [header for header in result_headers if header.startswith("valid_")]
num_y_points = len(valid_params) - 1

# Iterate through results and start plotting
for i in range(num_data):

    # Get validation data
    valid_x_list = get_x_list(num_y_points, result_dict["valid_x_end"][i])
    valid_y_list = [0] + [result_dict[f"valid_y_{j+1}"][i] for j in range(num_y_points)]

    # Get SM prediction data
    prd_x_list = get_x_list(num_y_points, result_dict["prd_x_end"][i])
    prd_y_list = [0] + [result_dict[f"prd_y_{j+1}"][i] for j in range(num_y_points)]
    
    # Plot everything
    plt.scatter(valid_x_list, valid_y_list, c="grey", label="Validation (NEML)")
    plt.plot(prd_x_list, prd_y_list, c="red", label="Prediction (SM)")
    plt.legend()
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.savefig(f"plot_{i+1}")
    plt.clf()
