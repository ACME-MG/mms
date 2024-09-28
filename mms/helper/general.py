"""
 Title:         General helper
 Description:   General helper functions
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
import math

def sort_dict(unsorted_dict:dict) -> dict:
    """
    Sorts a dictionary in ascending order based on its values
    
    Parameters:
    * `unsorted_dict`: Dictionary with unsorted values
    
    Returns the sorted dictionary
    """
    sorted_dict = dict(sorted(unsorted_dict.items(), key=lambda item: item[1]))
    return sorted_dict

def get_thinned_list(unthinned_list:list, density:int) -> list:
    """
    Gets a thinned list

    Parameters:
    * `unthinned_list`: The list before thinning
    * `density`:        The goal density of the thinned list

    Returns the thinned list
    """
    src_data_size = len(unthinned_list)
    step_size = src_data_size / density
    thin_indexes = [math.floor(step_size*i) for i in range(1, density - 1)]
    thin_indexes = [0] + thin_indexes + [src_data_size - 1]
    thinned_list = [unthinned_list[i] for i in thin_indexes]
    return thinned_list

def splice_list(unspliced_list:list, num_elements:int) -> list:
    """
    Splices a list of values into a list of lists of values
    
    Parameters:
    * `unspliced_list`: The single list of values
    * `num_elements`:   The number of elements to splice the list to
    
    Returns the spliced list
    """
    return [unspliced_list[i:i+num_elements] for i in range(0,len(unspliced_list),num_elements)]

def flatten(list_of_lists:list) -> list:
    """
    Flattens a 2D list into a 1D list
    
    Parameters:
    * `list_of_lists`: A list of lists (i.e., a 2D grid)
    
    Returns the flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]

def transpose(list_of_lists:list) -> list:
    """
    Transposes a 2D list of lists
    
    Parameters:
    * `list_of_lists`: A list of lists (i.e., a 2D grid)
    
    Returns the transposed list of lists
    """
    transposed = np.array(list_of_lists).T.tolist()
    return transposed

def integer_to_ordinal(n:int):
    """
    Converts an integer to an ordinal string
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix

def round_sf(value:float, sf:int) -> float:
    """
    Rounds a float to a number of significant figures

    Parameters:
    * `value`: The value to be rounded; accounts for lists
    * `sf`:    The number of significant figures

    Returns the rounded number
    """
    if isinstance(value, list):
        return [round_sf(v, sf) for v in value]
    format_str = "{:." + str(sf) + "g}"
    rounded_value = float(format_str.format(value))
    return rounded_value
