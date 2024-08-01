"""
 Title:         General helper
 Description:   General helper functions
 Author:        Janzen Choi

"""

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
    * `value`: The value to be rounded
    * `sf`:    The number of significant figures

    Returns the rounded number
    """
    format_str = "{:." + str(sf) + "g}"
    rounded_value = float(format_str.format(value))
    return rounded_value
