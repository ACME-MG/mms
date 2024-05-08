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
