"""
 Title:         NEML
 Description:   Helper functions that use NEML
 Author:        Janzen Choi

"""

# Libraries
from neml.math import rotations
from neml.cp import crystallography

def get_cubic_misorientation(euler_1:list, euler_2:list):
    """
    Determines the misorientation of two sets of euler angles (rads);
    assumes cubic structure

    Parameters:
    * `euler_1`: The first euler angle
    * `euler_2`: The second euler angle
    
    Returns the misorientation angle
    """
    euler_1 = rotations.CrystalOrientation(*euler_1, angle_type="radians", convention="bunge")
    euler_2 = rotations.CrystalOrientation(*euler_2, angle_type="radians", convention="bunge")
    sym_group = crystallography.SymmetryGroup("432")
    mori = sym_group.misorientation(euler_1, euler_2)
    _, mori_angle = mori.to_axis_angle()
    return mori_angle

def euler_to_quat(euler:list) -> list:
    """
    Converts euler-bunge to quaternion
    
    Parameters:
    * `euler`: The euler-bunge angle (rads) 
    
    Returns the quaternion as a list
    """
    orientation = rotations.Orientation(*euler, angle_type="radians", convention="bunge")
    return list(orientation.quat)
