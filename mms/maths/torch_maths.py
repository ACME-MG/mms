"""
 Title:         Torch Maths
 Description:   A collection of mathematical functions that use pytorch's
                operations to retain gradient information
 References:    [1] https://www.researchgate.net/publication/324088567_Computing_Euler_angles_with_Bunge_convention_from_rotation_matrix

 Author:        Janzen Choi

"""

# Libraries
import torch
from mms.maths.csl import get_symmetry_matrices

def get_torch_misorientation(euler_1_tensor:torch.tensor, euler_2_tensor:torch.tensor, crystal_type:str) -> torch.tensor:
    """
    Calculates the misorientation using torch operations
    
    Parameters:
    * `euler_1_tensor`: A tensor of the first euler-bunge angles
    * `euler_2_tensor`: A tensor of the second euler-bunge angles
    * `crystal_type`:   The type of crystal structure (e.g., cubic)
    
    Returns a list of misorientations
    """
    misorientations_1 = get_torch_misorientations(euler_1_tensor, euler_2_tensor, crystal_type)
    misorientations_2 = get_torch_misorientations(euler_2_tensor, euler_1_tensor, crystal_type)
    misorientations = misorientations_1 + misorientations_2
    return torch.min(misorientations)/2 # I don't know why we divide but it works
    
def get_torch_misorientations(euler_1_tensor:torch.tensor, euler_2_tensor:torch.tensor, crystal_type:str) -> torch.tensor:
    """
    Calculates the misorientation using torch operations
    
    Parameters:
    * `euler_1_tensor`: A tensor of the first euler-bunge angles
    * `euler_2_tensor`: A tensor of the second euler-bunge angles
    * `crystal_type`:   The type of crystal structure (e.g., cubic)
    
    Returns a list of misorientations
    """
    
    # Calculate rotation and symmetry matrices
    rotation_1 = euler_to_torch_matrix(euler_1_tensor)
    rotation_2 = euler_to_torch_matrix(euler_2_tensor)
    symmetries = get_symmetry_matrices(crystal_type)

    # Calculate misorientations
    misorientation_list = []
    for symmetry_1 in symmetries:
        symmetry_1 = torch.tensor(symmetry_1, dtype=torch.float32)
        operator_1 = torch.matmul(symmetry_1.float(), rotation_1.float())
        for symmetry_2 in symmetries:
            symmetry_2 = torch.tensor(symmetry_2, dtype=torch.float32)
            operator_2 = torch.matmul(symmetry_2.float(), rotation_2.float())
            delta = torch.matmul(operator_2, operator_1.T)
            cw = 0.5 * (torch.trace(delta) - 1)
            cw = torch.clamp(cw, -1.0, 1.0)
            misorientation = torch.acos(cw)
            misorientation_list.append(misorientation)
    
    # Return the misorientations
    return torch.tensor(misorientation_list, dtype=torch.float32)

def euler_to_torch_matrix(euler_tensor:torch.tensor) -> torch.tensor:
    """
    Determines the orientation matrices of a set of euler-bunge angles;
    from Ref. [1]
    
    Parameters:
    * `euler`: The euler angle in euler-bunge form (rads)

    Returns 3x3 orientation matrices
    """
    om_11 = torch.cos(euler_tensor[0])*torch.cos(euler_tensor[2]) - torch.sin(euler_tensor[0])*torch.sin(euler_tensor[2])*torch.cos(euler_tensor[1])
    om_12 = torch.sin(euler_tensor[0])*torch.cos(euler_tensor[2]) + torch.cos(euler_tensor[0])*torch.sin(euler_tensor[2])*torch.cos(euler_tensor[1])
    om_13 = torch.sin(euler_tensor[2])*torch.sin(euler_tensor[1])
    om_21 = -torch.cos(euler_tensor[0])*torch.sin(euler_tensor[2]) - torch.sin(euler_tensor[0])*torch.cos(euler_tensor[2])*torch.cos(euler_tensor[1])
    om_22 = -torch.sin(euler_tensor[0])*torch.sin(euler_tensor[2]) + torch.cos(euler_tensor[0])*torch.cos(euler_tensor[2])*torch.cos(euler_tensor[1])
    om_23 = torch.cos(euler_tensor[2])*torch.sin(euler_tensor[1])
    om_31 = torch.sin(euler_tensor[0])*torch.sin(euler_tensor[1])
    om_32 = -torch.cos(euler_tensor[0])*torch.sin(euler_tensor[1])
    om_33 = torch.cos(euler_tensor[1])
    om_1 = torch.stack((om_11, om_12, om_13))
    om_2 = torch.stack((om_21, om_22, om_23))
    om_3 = torch.stack((om_31, om_32, om_33))
    om = torch.stack((om_1, om_2, om_3))
    return om


