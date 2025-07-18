"""
Selection from Stanley: 
https://github.com/pjstanle/GeometricYaw/tree/paper/initial_submission
"""

import numpy                                                     as np
from floris.utilities                                            import rotate_coordinates_rel_west
from floris.tools                                                import FlorisInterface

# From generate_yaw_data.py
def place_turbines(nturbs,side,min_spacing):
    iterate = True
    while iterate:
        turbine_x = np.array([])
        turbine_y = np.array([])
        for i in range(nturbs):
            placed = False
            counter = 0
            while placed == False:
                counter += 1
                temp_x = np.random.rand()*side
                temp_y = np.random.rand()*side
                good_point = True
                for j in range(len(turbine_x)):
                    dist = np.sqrt((temp_y - turbine_y[j])**2 + (temp_x - turbine_x[j])**2)
                    if dist < min_spacing:
                        good_point = False
                if good_point == True:
                    turbine_x = np.append(turbine_x,temp_x)
                    turbine_y = np.append(turbine_y,temp_y)
                    placed = True
                # while loop
                if counter == 1000:
                    break

            # for loop
            if counter == 1000:
                    break

        if counter != 1000:
            return turbine_x, turbine_y

# From geometric_yaw.py        
def process_layout(turbine_x: np.ndarray, turbine_y: np.ndarray, rotor_diameter: float, spread: float = 0.1)->tuple:
    """
    returns the distance from each turbine to the nearest downstream waked turbine
    normalized by the rotor diameter. Right now "waked" is determind by a Jensen-like
    wake spread, but this could/should be modified to be the same as the trapezoid rule
    used to determine the yaw angles.

    turbine_x: turbine x coords (rotated)
    turbine_y: turbine y coords (rotated)
    rotor_diameter: turbine rotor diameter (float)
    spread=0.1: Jensen alpha wake spread value
    """
    nturbs = len(turbine_x)
    dx = np.zeros(nturbs) + 1E10
    dy = np.zeros(nturbs)
    # Turbine A
    for waking_index in range(nturbs):
        # Turbine B
        for waked_index in range(nturbs):
            # Check if turbine B is behind turbine A (x wise)
            if turbine_x[waked_index] > turbine_x[waking_index]:
                # Compute radius of wake of turbine A at the x cordinate of turbine B
                r = spread*(turbine_x[waked_index]-turbine_x[waking_index]) + rotor_diameter/2.0
                # Check if turbine B is in wake turbine A (y wise)
                if abs(turbine_y[waked_index]-turbine_y[waking_index]) < r:
                    # Check if turbine B is closer to turbine A than other trubines up untill now (x wise)
                    if (turbine_x[waked_index] - turbine_x[waking_index]) < dx[waking_index]:
                        # Save distance of turbine A to turbine B to the dx and dy of turbine A
                        dx[waking_index] = turbine_x[waked_index] - turbine_x[waking_index]
                        dy[waking_index] = turbine_y[waked_index] - turbine_y[waking_index]
        
        # If no turbines are found in de wake of turbine A, then dx and  dy is set to zero
        if dx[waking_index] == 1E10:
            dx[waking_index] = -rotor_diameter
            dy[waking_index] = -rotor_diameter

    return dx/rotor_diameter, dy/rotor_diameter

# From geometric_yaw.py                
def get_yaw_angles_Stanley(x: float, y: float, max_x = 30.0, max_y = 1.0, max_yaw = 25.0)->float:
    left_x=0.0
    top_left_y=1.0
    right_x=25.0
    top_right_y=1.0
    top_left_yaw=30.0
    top_right_yaw=0.0
    bottom_left_yaw=30.0
    bottom_right_yaw=0.0
    # I realize this is kind of a mess and needs to be clarified/cleaned up. As it is now:
    
    # x and y: dx and dy to the nearest downstream turbine in rotor diameteters with turbines rotated so wind is coming left to right
    # left_x: where we start the trapezoid...now that I think about it this should just be assumed as 0
    # top_left_y: trapezoid top left coord
    # right_x: where to stop the trapezoid. Basically, to max coord after which the upstream turbine won't yaw
    # top_right_y: trapezoid top right coord
    # top_left_yaw: yaw angle associated with top left point
    # top_right_yaw: yaw angle associated with top right point
    # bottom_left_yaw: yaw angle associated with bottom left point
    # bottom_right_yaw: yaw angle associated with bottom right point
    if x <= left_x or x>right_x:
        return 0.0
    else:
        dx = (x-left_x)/(right_x-left_x)
        edge_y = top_left_y + (top_right_y-top_left_y)*dx
        if abs(y) > edge_y:
            return 0.0
        else:
            top_yaw = top_left_yaw + (top_right_yaw-top_left_yaw)*dx
            bottom_yaw = bottom_left_yaw + (bottom_right_yaw-bottom_left_yaw)*dx
            yaw = bottom_yaw + (top_yaw-bottom_yaw)*abs(y)/edge_y
            if y < 0:
                return -yaw
            else:
                return yaw

# Adjusted from geometric_yaw.py 
def geometric_yaw_Stanley(farm: FlorisInterface,
                  max_x = 25.0, max_y = 1.0, max_yaw = 30.0)->np.ndarray:
   
    rotor_diameter = farm.floris.farm.rotor_diameters[0]
    layout_x = np.array(farm.floris.farm.layout_x)
    layout_y = np.array(farm.floris.farm.layout_y)
    n_turbines = len(farm.floris.farm.layout_x)
    wind_directions = farm.floris.flow_field.wind_directions
    
    yaw_angles = np.zeros((farm.floris.flow_field.n_wind_directions, farm.floris.flow_field.n_wind_speeds, n_turbines))
    
    for i in range(farm.floris.flow_field.n_wind_directions):
        turbine_coordinates_array = np.zeros((n_turbines,3))
        turbine_coordinates_array[:,0] = layout_x[:]
        turbine_coordinates_array[:,1] = layout_y[:]
        
        wind_direction = wind_directions[i]
        rotated_x, rotated_y, z_coord_rotated, x_center_of_rotation, y_center_of_rotation = rotate_coordinates_rel_west(np.array([wind_direction]), turbine_coordinates_array)
        processed_x, processed_y = process_layout(rotated_x[0][0],rotated_y[0][0],rotor_diameter)
        
        for k in range(n_turbines):
            yaws = get_yaw_angles_Stanley(processed_x[k], processed_y[k], max_x, max_y, max_yaw)
            for j in range(farm.floris.flow_field.n_wind_speeds):
                yaw_angles[i][j][k] = yaws
                
    return yaw_angles