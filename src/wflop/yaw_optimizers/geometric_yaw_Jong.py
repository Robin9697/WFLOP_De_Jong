import numpy                                                       as     np
from   floris.tools                                                import FlorisInterface

def get_yaw(dx: np.ndarray, dy: np.ndarray, max_yaw: float = 25.)->np.ndarray:
    
    x = np.minimum(np.maximum(dx/25., 0), 1)
    y = np.minimum(np.maximum(np.absolute(dy), 0), 1)

    yaw = 30.*(1 - np.power(x,1.336))*(1 - np.power(y,1.407))
    
    yaw = np.minimum(yaw, max_yaw)
    yaw = np.where(dy<0, -yaw, yaw)
    yaw = np.where(dx<=0, 0., yaw )
    
    return yaw

def get_distances(rotated_coordinates: np.ndarray)->np.ndarray:
    # rotated_coordinates are the x and y coordinates for every wind direction in rotor diameters
    # [[x_1(wd1), x_2(wd1), ..., x_n(wd1)],
    #  [y_1(wd1), y_2(wd1), ..., y_n(wd1)],
    #  [x_1(wd2), x_2(wd2), ..., x_n(wd2)],
    #  [y_1(wd2), y_2(wd2), ..., y_n(wd2)],
    #   ...
    #  [x_1(wdn), x_2(wdn), ..., x_n(wdn)],
    #  [y_1(wdn), y_2(wdn), ..., y_n(wdn)]] 
    
    # Generate num_wind_bins x n_turbines numpy arrays for dx and dy
    n_turbines  = len(rotated_coordinates[0])
    n_wind_bins = int(len(rotated_coordinates[:,0])/2)
    dx = -1*np.ones((n_wind_bins, n_turbines))
    dy = -1*np.ones((n_wind_bins, n_turbines))
    
    for j in range(n_turbines):
        # xy are the rotated coordinates adjusted where the coordinates of turbine j is the orgin
        xy = np.subtract(rotated_coordinates.T, rotated_coordinates[:,j]).T
        
        # Get x and y coordinates ordered from small to large x, size n_wind_bins x n_turbines
        ind = np.argsort(xy[::2], axis=1)
        x = np.take_along_axis(xy[::2], ind, axis=1)
        y = np.take_along_axis(xy[1::2], ind, axis=1)
        
        # Get indexes of coordinates where |y|<=1 and x>0
        [ind_wd, ind_turb] = np.where((abs(y)<=1) & (x>0))
        # Get index of corresponding wind direction
        for i in range(n_wind_bins):
            ind_t = ind_turb[np.where(ind_wd==i)]
            # If there is a turbine in the wake of turbine j, save the dx and dy of the closest one to the numpy array.
            if len(ind_t)>0:
                dx[i][j] = x[i][ind_t[0]]
                dy[i][j] = y[i][ind_t[0]]  
                
    return dx, dy

def geometric_yaw_Jong(farm: FlorisInterface, rotation_matrix: np.ndarray = None, max_yaw: float = 25.)->np.ndarray:
    
    # The rotaion matrix turns the layout such that wind comes from the west, for every wind direction
    # Rotation matrix can be made at initialization
    if type(rotation_matrix) is not np.ndarray:
        wind_directions = farm.floris.flow_field.wind_directions
        rotation_matrix = np.zeros((len(wind_directions)*2, 2))
        for i in range(len(wind_directions)):      
            theta = -(wind_directions[i] - 270.)*(np.pi/180.) # 270 degrees is wind from the west
            rotation_matrix[2*i:2*(i+1),:] = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    
    # Get dx and dy values from layout        
    layout = np.array([farm.floris.farm.layout_x,farm.floris.farm.layout_y])/farm.floris.farm.rotor_diameters[0]
    rotated_coordinates = np.matmul(rotation_matrix, layout) # n_wind_directions*2 x n_turbines
    # rotated_coordinates are the x and y coordinates for every wind direction in rotor diameters
    # [[x_1(wd1), x_2(wd1), ..., x_n(wd1)],
    #  [y_1(wd1), y_2(wd1), ..., y_n(wd1)],
    #  [x_1(wd2), x_2(wd2), ..., x_n(wd2)],
    #  [y_1(wd2), y_2(wd2), ..., y_n(wd2)],
    #   ...
    #  [x_1(wdn), x_2(wdn), ..., x_n(wdn)],
    #  [y_1(wdn), y_2(wdn), ..., y_n(wdn)]] 
    dx, dy = get_distances(rotated_coordinates) # n_wind_directions x n_turbines
    
    # Get yaw angles depending on the dx and dy of the layout.
    yaw_angles = get_yaw(dx, dy, max_yaw) # n_wind_directions x n_turbines
    # The yaw angles are independed of the wind speed, so the found yaw angles can simply be repeated.
    yaw_angles = np.tile(yaw_angles, (1,farm.floris.flow_field.n_wind_speeds))
    # yaw_angles should be of size wind_direction x wind_speed x turbine to be Floris complient.
    yaw_angles = yaw_angles.reshape((farm.floris.flow_field.n_wind_directions,\
                                     farm.floris.flow_field.n_wind_speeds,\
                                     len(farm.floris.farm.layout_x)))
      
    return yaw_angles