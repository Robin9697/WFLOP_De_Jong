import numpy                                                       as     np
from   floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from   floris.tools                                                import FlorisInterface

def yaw_optimizer_floris(farm: FlorisInterface, max_yaw: float = 25.):
    
    df      = YawOptimizationSR(fi = farm, minimum_yaw_angle=-max_yaw, maximum_yaw_angle=max_yaw)
    df_opt  = df.optimize(print_progress=False)
    
    yaw_angles = np.vstack(df_opt["yaw_angles_opt"])
    yaw_angles = yaw_angles.reshape((farm.floris.flow_field.n_wind_directions,\
                                     farm.floris.flow_field.n_wind_speeds,\
                                     len(farm.floris.farm.layout_x)),\
                                     order='F')
      
    return yaw_angles