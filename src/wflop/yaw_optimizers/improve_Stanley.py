# Standard library imports
import  os
from    sys                                                        import platform
from    datetime                                                   import datetime
import  math

# Third party imports
import pandas                                                      as     pd
import numpy                                                       as     np
import matplotlib.pyplot                                           as     plt
from   scipy.optimize                                              import curve_fit
from   floris.tools                                                import FlorisInterface
from   floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

# Locat imports
from   wflop.yaw_optimizers.geometric_yaw_Stanley                  import get_yaw_angles_Stanley, process_layout, place_turbines

################################################
# Get folders where results are saved
################################################

current_file     = os.path.abspath(os.getcwd())
results_folder = os.path.join(current_file, "results")
data_folder = "{}/{}".format(results_folder, "GY")
if not os.path.exists(data_folder):
    os.makedirs(data_folder)  
figures_folder = "{}/{}".format(results_folder, "Figures")
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)  
    
################################################
# Import and initialize FLORIS wind farm
################################################

data_filename    = os.path.join(current_file, "wflop\inputs\wind_farm.json")
farm             = FlorisInterface(data_filename)
farm.reinitialize(wind_speeds=[8.0], wind_directions=[270.0])
farm.floris.wake.model_strings['velocity_model'] = "gauss"

#############################################
# Functions
#############################################
        
def yaw_multiple_setups(farm: FlorisInterface, filename: str, turbs_array: list, spacing_array: list, nruns: int, minimum_spacing: float):
    # Open csv file and write header
    results_file = open(filename, 'w') 
    results_file.write('yaw,dx,dy,\n')
    
    # Generate layout with given parameters
    D = farm.floris.farm.rotor_diameters[0]
    for i in range(len(turbs_array)):
        nturbs = turbs_array[i]
        for j in range(len(spacing_array)):
            spacing = spacing_array[j]
            s = spacing*(np.sqrt(nturbs)-1)*D
            for k in range(nruns):
                layout_x, layout_y = place_turbines(nturbs,s,minimum_spacing*D)

                # Optimize yaw angles of the generated layout
                farm.reinitialize(layout_x=layout_x, layout_y=layout_y)
                df          = YawOptimizationSR(fi = farm, minimum_yaw_angle=-45.0, maximum_yaw_angle=45.0)
                df_opt      = df.optimize(print_progress=False)
                yaw_angles  = df_opt["yaw_angles_opt"][0]
                
                # Get (dx,dy) values of the generated layout
                dx, dy      = process_layout(layout_x, layout_y, D)
                
                # Save yaw angle, dx, dy and layout to csv file
                for k in range(nturbs):
                    results = [str(yaw_angles[k]), str(dx[k]), str(dy[k]), str(layout_x), str(layout_y), '\n']
                    print(results)
                    results_file.write(','.join(results))      
    
    # Close csv file                      
    results_file.close()

def plot_data(filename: str, plotname: str):
    
    # Load yaw angles, dx and dy from csv file
    df = pd.read_csv(filename, 
                     header = 0, 
                     index_col=False,
                     usecols=['yaw','dx','dy'])
    df = df[df['dx']<=50]
    df = df[df['dx']>=1]

    # Generate scatter plot
    fig, ax = plt.subplots(figsize=(10, 4))
    cm = plt.cm.get_cmap('Spectral')
    sc = plt.scatter(df["dx"],
                    df["dy"],
                    s=0.1, c=abs(df["yaw"]), cmap=cm,
                    vmin=0.0, vmax=np.max(df["yaw"]))
    
    # Set x and y lables
    plt.xlabel('dx (D)')
    plt.ylabel('dy (D)')
    
    # Show colorbar on right side of the scatter plot
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.set_title('Yaw angle (degree)', fontsize = 10)
    
    # Save figure
    plt.savefig(plotname)

def func0(xy, max_x, max_y, max_yaw, power_x, power_y)->float:
    
    dx, dy = xy
    
    x = dx/max_x
    y = np.absolute(dy)/max_y
    
    x = np.minimum(np.maximum(x, 0), 1)
    y = np.minimum(np.maximum(y, 0), 1)

    yaw = max_yaw*(1 - np.power(x,power_x))*(1 - np.power(y,power_y))
    
    return yaw

def func1(xy, max_yaw, power_x, power_y)->float:
    
    max_x = 25
    max_y = 1
    
    dx, dy = xy
    
    x = dx/max_x
    y = np.absolute(dy)/max_y
    
    x = np.minimum(np.maximum(x, 0), 1)
    y = np.minimum(np.maximum(y, 0), 1)

    yaw = max_yaw*(1 - np.power(x,power_x))*(1 - np.power(y,power_y))
    
    return yaw

def func2(xy, power_x, power_y)->float:
    
    max_x = 25
    max_y = 1
    max_yaw = 30
    
    dx, dy = xy
    
    x = dx/max_x
    y = np.absolute(dy)/max_y
    
    x = np.minimum(np.maximum(x, 0), 1)
    y = np.minimum(np.maximum(y, 0), 1)

    yaw = max_yaw*(1 - np.power(x,power_x))*(1 - np.power(y,power_y))
    
    return yaw

def data_fitting(run):
    # Load and preprocess the data
    filename = "{}/results_var_turbines_floris.csv".format(data_folder)
    df = pd.read_csv(filename, header = 0, index_col=False)
    # After a certain value of dx, the yaw is always set to zero.
    dx_max = math.ceil(np.max(df[df['yaw']>0]['dx']))
    df = df[df['dx']<dx_max]
    # After a certain value of dx, the yaw is always set to zero.
    dy_max = math.ceil(np.max(df[df['yaw']>0]['dy']))
    df = df[abs(df['dy'])<dy_max]
    # dx = -1 means there are no turbines in the wake, so the yaw is set to zero.
    df = df[df['dx']!=-1]
    # For the datafitting the aboslute value of the yaw is taken.
    df['yaw'] = np.absolute(df['yaw'])
    
    # First run without any value fixed, then fix max_x, max_y and max_yaw and run again.
    if run == 0:
        print("After a dx of {}D no yaw is applied anymore.".format(dx_max))
        print("After a dy of {}D no yaw is applied anymore.".format(dy_max))
        func = func0
        p0 = [25., 1., 30., 1., 1.]
        name = ['max_x', 'max_y', 'max_yaw', 'power_x', 'power_y']
    elif run == 1:
        print("Max_x and max_y are fixed.")
        func = func1
        p0 = [30., 1., 1.]
        name = ['max_yaw', 'power_x', 'power_y']
    elif run == 2:
        print("Max_x, max_y and max_yaw are fixed.")
        func = func2
        p0 = [1., 1.]
        name = ['power_x', 'power_y']
    
    # Fit parameters to data
    popt, pcov = curve_fit(func, (df['dx'], df['dy']), df['yaw'],
                           p0 = p0,
                           bounds = (0., np.inf))
    for k in range(len(popt)):
        print(name[k] + ' : ' + str(p0[k]) + ' -- > ' + str(popt[k]))
    
    # Get results fitted function
    x_range = np.linspace(np.min(df["dx"]), dx_max, 500)
    y_range = np.linspace(-dy_max, dy_max, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func((X, Y), *popt)
    
    # Plot data vs fitted function
    fig, ax_list = plt.subplots(2, 1, figsize=(10, 4))
    ax = ax_list.flatten()
    cm = plt.cm.get_cmap('Spectral')
    ax[0].scatter(df["dx"], df["dy"],
                s=2, c=abs(df["yaw"]), cmap=cm,
                vmin=0.0, vmax=25)
    ax[0].set_title('data') 
    ax[1].scatter(X.flatten(), Y.flatten(),
                s=2, c=abs(Z.flatten()), cmap=cm,
                vmin=0.0, vmax=25)
    ax[1].set_title('fit') 
    plt.show()

def geometric_yaw(version: str, filename: str, dx_range: np.ndarray, dy_range: np.ndarray):
    # Open csv file and write header
    results_file = open(filename, 'w') 
    results_file.write('yaw,dx,dy,\n')
    
    # Go through dx and dy values
    for dx in dx_range:
        for dy in dy_range:
            
            # Check if second turbine is in the wake of the first
            if abs(dy) <= 0.1*dx + 0.5:
                # Check if the minimum distance between the two trubines is at least 4D.
                if np.sqrt(dx**2 + dy**2) >= 4:
                    
                    # Get yaw angle
                    if version == "Stanley":
                        yaw_angle   = get_yaw_angles_Stanley(dx, dy)
                    elif version == "Jong":
                        yaw_angle   = get_yaw_Jong(dx, dy)
                        
                    # Save yaw angle, dx, dy and layout to csv file
                    results = [str(yaw_angle), str(dx), str(dy), '\n']
                    results_file.write(','.join(results))    
                    
    # Close csv file         
    results_file.close()

def get_yaw_Jong(dx: np.ndarray, dy: np.ndarray, max_yaw: float = 25.)->np.ndarray:
    
    max_x = 25.
    max_y = 1.
    max_yaw = 30.
    power_x = 1.336
    power_y = 1.407
    
    x = dx/max_x
    y = np.absolute(dy)/max_y
    
    x = np.minimum(np.maximum(x, 0), 1)
    y = np.minimum(np.maximum(y, 0), 1)

    yaw = max_yaw*(1 - np.power(x,power_x))*(1 - np.power(y,power_y))
    
    yaw = np.where(dy<0, -yaw, yaw)
    yaw = np.where(dx<=0, 0., yaw )
    
    return yaw

def compare_geometric(file_name_string: list, titles: list, plotname: str):
    
    # Generate figure
    fig, ax_list = plt.subplots(3, 1, figsize=(10, 10))
    ax_list = ax_list.flatten()
    
    for i in range(len(ax_list)):
        ax = ax_list[i]
         
        # Load yaw angles, dx and dy from csv file
        filename = "{}/results_{}.csv".format(data_folder, file_name_string[i]) 
        df = pd.read_csv(filename, 
                        header = 0, 
                        index_col=False,
                        usecols=['yaw','dx','dy'])

        # Generate scatter plot
        cm = plt.cm.get_cmap('Spectral')
        sc = ax.scatter(df["dx"],
                    df["dy"],
                    s=1, c=abs(df["yaw"]), cmap=cm,
                    vmin=0.0, vmax=25)
        
        # Set limits, title and lables
        ax.set_xlim([4,30])
        ax.set_ylim([-4,4])
        ax.set_title(titles[i])
        if i == 1:
            ax.set_ylabel('dy (D)')
        elif i == 2:
            ax.set_xlabel('dx (D)')
    
    # Show colorbar on right side of the scatter plot
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.13, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.set_title('Yaw angle (degree)', fontsize = 12, pad=10)
    
    # Save figure
    plt.savefig(plotname)

#############################################
# Run
############################################

# run:
# 1: Create data var turbines (takes a few hours)
# 2: Plot data var turbines
# 3: Data fitting
# 4: Create data 2 turbines
# 5: Plot comparison data, Stanley and Jong

run = 5

if run == 1:
    turbs_array = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    spacing_array = [7., 8., 9., 10., 15., 20., 30.]
    nruns = 10
    filename = "{}/results_var_turbines_floris.csv".format(data_folder)
    start = datetime.now()
    yaw_multiple_setups(farm            = farm,
                        filename        = filename, 
                        turbs_array     = turbs_array,
                        spacing_array   = spacing_array,
                        nruns           = nruns,
                        minimum_spacing = 4)
    end = (datetime.now()-start).total_seconds()
    print("Time (min): " + str(end/60.)) 
if run == 2:   
    filename = "{}/results_var_turbines_floris.csv".format(data_folder)
    plotname = "{}/plot_var_turbines_floris.png".format(figures_folder)  
    plot_data(          filename        = filename,
                        plotname        = plotname)
if run == 3:
    data_fitting(0)
    data_fitting(1)
    data_fitting(2)
if run == 4:
    dx_range        = np.arange(0, 40, 0.01)
    dy_range        = np.arange(-4, 4, 0.01)
    filename = "{}/results_2_turbines_geometric_Stanley.csv".format(data_folder)
    geometric_yaw(  version = "Stanley",
                    filename        = filename,
                    dx_range        = dx_range,
                    dy_range        = dy_range)
    filename = "{}/results_2_turbines_geometric_Jong.csv".format(data_folder)
    geometric_yaw(  version = "Jong",
                    filename        = filename,
                    dx_range        = dx_range,
                    dy_range        = dy_range)
if run == 5:
    plotname = "{}/plot_compare_geometric.png".format(figures_folder)  
    file_name_string = ['var_turbines_floris', '2_turbines_geometric_Stanley','2_turbines_geometric_Jong']
    titles = ['FLORIS optimized yaw angles', 'Geometric yaw Stanley', 'Geometric yaw De Jong']
    compare_geometric(file_name_string, titles, plotname)