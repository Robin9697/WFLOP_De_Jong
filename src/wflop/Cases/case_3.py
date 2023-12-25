# Standard library imports
import json
import os
from   sys                                                    import platform

# Third party imports
import pandas                                                 as     pd
from   datetime                                               import datetime 

# Local application imports
from   wflop.genetic_algorithm.WFLOP_GA_class                 import Wflop_ga

################################################
# Input
################################################

folder_name_save = "Case_3/Iteration_GA_2"
objective_list = ['LCOE', 'LCOE', 'AEP', 'AEP']
type_list = ['Joint', 'Sequential', 'Joint', 'Sequential' ]

num_run_file = open('run_num_case_3.txt', 'r')
num_run_org = num_run_file.read()
num_run_file.close()

objective = objective_list[int(num_run_org) % 4]
opt_type = type_list[int(num_run_org) % 4]
num_run = int(num_run_org) // 4

print("num_run_org = {}".format(num_run_org))
print("num_run = {}".format(num_run))
print("file = results_{}_{}_{}".format(num_run, objective, opt_type))
print("folder_name = {}".format(folder_name_save))

################################################
# Importing data from files
################################################

class PlatformError(Exception):
    pass

owd = os.getcwd()

if platform == "linux" or platform == "linux2":   

    os.chdir("wflop/inputs")
    
    parameters       = json.load(open("parameters_case_2.json"))
    
    floris_wind_farm = json.load(open("wind_farm.json"))
    
    # wind_rose.csv should have the following colums: wind speed, wind direction, frequency
    wind_rose        = pd.read_csv("wind_rose.csv", \
                            header      = 0 , \
                            index_col   = False)
    wind_rose        = wind_rose.to_numpy()  
    
    # domain_constraints.parquet should have the following colums: x, y, z
    domain           = pd.read_parquet("domain.parquet")
    domain           = domain.to_numpy()
    
    os.chdir(owd)

elif platform == "win32" or platform == "darwin":

    current_file     = os.path.abspath(os.getcwd())

    data_filename    = os.path.join(current_file, "wflop\inputs\parameters_case_2.json")
    parameters       = json.load(open(data_filename))
    
    data_filename    = os.path.join(current_file, "wflop\inputs\wind_farm.json")
    floris_wind_farm = json.load(open(data_filename))
    
    # wind_rose.csv should have the following colums: wind speed, wind direction, frequency
    data_filename    = os.path.join(current_file, "wflop\inputs\wind_rose.csv")
    wind_rose        = pd.read_csv(data_filename, \
                            header      = 0 , \
                            index_col   = False)
    wind_rose        = wind_rose.to_numpy()  
    
    # domain_constraints.parquet should have the following colums: x, y, z
    data_filename    = os.path.join(current_file, "wflop\inputs\domain.parquet")
    domain           = pd.read_parquet(data_filename)
    domain           = domain.to_numpy()
       
################################################
# Run
################################################

t = datetime.now()
print("Start time: {}".format(t))

# Generate wfga object
wfga = Wflop_ga(parameters  =   parameters,
        wind_rose           =   wind_rose,
        domain              =   domain,
        floris_wind_farm    =   floris_wind_farm,
        substation          =   [532359.5, 5851358.4],
            print_progress      =   True)

# Go to folder of the results of case 2
results_sub_folder= "results/GA/{}".format('Case_2')
if platform == "linux" or platform == "linux2": 
    os.chdir(results_sub_folder)

# Create filename    
if platform == "linux" or platform == "linux2": 
    filename_results = "results_{}_{}_{}.csv".format(num_run, objective, opt_type)
elif platform == "win32" or platform == "darwin":    
    filename_results = "{}/results_{}_{}_{}.csv".format(results_sub_folder, num_run, objective, opt_type)

     
if os.path.exists(filename_results):     
    # Get layout and original objective  
    df = pd.read_csv(filename_results, header=None, names=['objective ', 'layout', 'time'])
    obj_org = df['objective '].iloc[-1]
    layout = df['layout'].iloc[-1]   
    layout = layout.replace('  ', ' ').replace('  ', ' ').replace('[ ', '').replace(']', '')
    layout = [int(i) for i in layout.split(' ')]

    # Compute robust objective 
    aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                        optimizer = "geometric_yaw_Jong", 
                                        complete_wind_rose = False,
                                        noise_in_wind_rose = False)
    if objective == 'AEP':
        obj_robust = aep
    elif objective == 'LCOE':
        obj_robust = lcoe
    
    # Make dataframe of parameters to be saved
    df = pd.DataFrame(data=
         {"Robust objective": [obj_robust],
          "Orginial objective": [obj_org],
          "Objective": [objective],
          "Joint/Sequetial": [opt_type],
          "Num_run": [num_run],
          "Layout": [layout]})
    
    # Go to folder where results should be saved
    os.chdir(owd)
    results_sub_folder= "results/GA/{}".format(folder_name_save)
    if not os.path.exists(results_sub_folder):
        os.makedirs(results_sub_folder)
    if platform == "linux" or platform == "linux2": 
        os.chdir(results_sub_folder)
    
    # Save robust objective as csv
    if platform == "linux" or platform == "linux2": 
        filename = "results_robust_{}_{}_{}.csv".format(num_run, objective, opt_type)
    elif platform == "win32" or platform == "darwin":    
        filename = "{}/results_robust_{}_{}_{}.csv".format(results_sub_folder, num_run, objective, opt_type)
    df.to_csv(filename, index=False)
    
    # Save turbine powers as json
    if platform == "linux" or platform == "linux2": 
        filename = "turbine_powers_{}_{}_{}.json".format(num_run, objective, opt_type)
    elif platform == "win32" or platform == "darwin":    
        filename = "{}/turbine_powers_{}_{}_{}.json".format(results_sub_folder, num_run, objective, opt_type)
    power_per_turbine_windbin.dump(filename)
    
else:
    print("File doesn't exists: {}".format(filename_results))
    
print("Run time: {}".format(datetime.now()-t))
print("End time: {}".format(datetime.now()))