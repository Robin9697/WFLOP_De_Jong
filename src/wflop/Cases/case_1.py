# Standard library imports
import json
import os
from   sys                                                    import platform
import math
import copy

# Third party imports
import pandas                                                 as     pd
import numpy                                                  as     np
from   datetime                                               import datetime
from   queue                                                  import Queue

# Local application imports
from   wflop.genetic_algorithm.WFLOP_GA_class                 import Wflop_ga

################################################
# Input
################################################

folder_name = "Case_1/Iteration_GA_2"
rho_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

num_run_file = open('run_num_case_1.txt', 'r')
num_run_org = num_run_file.read()
num_run_file.close()

rho = rho_list[int(num_run_org) % len(rho_list)]
num_run = int(num_run_org) // len(rho_list)

print("num_run_org = {}".format(num_run_org))
print("num_run = {}".format(num_run))
print("rho = {}".format(rho))
print("folder_name = {}".format(folder_name))

################################################
# Importing data from files
################################################

class PlatformError(Exception):
    pass

owd = os.getcwd()

if platform == "linux" or platform == "linux2":   

    os.chdir("wflop/inputs")
    
    parameters       = json.load(open("parameters_case_1.json"))
    
    floris_wind_farm = json.load(open("wind_farm.json"))
    
    # wind_rose.csv should have the following colums: wind speed, wind direction, frequency
    wind_rose        = pd.read_csv("wind_rose.csv", \
                            header      = 0 , \
                            index_col   = False)
    wind_rose        = wind_rose.to_numpy()  
    
    os.chdir(owd)

elif platform == "win32" or platform == "darwin":

    current_file     = os.path.abspath(os.getcwd())

    data_filename    = os.path.join(current_file, "wflop\inputs\parameters_case_1.json")
    parameters       = json.load(open(data_filename))
    
    data_filename    = os.path.join(current_file, "wflop\inputs\wind_farm.json")
    floris_wind_farm = json.load(open(data_filename))
    
    # wind_rose.csv should have the following colums: wind speed, wind direction, frequency
    data_filename    = os.path.join(current_file, "wflop\inputs\wind_rose.csv")
    wind_rose        = pd.read_csv(data_filename, \
                            header      = 0 , \
                            index_col   = False)
    wind_rose        = wind_rose.to_numpy()  
    
################################################
# Set up folder for results
################################################
      
results_sub_folder = "results/GA/{}".format(folder_name)
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)

if platform == "linux" or platform == "linux2": 
    os.chdir(results_sub_folder)
    
###########################################################
# Functions
###########################################################

def generate_queue():        
    # Generate wfga object
    domain = 1000*math.sqrt((parameters["n_turbines"]*15)/rho)
    wfga = Wflop_ga(parameters  =   parameters,
            wind_rose           =   wind_rose,
            domain              =   domain,
            floris_wind_farm    =   floris_wind_farm,
            substation          =   [0,0],
            print_progress      =   True)

    # Generate and save initial population
    initial_pop = wfga.generate_random_pop()
    if platform == "linux" or platform == "linux2": 
        filename_initial_pop = "initial_pop_{}_{}.csv".format(num_run, rho)
    elif platform == "win32" or platform == "darwin":    
        filename_initial_pop = "{}/initial_pop_{}_{}.csv".format(results_sub_folder, num_run, rho)
    if not os.path.exists(filename_initial_pop):
        np.savetxt(filename_initial_pop, initial_pop)
    
    # Put items in queue
    queue = Queue()    
    queue.put([copy.deepcopy(wfga), initial_pop, num_run, rho, "geometric_yaw_Jong"])       
    queue.put([copy.deepcopy(wfga), initial_pop, num_run, rho, "None"])  
    
    return queue      

def func(wfga, initial_pop, i, rho, yaw):    

    # Create filename
    k = 'Sequential'
    if yaw == "geometric_yaw_Jong":
        k = "Joint"
    if platform == "linux" or platform == "linux2": 
        filename_results = "results_{}_{}_{}.csv".format(i, rho, k)
    elif platform == "win32" or platform == "darwin":    
        filename_results = "{}/results_{}_{}_{}.csv".format(results_sub_folder, i, rho, k)
    
    # Run GA
    if not os.path.exists(filename_results):    
        try:    
            wfga.yaw_optimizer = yaw    
            wfga.genetic_alg(results_file = filename_results,
                             initial_pop  = initial_pop     )
        except KeyError or AttributeError:
            print('Error has occured in: {}'.format(filename_results))
    else:
        print("File already exists: {}".format(filename_results))

################################################
# Run
################################################

t = datetime.now()
print("Start time: {}".format(t))

queue = generate_queue()
while not queue.empty():
    [wfga, initial_pop, i, rho, yaw] = queue.get()
    func(wfga, initial_pop, i, rho, yaw)
   
print("Run time: {}".format(datetime.now()-t))
print("End time: {}".format(datetime.now()))