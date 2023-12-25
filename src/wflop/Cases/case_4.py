# Standard library imports
import json
import os
from   sys                                                    import platform
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

folder_name = "Case_4/Iteration_GA_2"
objective = 'LCOE'
ecology_list = [2500, 3000, 3500, 4000, 4500, 5000]

num_run_file = open('run_num_case_4.txt', 'r')
num_run_org = num_run_file.read()
num_run_file.close()

eco_level = ecology_list[int(num_run_org) % len(ecology_list)]
num_run = int(num_run_org) // len(ecology_list)

print("num_run_org = {}".format(num_run_org))
print("num_run = {}".format(num_run))
print("ecology level = {}".format(eco_level))
print("folder_name = {}".format(folder_name))

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
# Set up folder for results
################################################
      
results_sub_folder= "results/GA/{}".format(folder_name)
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)

if platform == "linux" or platform == "linux2": 
    os.chdir(results_sub_folder)
    
###########################################################
# Functions
###########################################################

def ecology_domain(distance, domain):
    # Generate empty new domian
    domain_new       = np.empty((0,3), float)
    
    # End points of the closest border of the bruine bank to the alpha 1 domain of Ijmuiden ver
    bruine_bank = np.array([[520457,5848100],[533881,5843854]])
    
    for point in domain:
        
        # Get smallest distance from point in domain to the bruine bank
        p = np.asarray([point[0], point[1]])
        d = np.linalg.norm(np.cross(bruine_bank[1]-bruine_bank[0], bruine_bank[0]-p))/np.linalg.norm(bruine_bank[1]-bruine_bank[0])
        
        # Check if the distance is larger than the given contraint and if so add the point to the new domain
        if abs(d) > distance:
            domain_new = np.append(domain_new, np.array([[point[0], point[1], point[2]]]), axis=0)
    
    return domain_new

def generate_queue(wfga, num_run, eco_level, results_sub_folder):
    # Generate and save initial population
    initial_pop = wfga.generate_random_pop()
    if platform == "linux" or platform == "linux2": 
        filename_initial_pop = "initial_pop_{}_{}.csv".format(num_run, eco_level)
    elif platform == "win32" or platform == "darwin":    
        filename_initial_pop = "{}/initial_pop_{}_{}.csv".format(results_sub_folder, num_run, eco_level)
    if not os.path.exists(filename_initial_pop):
        np.savetxt(filename_initial_pop, initial_pop)
        
    # Put items in the queue 
    queue = Queue()    
    queue.put([copy.deepcopy(wfga), initial_pop, eco_level, num_run, objective, "geometric_yaw_Jong"])   
    queue.put([copy.deepcopy(wfga), initial_pop, eco_level, num_run, objective, "None"])     
    
    return queue  

def func(wfga, initial_pop, eco_level, num_run, objective, yaw):    

    # Create filename
    k = 'Sequential'
    if yaw == "geometric_yaw_Jong":
        k = "Joint"
    if platform == "linux" or platform == "linux2": 
        filename_results = "results_{}_{}_{}_{}.csv".format(num_run, objective, k, eco_level)
    elif platform == "win32" or platform == "darwin":    
        filename_results = "{}/results_{}_{}_{}_{}.csv".format(results_sub_folder, num_run, objective, k, eco_level)
    
    # Run GA
    if not os.path.exists(filename_results):    
        try:    
            wfga.yaw_optimizer = yaw    
            wfga.objective     = objective 
            wfga.genetic_alg(    results_file = filename_results,
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

# Get the domain based on the ecology level
eco_domain           = ecology_domain(eco_level, domain)

# Generate wfga object
wfga = Wflop_ga(parameters  =   parameters,
            wind_rose           =   wind_rose,
            domain              =   eco_domain,
            floris_wind_farm    =   floris_wind_farm,
            substation          =   [532359.5, 5851358.4],
            print_progress      =   True)

queue = generate_queue(wfga, num_run, eco_level, results_sub_folder)
while not queue.empty():
    [wfga, initial_pop, eco_level, num_run, objective, yaw] = queue.get()
    func(wfga, initial_pop, eco_level, num_run, objective, yaw)

print("Run time: {}".format(datetime.now()-t))
print("End time: {}".format(datetime.now()))