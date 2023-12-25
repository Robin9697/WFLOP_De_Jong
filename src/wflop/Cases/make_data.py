# Standard library imports
import json
import os
from   sys                                                    import platform

# Third party imports
import pandas                                                 as     pd
from   datetime                                               import datetime 
import numpy                                                  as     np

# Local application imports
from   wflop.genetic_algorithm.WFLOP_GA_class                 import Wflop_ga

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
# Input
################################################

# case 0: Geometric yaw
# case 1: Test case
# case 2: IJmuiden Ver: AEP and LCOE
# case 3: IJmuiden Ver: Robustness
# case 4: IJmuiden Ver: Ecology
# case 5: IJmuiden Ver: Penalty
case = 2

if case == 0:
    num_run = range(50)
elif case == 2:
    objective_list = ['LCOE', 'AEP']
    x_list = [2000]
    num_run = range(50)
elif case == 1:
    objective_list = ['AEP']
    num_run = range(100)
    x_list = np.arange(8, 21)
elif case == 3:
    objective_list = ['LCOE', 'AEP']
    num_run = range(20)
    robust_list = ['Original objective', 'FLORIS yaw', '360 wind directions', '22 wind speeds', 'Robust objective']
elif case == 4:
    objective_list = ['LCOE']
    num_run = range(50)
    x_list = [2000, 2500, 3000, 3500, 4000, 4500, 5000]
elif case == 5:
    objective_list = ['LCOE']
    num_run = range(50)
    x_list = [2000, 2500, 3000, 3500, 4000, 4500, 5000]
    
###########################################################
# Functions
###########################################################   
    
def ecology_domain(distance, domain):
    # Generate empty new domain
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

def dataframe_data(case):
    
    # Make dataframe of parameters to be saved
    if case == 0:
        df_data = pd.DataFrame(columns=
                        ["num_run",
                        "Yaw optimization type",
                        "Joint/Sequential",
                        "AEP",
                        "LCOE",
                        "Layout"])
    elif case == 1:
        df_data = pd.DataFrame(columns=
                        ["num_run",
                        "Objective",
                        "Optimization type",
                        "Power density",
                        "AEP",
                        "Layout",
                        "Iterations"])
    elif case == 2:
        df_data = pd.DataFrame(columns=
                        ["num_run",
                        "Objective",
                        "Optimization type",
                        "AEP",
                        "LCOE",
                        "Layout",
                        "Iterations"])
    elif case == 3:
        df_data = pd.DataFrame(columns=
                        ["num_run",
                        "Objective",
                        "Optimization type",
                        "Robustness",
                        "AEP",
                        "LCOE",
                        "Layout"])
    elif case in [4,5]:
        df_data = pd.DataFrame(columns=
                            ["num_run",
                            "Objective",
                            "Optimization type",
                            "Distance to Bruine Bank",
                            "AEP",
                            "LCOE",
                            "Layout",
                            "Iterations"])
        
    return df_data

def case1245(case, x_list, num_run, objective_list, parameters, wind_rose, domain, floris_wind_farm):
    
    # Make dataframe of parameters to be saved
    df_data = dataframe_data(case)

    for x_item in x_list:
        # Update domain to ecology level in case 4 and 5
        if case in [4,5]:
            domain_updated          = ecology_domain(x_item, domain)
        else:
            domain_updated          = domain

        # Generate wfga object
        wfga = Wflop_ga(parameters  =   parameters,
                wind_rose           =   wind_rose,
                domain              =   domain_updated,
                floris_wind_farm    =   floris_wind_farm,
                substation          =   [532359.5, 5851358.4])

        for i in num_run:
            for objective in objective_list:
                for opt_type in ['Joint', 'Sequential']:
                    
                    # Get results filename
                    if case == 1:
                        filename_results = "results/GA/Case_{}/Iterations_GA/results_{}_{}_{}.csv".format(case, i, x_item, opt_type)
                    elif case == 2:
                        filename_results = "results/GA/Case_{}/Iterations_GA/results_{}_{}_{}.csv".format(case, i, objective, opt_type)
                    elif case in [4,5]:
                        if x_item == 2000:
                            filename_results = "results/GA/Case_2/Iterations_GA/results_{}_{}_{}.csv".format(i, objective, opt_type)
                        else:
                            filename_results = "results/GA/Case_4/Iterations_GA/results_{}_{}_{}_{}.csv".format(i, objective, opt_type, x_item)
                    
                    if os.path.exists(filename_results):    
                        # Get layout
                        df = pd.read_csv(filename_results, header=None, names=['objective', 'layout', 'time'])
                        layout = df['layout'].iloc[-1]
                        layout = layout.replace('  ', ' ').replace('  ', ' ').replace('[ ', '').replace(']', '')
                        layout = [int(i) for i in layout.split(' ')]

                        # Compute objective
                        if case == 1:
                            aep = df['objective'].iloc[-1]
                        elif case == 5:
                            aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                            optimizer = "None", 
                                                            complete_wind_rose = False)
                        else:
                            aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                            optimizer = "geometric_yaw_Jong", 
                                                            complete_wind_rose = False)
                        
                        # Save layout and objectives to dataframe
                        if case == 1:
                            df_data.loc[-1] = [i, objective, opt_type, x_item, aep, layout, len(df)]
                        elif case == 2:
                            df_data.loc[-1] = [i, objective, opt_type, aep, lcoe, layout, len(df)]
                        else:
                            df_data.loc[-1] = [i, objective, opt_type, x_item, aep, lcoe, layout, len(df)]
                        df_data.index = df_data.index + 1
                        df_data = df_data.sort_index()
                        df_data.to_csv("results/GA/Case_{}/Data_case_{}.csv".format(case, case), index=False)
                        
                        # Save turbine powers as json
                        if case in [2,4,5]:
                            filename = "results/GA/Case_{}/Turbine_powers/turbine_powers_{}_{}_{}.json".format(case, i, objective, opt_type)
                            power_per_turbine_windbin.dump(filename)
                        
                    else:
                        print("File doesn't exists: {}".format(filename_results))
                        
def case3(parameters, wind_rose, domain, floris_wind_farm, num_run, objective_list, robust_list, case):
    
    # Make dataframe of parameters to be saved
    df_data = dataframe_data(case)

    # Generate wfga object
    wfga = Wflop_ga(parameters  =   parameters,
            wind_rose           =   wind_rose,
            domain              =   domain,
            floris_wind_farm    =   floris_wind_farm,
            substation          =   [532359.5, 5851358.4])
    
    # Get data case 2
    filename_case_2 = "results/GA/Case_2/Data_case_2.csv"
    if os.path.exists(filename_case_2):
        df_case_2 = pd.read_csv(filename_case_2, header=0)
    else:
        print("Case 3 can only be run after case 2.")
    
    for i in num_run:
        for objective in objective_list:
            for opt_type in ['Joint', 'Sequential']:
                   
                # Get layout 
                df_item = df_case_2.loc[(df_case_2['num_run']==i) &
                                        (df_case_2['Objective']==objective) &
                                        (df_case_2['Optimization type']==opt_type)]
                layout = df_item['Layout'].iloc[0].replace('[', '').replace(']', '').split(', ')
                layout = [int(i) for i in layout]
                    
                for robustness in robust_list:
                    
                    # Get AEP and LCOE depending on the robustness case
                    if robustness == "Original objective":
                         aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                                optimizer = "geometric_yaw_Jong", 
                                                                complete_wind_rose = False)
                    elif robustness == "FLORIS yaw":
                        filename_results = "results/GA/Case_3/FLORIS_yaw/results_robust_{}_{}_{}.csv".format(i, objective, opt_type)
                        df_robust = pd.read_csv(filename_results, header=0)
                        layout2 = df_robust['Layout'].iloc[0].replace('[', '').replace(']', '').split(', ')
                        layout2 = [int(i) for i in layout2]
                        aep = None
                        lcoe = None
                        if layout != layout2:
                            print('Layouts are not the same.')
                        elif objective == 'AEP':
                            aep = df_robust['Robust objective'].iloc[0]
                        elif objective == 'LCOE':
                            lcoe = df_robust['Robust objective'].iloc[0]
                    elif robustness == '360 wind directions':
                        parameter_save = wfga.n_wind_direction_bins
                        wfga.n_wind_direction_bins = 360
                        aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                                optimizer = "geometric_yaw_Jong", 
                                                                complete_wind_rose = False)
                        wfga.n_wind_direction_bins = parameter_save
                    elif robustness == '22 wind speeds':
                        parameter_save = wfga.n_wind_speed_bins
                        wfga.n_wind_speed_bins = 22
                        aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                                optimizer = "geometric_yaw_Jong", 
                                                                complete_wind_rose = False)   
                        wfga.n_wind_speed_bins = parameter_save
                    elif robustness == 'Robust objective':      
                        filename_results = "results/GA/Case_3/Robust_objective/results_robust_{}_{}_{}.csv".format(i, objective, opt_type)
                        df_robust = pd.read_csv(filename_results, header=0)
                        layout2 = df_robust['Layout'].iloc[0].replace('[', '').replace(']', '').split(', ')
                        layout2 = [int(i) for i in layout2]
                        if layout != layout2:
                            print('Layouts are not the same.')
                            layout = layout2
                        if objective == 'AEP':
                            aep = df_robust['Robust objective'].iloc[0]
                            lcoe = None
                        elif objective == 'LCOE':
                            aep = None
                            lcoe = df_robust['Robust objective'].iloc[0]
                    
                    # Save layout and objectives to dataframe    
                    df_data.loc[-1] = [i, objective, opt_type, robustness, aep, lcoe, layout]
                    df_data.index = df_data.index + 1
                    df_data = df_data.sort_index()
                    df_data.to_csv("results/GA/Case_{}/Data_case_{}.csv".format(case, case), index=False)
                    
                    # Save turbine powers as json
                    if robustness not in ["FLORIS yaw", 'Robust objective']:
                        filename = "results/GA/Case_{}/Turbine_powers/turbine_powers_{}_{}_{}_{}.json".format(case, i, objective, opt_type, robustness.replace(' ', '_'))
                        power_per_turbine_windbin.dump(filename)

def case0(num_run, parameters, wind_rose, domain, floris_wind_farm):
    
    # Make dataframe of parameters to be saved
    df_data = dataframe_data(case)

    # Generate wfga object
    wfga = Wflop_ga(parameters  =   parameters,
            wind_rose           =   wind_rose,
            domain              =   domain,
            floris_wind_farm    =   floris_wind_farm,
            substation          =   [532359.5, 5851358.4])
    
    # Get data case 3
    filename_case_3 = "results/GA/Case_3/Data_case_3.csv"
    if os.path.exists(filename_case_3):
        df_case_3 = pd.read_csv(filename_case_3, header=0)
    else:
        print("Case 0 can only be run after case 3.")
    
    for i in num_run:
        for opt_type in ['Joint', 'Sequential']:
            
            # Get layout     
            df_item = df_case_3.loc[(df_case_3['num_run']==i)&
                                    (df_case_3['Optimization type']==opt_type)]
            layout = df_item['Layout'].iloc[0].replace('[', '').replace(']', '').split(', ')
            layout = [int(i) for i in layout]
                
            for yaw_opt in ["yaw_optimizer_floris", "geometric_yaw_Jong", "geometric_yaw_Stanley", "None"]:

                # Get AEP and LCOE depending on the yaw angle optimization
                if yaw_opt == "yaw_optimizer_floris":
                    aep = df_item.loc[(df_item['Robustness']=="FLORIS yaw") &
                                    (df_case_3['Objective']=='AEP')].iloc[0]['AEP']
                    lcoe = df_item.loc[(df_item['Robustness']=="FLORIS yaw") &
                                    (df_case_3['Objective']=='LCOE')].iloc[0]['LCOE']
                elif yaw_opt == "geometric_yaw_Jong":
                    aep = df_item.loc[(df_item['Robustness']=="Original objective") &
                                    (df_case_3['Objective']=='AEP')].iloc[0]['AEP']
                    lcoe = df_item.loc[(df_item['Robustness']=="Original objective") &
                                    (df_case_3['Objective']=='LCOE')].iloc[0]['LCOE']
                elif yaw_opt == "geometric_yaw_Stanley":
                    aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                        optimizer = yaw_opt, 
                                                        complete_wind_rose = False)
                elif yaw_opt == "None":
                    aep, lcoe, power_per_turbine_windbin = wfga.robust_objective(layout, 
                                                        optimizer = yaw_opt, 
                                                        complete_wind_rose = False)
                
                # Save layout and objectives to dataframe
                df_data.loc[-1] = [i, yaw_opt, opt_type, aep, lcoe, layout]
                df_data.index = df_data.index + 1
                df_data = df_data.sort_index()
                df_data.to_csv("results/GY/Data_comparison.csv", index=False)
       
################################################
# Run
################################################

t = datetime.now()
print("Start time: {}".format(t))

if case == 0:
    case0(num_run, parameters, wind_rose, domain, floris_wind_farm)
elif case in [1,2,4,5]:
    case1245(case, x_list, num_run, objective_list, parameters, wind_rose, domain, floris_wind_farm)
elif case == 3:
    case3(case, robust_list, num_run, objective_list, parameters, wind_rose, domain, floris_wind_farm)
    
print("Run time: {}".format(datetime.now()-t))
print("End time: {}".format(datetime.now()))