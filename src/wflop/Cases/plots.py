# Standard library imports
import json
import os
from   sys                                                    import platform
import math
import copy
import statistics

# Third party imports
import pandas                                                 as     pd
import matplotlib.pyplot                                      as     plt
from   matplotlib                                             import colors
import numpy                                                  as     np
import floris.tools.visualization                             as     wakeviz
from   pylab                                                  import text
import plotly.express                                         as     px
import astropy.convolution                                    as     ac

# Local application imports
from wflop.genetic_algorithm.WFLOP_GA_class                   import Wflop_ga
from wflop.yaw_optimizers.geometric_yaw_Jong                  import geometric_yaw_Jong

################################################
# Input
################################################

case = 5 # 1: test, 2: IJmuiden Ver, 3: robust, 4: ecology, 5: penalty
plot_num = 6 # 1: boxplot, 2: wake, 3: power density, 4: wind rose, 5: domain, 6: GY comparison

plot_names = ['', 'box_plot', 'wake_plot', 'power_density_plot', 'dx_dy_plot', 'wind_rose', 'domain', 'GA_plot']
title = 'results/Figures/Case_{}_{}.png'.format(case, plot_names[plot_num])

# Robust cut
objective = 'LCOE'
if objective == 'AEP':
    robust_cut = [2820, 2900, 5000, 5060, 10]
elif objective == 'LCOE':
    robust_cut = [-117, -114, -66, -63, 0.2]
    
# Layout limits
x_y_lim = [522000, 536000, 5847000, 5863000]

# Parameters per case and plot
if case == 1:
    folder_name = "Case_1"
    objective = 'AEP'
    x_as_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] #boxplot
    num_runs_list = range(100) #box_plot
    x_y_lim = [0, 6000, 0, 6000]
    if plot_num in [2,5]: #wake_plot
        parameters_list = [8, 12, 16, 20]
    else:
        parameters_list = x_as_list 
elif case == 2:
    folder_name = "Case_2"
    x_as_list = ['AEP', 'LCOE'] #box_plot
    num_runs_list = np.arange(50) #box plot and power_density_plot
    parameters_list = ['Sequential AEP'] #['Joint AEP', 'Joint LCOE', 'Sequential AEP', 'Sequential LCOE'] #wake_plot
    parameters_list_PD = ['Joint AEP', 'Joint LCOE', 'Initial layout', 'Sequential AEP', 'Sequential LCOE', 'Initial layout'] #power_denstiy_plot
elif case == 3:
    folder_name = "Case_3"
    x_as_list = ['Original objective', 'FLORIS yaw', '360 wind directions', '22 wind speeds', 'Robust objective'] #box_plot
    num_runs_list = range(20)
    parameters_list = x_as_list #wake_plot
elif case == 4:
    folder_name = "Case_4"
    objective = 'LCOE'
    x_as_list = [2000, 2500, 3000, 3500, 4000, 4500, 5000] #boxplot
    num_runs_list = range(50) #box_plot
    parameters_list = [2000, 3000, 4000, 5000] #wake_plot
    parameters_list_PD = [2000, 3000, 4000, 5000] #power_denstiy_plot
elif case == 5:
    folder_name = "Case_5"
    objective = 'LCOE'
    x_as_list = [2000, 2500, 3000, 3500, 4000, 4500, 5000] #boxplot
    num_runs_list = np.arange(50) #box plot and power_density_plot
data_folder = "results/GA/{}".format(folder_name)
    
################################################
# Importing data from files
################################################

if platform == "linux" or platform == "linux2":   
    print("Not adjusted for linux.")

elif platform == "win32" or platform == "darwin":
    current_file     = os.path.abspath(os.getcwd())

    data_filename    = os.path.join(current_file, "wflop\inputs\parameters_case_1.json")
    parameters1       = json.load(open(data_filename))
    data_filename    = os.path.join(current_file, "wflop\inputs\parameters_case_2.json")
    parameters2       = json.load(open(data_filename))

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
    
#############################################
# Plot functions
#############################################      
        
def set_box_color(bp, edge_color, fill_color, greyscale = False):
    
    # Convert colors to greyscale
    if greyscale:
        ec = np.dot(colors.to_rgba(edge_color), [0.2989, 0.5870, 0.1140, 0])
        edge_color = (ec, ec, ec, 1)
        fc = np.dot(colors.to_rgba(fill_color), [0.2989, 0.5870, 0.1140, 0])
        fill_color = (fc, fc, fc, 1)
    
    # Set color of every boxplot element
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
        
def show_improvement(bpj, data_joint, data_seq, objective, textdy = 0.1):
    for i in range(len(bpj['medians'])):
        
        # Get position of median line
        line = bpj['medians'][i]
        x, y = line.get_xydata()[0] + 0.5*(line.get_xydata()[1]-line.get_xydata()[0])
        
        # Compute improvement
        imp = 100*((np.median(data_joint[i])/np.median(data_seq[i])) -1)
        if objective == 'LCOE':
            imp = -imp
            
        # Overlay improvement obove median line
        if imp > 0:
            text(x, y+textdy, "+{}%".format(round(imp,2)), horizontalalignment='center', weight='bold')
        else:
            text(x, y+textdy, "{}%".format(round(imp,2)), horizontalalignment='center', weight='bold')

def latex_table_median_and_variance(data, col1_list, col2_list = [''], col3_list = ['','']):
    
    # Create string
    latex_table = str()
          
    for i in range(len(data[0])):
        
        # Get value of the first column
        col1 = col1_list[i] 
        if type(col1) == int:
            if col1 >= 2000:
                col1 = round(col1/1000., 1)
                
        for j in range(len(data)):
            
            # Get value of the second and third column
            col2 = col2_list[j]
            col3 = col3_list[j]
            
            # Get median and variance
            var = statistics.variance(data[j][i])
            med = statistics.median(data[j][i])
            
            # Add latex table row to string
            if col2 == '':
                latex_table = "{}{}  & {:.5f} & {:.5f} \\\ \hline \n".format(latex_table,
                          col1, med, var)
            elif col3 == '':
                latex_table = "{}{} & {} & {:.5f} & {:.5f} \\\ \hline \n".format(latex_table,
                          col1, col2, med, var)
            else:
                latex_table = "{}{} & {} & {} & {:.5f} & {:.5f} \\\ \hline \n".format(latex_table,
                          col1, col2, col3, med, var)
            
    print(latex_table)
        
def box_plot(case, objective, num_runs_list, x_as_list, robust_cut, title = None):        
    
    # Make sure box plot is made twice in case 5
    case_list = [case]
    if case == 5:
        case_list = [4,5]

    for case in case_list:
        data_joint, data_seq, labels = [], [], []
        
        # Put data from data file in dataframe
        df_data = pd.read_csv("results/GA/Case_{}/Data_case_{}.csv".format(case, case), header=0)
        if case != 2:
            df_data = df_data.loc[df_data['Objective']==objective]
            
        for x_item in x_as_list:
            
            # Get data for specific x axis item
            if case == 1:
                labels.append(x_item)
                df1 = df_data.loc[df_data['Power density'] == x_item]
            elif case == 2:
                labels.append('Optimized for {}'.format(x_item))
                objective = x_item
                df1 = df_data.loc[df_data['Objective'] == objective]
            elif case == 3:
                labels.append(x_item)
                df1 = df_data.loc[df_data['Robustness'] == x_item]
            elif case in [4,5]:
                labels.append(round(x_item/1000., 1))
                df1 = df_data.loc[df_data['Distance to Bruine Bank'] == x_item]
            
            # Add datapoints of the N joint a sequential runs to the corresponding list 
            dat_joint, dat_seq = [], []
            for i in num_runs_list:  
                df2 = df1.loc[df1['num_run']==i]
                df_data_joint = df2.loc[df2['Optimization type']=='Joint']
                df_data_seq = df2.loc[df2['Optimization type']=='Sequential']
                obj_joint = df_data_joint[objective].iloc[0]
                obj_seq = df_data_seq[objective].iloc[0]
                dat_joint.append(obj_joint)
                dat_seq.append(obj_seq)      
            data_joint.append(dat_joint)
            data_seq.append(dat_seq)
        
        # Print latex table
        if len(case_list) == 2 and case == 4:
            data3 = data_seq
            data4 = data_joint
        elif case == 5:
            latex_table_median_and_variance([data_seq, data_joint, data3, data4], x_as_list, 
                                            ['Sequential', 'Joint', 'Sequential', 'Joint'],
                                            ['Without', 'Without', 'With', 'With'])
        else:
            latex_table_median_and_variance([data_seq, data_joint], x_as_list, ['Sequential', 'Joint'])
        
        # Create box plot
        if case == 2:
            fig, ax1 = plt.subplots(figsize=(10*0.7, 8*0.7)) 
            ax1.set_ylabel(r"AEP $\left( GWh \right)$") 
            ax1.tick_params(axis ='y') 
            for ax_num in [0]:
                ax = ax1
                bps = ax.boxplot([data_seq[ax_num]], positions=np.array([ax_num])*2.0-0.4, widths=0.6, patch_artist=True)
                bpj = ax.boxplot([data_joint[ax_num]], positions=np.array([ax_num])*2.0+0.4, widths=0.6, patch_artist=True)
                set_box_color(bps, 'indigo', 'mediumslateblue')
                set_box_color(bpj, 'steelblue', 'powderblue')
                show_improvement(bpj, [data_joint[ax_num]], [data_seq[ax_num]], 'AEP', 0.2)
            ax2 = ax1.twinx()
            ax2.tick_params(axis ='y') 
            ax2.set_ylabel(r"LCOE $\left( \frac{€}{MWh} \right)$") 
            for ax_num in [1]:
                ax = ax2
                bps = ax.boxplot([data_seq[ax_num]], positions=np.array([ax_num])*2.0-0.4, widths=0.6, patch_artist=True)
                bpj = ax.boxplot([data_joint[ax_num]], positions=np.array([ax_num])*2.0+0.4, widths=0.6, patch_artist=True)
                set_box_color(bps, 'indigo', 'mediumslateblue')
                set_box_color(bpj, 'steelblue', 'powderblue')
                show_improvement(bpj, [data_joint[ax_num]], [data_seq[ax_num]], 'LCOE', 0.007)
            plt.legend([bps["boxes"][0], bpj["boxes"][0]], ['Sequential', 'Joint'], loc='upper center')
            
        elif case == 3:
            fig, ax = plt.subplots(figsize=(10*0.8, 8*0.8), constrained_layout = True)
            data_seq_cut = copy.deepcopy(data_seq)
            data_seq_cut[3] = np.subtract(data_seq[3], (robust_cut[2]-robust_cut[1]) - 2*robust_cut[4])
            data_seq_cut[4] = np.subtract(data_seq[4], (robust_cut[2]-robust_cut[1]) - 2*robust_cut[4])
            data_joint_cut = copy.deepcopy(data_joint)
            data_joint_cut[3] = np.subtract(data_joint[3], (robust_cut[2]-robust_cut[1]) - 2*robust_cut[4])
            data_joint_cut[4] = np.subtract(data_joint[4], (robust_cut[2]-robust_cut[1]) - 2*robust_cut[4])
            bps = plt.boxplot(data_seq_cut, positions=np.array(range(len(data_seq)))*2.0-0.4, widths=0.6, patch_artist=True)
            bpj = plt.boxplot(data_joint_cut, positions=np.array(range(len(data_joint)))*2.0+0.4, widths=0.6, patch_artist=True)
            set_box_color(bps, 'indigo', 'mediumslateblue')
            set_box_color(bpj, 'steelblue', 'powderblue')
            show_improvement(bpj, data_joint, data_seq, objective, 0.02)
            plt.legend([bps["boxes"][0], bpj["boxes"][0]], ['Sequential', 'Joint'], loc='upper right')
            
            y_range = np.arange(robust_cut[0], robust_cut[3] - (robust_cut[2]-robust_cut[1]) + 3*robust_cut[4], robust_cut[4])
            labelsy = []
            for lab in np.arange(robust_cut[0], robust_cut[1] + 0.5*robust_cut[4], robust_cut[4]):
                labelsy.append(str(round(lab,1)))
            labelsy.append(r"$\vdots$")
            for lab in np.arange(robust_cut[2], robust_cut[3] + 0.5*robust_cut[4], robust_cut[4]):
                labelsy.append(str(round(lab,1)))
            plt.yticks(y_range, labelsy)
            
        else:
            if case != 5:
                fig, ax = plt.subplots(figsize=(10, 8), constrained_layout = True)
            greyscale = False
            if len(case_list) == 2 and case == 4:
                greyscale = True
            bps = ax.boxplot(data_seq, positions=np.array(range(len(data_seq)))*2.0-0.4, widths=0.6, patch_artist=True)
            bpj = ax.boxplot(data_joint, positions=np.array(range(len(data_joint)))*2.0+0.4, widths=0.6, patch_artist=True)
            set_box_color(bps, 'indigo', 'mediumslateblue', greyscale)
            set_box_color(bpj, 'steelblue', 'powderblue', greyscale)
            show_improvement(bpj, data_joint, data_seq, objective, 0.02)
            if case == 5:
                leg2 = ax.legend([bps["boxes"][0], bpj["boxes"][0]], 
                           ['Sequential', 'Joint'], 
                           loc='upper left', title='Without yaw control:', alignment='left',
                           bbox_to_anchor = (0.8,0.5))
                ax.add_artist(leg1)
            elif len(case_list) == 2 and case == 4:
                leg1 = ax.legend([bps["boxes"][0], bpj["boxes"][0]], 
                           ['Sequential', 'Joint'], 
                           loc='upper left', title='With yaw control:', alignment='left',
                           bbox_to_anchor = (0.8,1.))
            else:
                ax.legend([bps["boxes"][0], bpj["boxes"][0]], ['Sequential', 'Joint'], loc='upper right')
    
    # Set x axis values
    plt.xticks(range(0, len(labels) * 2, 2), labels)
    
    # Set x and y lables
    if case == 1:
        plt.xlabel(r"Power density $\left( \frac{W}{m^2} \right)$")
    elif case in [4,5]:
        plt.xlabel(r"Distance to nature reserve $(km)$")
    if objective == 'LCOE':
        plt.ylabel(r"LCOE $\left( \frac{€}{MWh} \right)$")
    elif objective == 'AEP':
        plt.ylabel(r"AEP $\left( GWh \right)$") 
    plt.tight_layout()
    
    # Save or show figure
    if title is not None:
        plt.savefig(title)
    else:
        plt.show()
        
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

def wake_plot(case, parameters_list, x_y_lim, title, best_layout,
              parameters1, parameters2, wind_rose, domain, floris_wind_farm):    
    
    # Generate figure
    if len(parameters_list) == 1: 
        fig, axarr = plt.subplots(1, 1, figsize=(7*((x_y_lim[1]-x_y_lim[0])/(x_y_lim[3]-x_y_lim[2])), 7))
        axarr = [axarr]
    elif case == 1:
        fig, axarr = plt.subplots(1, 4, figsize=(13, 3))
        axarr = axarr.flatten()
    else:  
        fig, axarr = plt.subplots(2, 2, figsize=(14*((x_y_lim[1]-x_y_lim[0])/(x_y_lim[3]-x_y_lim[2])), 14))
        axarr = axarr.flatten()

    # Generate wfga object for case 2 and 3
    if case in [2,3]:
        wfga = Wflop_ga(parameters  =   parameters2,
                wind_rose           =   wind_rose,
                domain              =   domain,
                floris_wind_farm    =   floris_wind_farm,
                substation          =   [532359.5, 5851358.4])

    # Get data
    df = pd.read_csv("results/GA/Case_{}/Data_case_{}.csv".format(case, case), header=0)
    
    q = -1  
    for parameter in parameters_list:  
        q = q + 1
        ax = axarr[q] 

        # Generate wfga object for case 1
        if case == 1:
            domain = 1000*math.sqrt((parameters1["n_turbines"]*15)/parameter)
            wfga = Wflop_ga(parameters  =   parameters1,
                    wind_rose           =   wind_rose,
                    domain              =   domain,
                    floris_wind_farm    =   floris_wind_farm,
                    substation          =   [0,0])

        # Get optimization type and objective for case 2
        elif case == 2:
            if 'Joint' in parameter:
                opt_typ = 'Joint'
            elif 'Sequential' in parameter:
                opt_typ = 'Sequential'
            if 'AEP' in parameter:
                objective = 'AEP'
            elif 'LCOE' in parameter:
                objective = 'LCOE'
         
        # Generate wfga object for case 4     
        elif case == 4:
            eco_domain           = ecology_domain(parameter, domain)
            wfga = Wflop_ga(parameters  =   parameters2,
                wind_rose           =   wind_rose,
                domain              =   eco_domain,
                floris_wind_farm    =   floris_wind_farm,
                substation          =   [532359.5, 5851358.4])

        if case == 1:
            # Generate simple grid layout
            ls = np.linspace(domain/8., 7*domain/8., 4)
            layout_x, layout_y = np.meshgrid(ls,ls)
            layout_x = layout_x.flatten()
            layout_y = layout_y.flatten()
            
        else:
            # Select layouts based on given parameters
            if case == 2:
                df2 = df.loc[(df['Objective']               == objective) &
                             (df['Optimization type']       == opt_typ)   ]
            elif case == 4:
                df2 = df.loc[(df['Objective']               == 'LCOE')    &
                             (df['Optimization type']       == 'Joint')   &
                             (df['Distance to Bruine Bank'] == parameter) ]
            else:
                print("Wake plot only made for case 2 and 4.")
            
            if best_layout:
                # Get best layout
                if df2['Objective'].iloc[0] == 'AEP':
                    imax = df2.idxmax()['AEP']
                elif df2['Objective'].iloc[0] == 'LCOE':
                    imax = df2.idxmax()['LCOE']
            else:
                print("Only best layout is implemented, except for case 1.")
            
            # Get aep, lcoe and layout
            aep      = df2['AEP'].loc[imax]
            lcoe     = df2['LCOE'].loc[imax]
            layout   = df2['Layout'].loc[imax]
            layout   = layout.replace('[', '').replace(']', '').split(', ')
            layout   = [int(i) for i in layout]
            layout_x = wfga.x_positions[layout]
            layout_y = wfga.y_positions[layout]
        
        # Update FLORIS object
        fi = copy.deepcopy(wfga.fi)
        fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
        
        # Generate FLORIS object with a single wind condition and calculate yaw angles
        fi2 = copy.deepcopy(fi)
        if best_layout:
            wind_direction = 225.
            fi2.reinitialize(wind_directions=[wind_direction], wind_speeds=[8.])
            yaw_angles2 = geometric_yaw_Jong(fi2)
        else:
            wind_direction = 270.
            fi2.reinitialize(wind_directions=[wind_direction], wind_speeds=[8.])
            yaw_angles2 = np.zeros((fi2.floris.flow_field.n_wind_directions, fi2.floris.flow_field.n_wind_speeds, len(fi2.floris.farm.layout_x)))
        
        # Calculate wakes
        buffer = 5000
        horizontal_plane = fi2.calculate_horizontal_plane(
                            x_resolution=200,
                            y_resolution=100,
                            height=90.0,
                            yaw_angles=yaw_angles2,
                            x_bounds=[x_y_lim[0]-buffer,x_y_lim[1]+buffer],
                            y_bounds=[x_y_lim[2]-buffer,x_y_lim[3]+buffer])
        
        # Plot wakes and turbines
        ax.set_facecolor((0.706, 0.016, 0.15))
        wakeviz.visualize_cut_plane(horizontal_plane, ax=ax)
        wakeviz.plot_turbines(ax,
                            layout_x,
                            layout_y,
                            yaw_angles2[0][0]+(270-wind_direction),
                            fi2.floris.farm.rotor_diameters[0][0])
        
        if case == 1:
            # Plot black square indicating the doamin
            ax.plot([0, 0, domain, domain], [0, domain, domain, 0], color = 'k')
            
            # Set title and limits.
            if len(parameters_list) != 1:
                ax.set_title("Power density of {} ".format(parameter) + r"$\frac{W}{m^2}$")
            ax.set_xlim([0, 6000])
            ax.set_ylim([0, 6000])
        
        else:
            # Set title and limits
            if len(parameters_list) != 1:
                ax.set_title("{} gives {} GWh or {}€/MWh".format(parameter, round(aep), round(-lcoe,2)))
            else:
                print("AEP = {} \nLCOE = {}".format(aep,lcoe))
            ax.set_xlim([x_y_lim[0],x_y_lim[1]])
            ax.set_ylim([x_y_lim[2],x_y_lim[3]])
            
        # Set labels and convert meters to km on the axis.
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        xt = np.arange(x_y_lim[0],x_y_lim[1],1000)
        yt = np.arange(x_y_lim[2],x_y_lim[3],1000)
        ax.set_xticks(xt, [str(int((i-x_y_lim[0])/1000.)) for i in xt])
        ax.set_yticks(yt, [str(int((i-x_y_lim[2])/1000.)) for i in yt])      
        
    # Save or show figure
    plt.tight_layout()
    if title is not None:
        plt.savefig(title)
    else:
        plt.show()
        
def power_density_plot(case, x_y_lim, parameters_list, title = None):

    # Generate figure
    if len(parameters_list) == 1: 
        fig, axarr = plt.subplots(1, 1, figsize=(7*((x_y_lim[1]-x_y_lim[0])/(x_y_lim[3]-x_y_lim[2])), 7))
        axarr = [axarr]
    elif case == 2:
        fig, axarr = plt.subplots(2, 3, figsize=(15*((x_y_lim[1]-x_y_lim[0])/(x_y_lim[3]-x_y_lim[2])), 10))
        axarr = axarr.flatten()
    elif case == 4:
        fig, axarr = plt.subplots(1, 4, figsize=(20*((x_y_lim[1]-x_y_lim[0])/(x_y_lim[3]-x_y_lim[2])), 5))
        axarr = axarr.flatten()

    # Generate wfga object
    wfga = Wflop_ga(parameters  =   parameters2,
            wind_rose           =   wind_rose,
            domain              =   domain,
            floris_wind_farm    =   floris_wind_farm,
            substation          =   [532359.5, 5851358.4])
    
    # Genarate random intial population
    initial_population = wfga.generate_random_pop()
    
    # Get data
    df = pd.read_csv("results/GA/Case_{}/Data_case_{}.csv".format(case, case), header=0)
    
    q = -1     
    for parameter in parameters_list:           
        q = q + 1
        ax = axarr[q] 
        
        positions = []
        
        if type(parameter) == int:
            # Update wfga object
            eco_domain                  = ecology_domain(parameter, domain)
            wfga = Wflop_ga(parameters  =   parameters2,
                wind_rose               =   wind_rose,
                domain                  =   eco_domain,
                floris_wind_farm        =   floris_wind_farm,
                substation              =   [532359.5, 5851358.4])
            
            # Get layout
            layouts = df.loc[(df['Distance to Bruine Bank'] == parameter) &
                        (df['Optimization type'] == 'Joint')]['Layout'].values

        elif 'Initial' in parameter:
                objective = 'Initial'
                positions = initial_population[:len(num_runs_list),:].flatten()
                
        else:
            # Get optimization type and objective
            if 'Joint' in parameter:
                opt_typ = 'Joint'
            elif 'Sequential' in parameter:
                opt_typ = 'Sequential'
            if 'AEP' in parameter:
                objective = 'AEP'
            elif 'LCOE' in parameter:
                objective = 'LCOE'
            
            # Get layout
            layouts = df.loc[(df['Objective'] == objective) &
                            (df['Optimization type'] == opt_typ)]['Layout'].values
        
        # Get positions from all the layouts            
        if len(positions) == 0:
            for i in num_runs_list:
                layout = layouts[i]
                layout = layout.replace('[', '').replace(']', '').split(', ')
                layout = [int(i) for i in layout]
                positions.extend(layout)
        
        # Get x and y positions
        layout_x = wfga.x_positions[positions]
        layout_y = wfga.y_positions[positions]
        
        # Convert coordiantes to 2D histogram
        xt = np.arange(x_y_lim[0],x_y_lim[1],100)
        yt = np.arange(x_y_lim[2],x_y_lim[3],100)
        H, xedges, yedges = np.histogram2d(layout_x, layout_y, bins=(xt,yt))
        
        # Smooth histogram
        data = ac.convolve(H.T, ac.kernels.Gaussian2DKernel(x_stddev=10))
        data = data/np.sum(data)
        
        # Plot histogram
        ax.imshow(data,
                    interpolation='none', origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='Spectral')    
        
        # Set title
        if case == 4:
            ax.set_title("{} km distance".format(round(parameter/1000., 1)))
        else:
            ax.set_title(parameter)
        
        # Set the values and labels of the x and y axis.
        xt = np.arange(x_y_lim[0],x_y_lim[1],1000)
        yt = np.arange(x_y_lim[2],x_y_lim[3],1000)
        ax.set_xticks(xt, [str(int((i-x_y_lim[0])/1000.)) for i in xt])
        ax.set_yticks(yt, [str(int((i-x_y_lim[2])/1000.)) for i in yt])
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
    
    # Save or plot figure
    plt.tight_layout()        
    if title is not None:
        plt.savefig(title)
    else:
        plt.show()
        
def wind_roses(wind_rose, parameters2, domain, floris_wind_farm):
    
    # Plot and save complete wind rose
    df = pd.DataFrame(wind_rose, columns = ["Wind speed (m/s)", 'direction', 'frequency'])
    ws_max = np.max(df["Wind speed (m/s)"])
    fig = px.bar_polar(df, r="frequency", theta="direction",
                    color="Wind speed (m/s)", template="none",
                    color_continuous_scale=px.colors.diverging.Spectral,
                    range_color = [0, ws_max])
    fig.write_image("results/Figures/wind_rose_complete.png", width=550, height=400, scale = 1.) 
    
    # Plot and save histogram of the wind speeds
    fig, ax = plt.subplots(1,1, figsize=(7, 5))
    wind_speeds = np.linspace(0,28,29)
    frequencies = np.sum(wind_rose[:,2].reshape(-1,360), axis=1)
    print("The average wind speeds is {} m/s.".format(round(np.sum(frequencies*wind_speeds),2)))
    print("The mode of the wind speeds is {} m/s.".format(wind_speeds[np.argmax(frequencies)]))
    ax.bar(wind_speeds,frequencies)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Wind speed (m/s)')
    plt.tight_layout()
    fig.savefig('results/Figures/wind_speeds.png')

    # Plot and save simplified wind rose
    wfga = Wflop_ga(parameters      =   parameters2,
                wind_rose           =   wind_rose,
                domain              =   domain,
                floris_wind_farm    =   floris_wind_farm,
                substation          =   [532359.5, 5851358.4])
    df = pd.DataFrame(data={"Wind speed (m/s)": wfga.wind_speeds*len(wfga.wind_directions), 
                    'direction': wfga.wind_directions, 
                    'frequency': wfga.frequencies.transpose()[0]})
    fig = px.bar_polar(df, r="frequency", theta="direction",
                    color="Wind speed (m/s)", template="none",
                    color_continuous_scale=px.colors.diverging.Spectral,
                    range_color = [0, ws_max])
    fig.write_image("results/Figures/wind_rose_simplefied.png", width=550, height=400, scale = 1.) 
    
def domain_plot(x_y_lim, parameters2, wind_rose, domain, floris_wind_farm, title = None):
    
    # Generate fiure
    adjusted_y_lim = 5845000 # Adjusted to incorperate the boundary of the nature reserve
    fig, ax = plt.subplots(1,1, figsize=(8.75*((x_y_lim[1]-x_y_lim[0])/(x_y_lim[3]-adjusted_y_lim)), 7))
   
    # Generate wfga object
    wfga = Wflop_ga(parameters      =   parameters2,
                wind_rose           =   wind_rose,
                domain              =   domain,
                floris_wind_farm    =   floris_wind_farm,
                substation          =   [532359.5, 5851358.4])
    
    # Plot water depth
    cm = plt.cm.get_cmap('Spectral') 
    sc = ax.scatter(wfga.x_positions/1000., wfga.y_positions/1000., s=0.5, 
                    c=wfga.z_positions, cmap=cm)
    
    # Plot substation
    ax.scatter(532359.5/1000., 5851358.4/1000., s=50, color='k', marker="*")
    
    # Plot Nature reserve Bruine Bank
    bruine_bank = np.array([[520457/1000.,5848100/1000.],[533881/1000.,5843854/1000.]])
    ax.plot(bruine_bank[:,0], bruine_bank[:,1], color='grey')
    ax.fill_between(bruine_bank[:,0], bruine_bank[:,1], adjusted_y_lim/1000., color='grey')
    
    # Set limits and labels
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.set_xlim([x_y_lim[0]/1000.,x_y_lim[1]/1000.])
    ax.set_ylim([adjusted_y_lim/1000.,x_y_lim[3]/1000.])
    
    # Plot colorbar on the right
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.set_title('        Water depth (m)', fontsize = 10)

    # Save or plot figure    
    if title is not None:
        plt.savefig(title)
    else:
        plt.show()
        
def box_plot_GY_comparison(objective):
    
    # Get data of sequential optimized layouts
    df_data = pd.read_csv("results/GY/Data_comparison.csv", header=0)
    df_data = df_data.loc[df_data['Joint/Sequential']=='Sequential']
    
    # Set labels x axis
    labels = ["FLORIS yaw", "GY De Jong", "GY Stanley", "No yaw"]
    
    data_aep, data_lcoe = [], []
    for yaw_opt in ["yaw_optimizer_floris", "geometric_yaw_Jong", "geometric_yaw_Stanley","None"]:
        
        # Get aep and lcoe values from dataframe
        if yaw_opt == "None":
            df3 = df_data.loc[(df_data['Yaw optimization type']!="yaw_optimizer_floris") &
                              (df_data['Yaw optimization type']!="geometric_yaw_Jong")   &
                              (df_data['Yaw optimization type']!="geometric_yaw_Stanley")]
        else:
            df3 = df_data.loc[df_data['Yaw optimization type']==yaw_opt]
        data_aep.append(df3['AEP'].values)
        data_lcoe.append(df3['LCOE'].values)
    
    # Generate figure
    fig, ax = plt.subplots(figsize=(10*0.7, 8*0.7))
    
    # Make box plot and print latex table
    if objective == 'AEP':
        latex_table_median_and_variance([data_aep], labels)
        bp = ax.boxplot(data_aep, patch_artist=True, labels=labels)  
    elif objective == 'LCOE':
        latex_table_median_and_variance([data_lcoe], labels)
        bp = ax.boxplot(data_lcoe, patch_artist=True, labels=labels) 
    
    # Set color of boxplot    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='indigo')
    for patch in bp['boxes']:
        patch.set_facecolor('mediumslateblue')
    
    # Set y label
    if objective == 'LCOE':
        plt.ylabel(r"LCOE $\left( \frac{€}{MWh} \right)$")
    elif objective == 'AEP':
        plt.ylabel(r"AEP $\left( GWh \right)$")
        
    # Save figure  
    plt.tight_layout()  
    plt.savefig('results/Figures/Comparison_GY_{}.png'.format(objective))
         
# #############################################
# Plots
# #############################################  

if plot_num == 1:
    box_plot(case, objective, num_runs_list, x_as_list, robust_cut, title)

if plot_num == 2:
    if case not in [2,4]:
        print("Wake plot only implemented for case 2 and 4.")
    else:
        wake_plot(case, parameters_list, x_y_lim, title, True,
              parameters1, parameters2, wind_rose, domain, floris_wind_farm)
    
if plot_num == 3:
    if case not in [2,4]:
        print("Power density plot only implemented for case 2 and 4.")
    else:
        power_density_plot(case, x_y_lim, parameters_list_PD, title)

if plot_num == 4:
    wind_roses(wind_rose, parameters2, domain, floris_wind_farm)
    
if plot_num == 5:
    if case == 1:
        wake_plot(case, parameters_list, x_y_lim, title, False,
                  parameters1, parameters2, wind_rose, domain, floris_wind_farm)
    else:    
        domain_plot(x_y_lim, parameters2, wind_rose, domain, floris_wind_farm,'results/Figures/IJmuiden_Ver_domain.png')
        
if plot_num == 6:
    box_plot_GY_comparison('AEP')