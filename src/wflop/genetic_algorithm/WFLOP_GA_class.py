""" 

Solving the wind farm layout optimization problem using a genetic algorithm.

"""

# Standard library imports
from    datetime                                                  import datetime
import  copy
import  math
import  os

# Third party imports
import  numpy                                                     as     np
from    floris.tools                                              import FlorisInterface

# Local application imports
from    wflop.yaw_optimizers.geometric_yaw_Stanley                import geometric_yaw_Stanley
from    wflop.yaw_optimizers.geometric_yaw_Jong                   import geometric_yaw_Jong
from    wflop.yaw_optimizers.yaw_optimizer_floris                 import yaw_optimizer_floris

################################################
# Wflop_ga class 
# Wind Farm Layout optimization using Genetic Algorithm
################################################

class Wflop_ga:
    
    def __init__(self, 
                 parameters           : dict                                ,
                 wind_rose            : np.ndarray                          ,
                 domain               : np.ndarray or float or int          ,
                 floris_wind_farm     : dict                                ,
                 substation           : list                        = [0,0] ,
                 robust               : int                         = 0     ,
                 random_seed          : int or None                 = None  ,
                 print_progress       : bool                        = False ):
        
        Wflop_ga.__check_parameters(parameters)
        
        # ==========================================
        # Properties
        # ==========================================
        
        self.__n_turbines                           = parameters["n_turbines"]
        self.__objective                            = parameters["objective"]
        self.__yaw_optimizer                        = parameters["yaw_optimizer"]
        self.__max_yaw                              = parameters["max_yaw"]
        
        self.__pop_size                             = parameters["pop_size"]          
        self.__parent_rate                          = parameters["parent_rate"]
        self.__n_parents                            = max(2, round(self.pop_size * self.parent_rate))
        self.__mutation_rate                          = parameters["mutation_rate"]
        self.__n_mutations                          = round(self.pop_size * self.mutation_rate)
        self.__worst_turbines_to_move_rate          = parameters["worst_turbines_to_move_rate"]
        if self.worst_turbines_to_move_rate == 0:
            self.__n_worst_turbines_to_move         = 0
        else:
            self.__n_worst_turbines_to_move         = max(1, round(self.n_turbines * self.worst_turbines_to_move_rate))
        self.__best_layouts_to_keep_rate            = parameters["best_layouts_to_keep_rate"]
        if self.best_layouts_to_keep_rate == 0:
            self.__n_best_layouts_to_keep           = 0
        else:
            self.__n_best_layouts_to_keep           = max(1,round(self.pop_size * self.best_layouts_to_keep_rate))
        self.__dynamic_mutation_step                = parameters["dynamic_mutation_step"]
        if self.__dynamic_mutation_step == 0:
            self.__dynamic_mutation_step            = self.max_generations + 1
        else:
            self.__dynamic_mutation_step            = round(self.__dynamic_mutation_step)
        self.__select_method                        = parameters["select_method"]
        self.__rng                                  = np.random.default_rng(random_seed)

        self.__n_wind_speed_bins                    = parameters["n_wind_speed_bins"]
        self.__n_wind_direction_bins                = parameters["n_wind_direction_bins"]
        self.__cut_in_speed                         = math.ceil(parameters["cut_in_speed"])
        self.__cut_out_speed                        = math.ceil(parameters["cut_out_speed"])
        self.__wind_rose                            = wind_rose
        wind_data                                   = Wflop_ga.__generate_wind_rose(self, self.__n_wind_direction_bins, self.__n_wind_speed_bins)
        self.__wind_speeds                          = wind_data[0]
        self.__wind_directions                      = wind_data[1] 
        self.__frequencies                          = wind_data[2].transpose()
        self.__wind_speeds_complete_wind_rose       = wind_data[3]
        self.__wind_directions_complete_wind_rose   = wind_data[4]
        self.__frequencies_complete_wind_rose       = wind_data[5].transpose()
        
        floris_wind_farm["wake"]["model_strings"]["velocity_model"] = parameters["velocity_wake_model"]
        floris_wind_farm["flow_field"]["wind_directions"]           = self.wind_directions 
        floris_wind_farm["flow_field"]["wind_speeds"]               = self.wind_speeds     
        self.__floris_wind_farm                                     = floris_wind_farm      
        self.__fi                                           	    = FlorisInterface(self.floris_wind_farm)
        
        self.__minimal_spacing                      = parameters["minimal_spacing"]*self.fi.floris.farm.rotor_diameters[0]
        self.__domain                               = domain
        if type(domain) == np.ndarray:
            domain_data                             = Wflop_ga.__generate_domain(self, domain)
        else:
            try:
                domain                              = float(domain)
                domain_data                             = Wflop_ga.__generate_simple_domain(self, domain)
            except TypeError:
                raise InputError("Domain should be of type np.ndarray, float or int.")
        self.__x_positions                          = domain_data[0]
        self.__y_positions                          = domain_data[1]
        self.__z_positions                          = domain_data[2]
        self.__initial_pop_positions_ind            = domain_data[3]
        self.__n_positions                          = len(self.x_positions)
        self.__positions                            = range(self.n_positions)
        self.__substation                           = substation
        
        fi_complete_wind_rose                       = FlorisInterface(self.floris_wind_farm)
        fi_complete_wind_rose.reinitialize(wind_directions = self.wind_directions_complete_wind_rose,\
                                           wind_speeds     = self.wind_speeds_complete_wind_rose     )
        self.__fi_complete_wind_rose                = fi_complete_wind_rose
        
        self.__rotation_matrix                      = Wflop_ga.__generate_rotation_matrix(self.wind_directions)
        
        self.__max_generations                       = parameters["max_generations"]
        self.__stagnation_generations_stop           = parameters["stagnation_generations_stop"]
        
        # ==========================================
        # Attributes
        # ==========================================
            
        self.pop_ind                                = np.zeros((self.pop_size, self.n_turbines), dtype=np.int32)
        self.parent_pop_ind                         = np.zeros((self.n_parents, self.n_turbines), dtype=np.int32)
        
        self.best_obj_gen                           = []
        self.best_layout_gen                        = []
        self.generation_time                        = []
        
        self.power_per_layout_windbin_turbine       = np.zeros((self.pop_size, self.n_wind_direction_bins, self.n_wind_speed_bins, self.n_turbines))
        
        self.best_layouts_to_keep                   = np.zeros((self.n_best_layouts_to_keep, self.n_turbines), dtype=np.int32)
        self.best_layouts_to_keep_fitness           = np.zeros(self.n_turbines, dtype=np.int32)
        
        if robust not in [0,1,2]:
            raise InputError("The robust level can be set to 0, 1 or 2.")
        self.robust                                 = robust
        self.print_progress                         = print_progress
            
        return
    
    # =============================== properties ===================================
    
    @property
    def n_turbines(self):
        return self.__n_turbines
    
    @property
    def objective(self):
        return self.__objective
    
    @objective.setter
    def objective(self,x):
        if x not in ["AEP", "LCOE"]:
            raise ParameterError("The objective can be AEP or LCOE.")
        else:
            self.__objective = x
    
    @property
    def yaw_optimizer(self):
        return self.__yaw_optimizer
    
    @yaw_optimizer.setter
    def yaw_optimizer(self,x):
        if x not in ["yaw_optimizer_floris", "geometric_yaw_Stanley", "geometric_yaw_Jong", "None"]:
            raise ParameterError("No yaw optimizer was recognized, please choose yaw_optimizer_floris, geometric_yaw_Stanley, geometric_yaw_Jong or None.")
        self.__yaw_optimizer = x
    
    @property
    def max_yaw(self):
        return self.__max_yaw
    
    @property
    def pop_size(self):
        return self.__pop_size

    @property
    def parent_rate(self):
        return self.__parent_rate
    
    @parent_rate.setter
    def parent_rate(self, x):
        if x<0 or x>1:
            raise ParameterError("The parent rate should be between 0 and 1.")
        self.__parent_rate = x
        self.__n_parents = max(2, round(self.pop_size * x))
        
    @property
    def n_parents(self):
        return self.__n_parents
    
    @property
    def mutation_rate(self):
        return self.__mutation_rate
    
    @mutation_rate.setter
    def mutation_rate(self, x):
        if x<0 or x>1:
            raise ParameterError("The mutation rate should be between 0 and 1.")
        self.__mutation_rate = x
        self.__n_mutations = round(self.pop_size * x)
    
    @property
    def n_mutations(self):
        return self.__n_mutations
    
    @property
    def worst_turbines_to_move_rate(self):
        return self.__worst_turbines_to_move_rate
    
    @worst_turbines_to_move_rate.setter
    def worst_turbines_to_move_rate(self,x):
        if x<0 or x>1:
            raise ParameterError("The worst turbines to move rate should be between 0 and 1.")
        self.__worst_turbines_to_move_rate = x
        if x == 0:
            self.__n_worst_turbines_to_move         = 0
        else:
            self.__n_worst_turbines_to_move         = max(1, round(self.n_turbines * x))
    
    @property
    def n_worst_turbines_to_move(self):
        return self.__n_worst_turbines_to_move
    
    @n_worst_turbines_to_move.setter
    def n_worst_turbines_to_move(self,x):
        if x<0 or x>self.n_turbines:
            raise ParameterError("The number of worst tubines to move should be between 0 and the number of turbines.")
        self.__n_worst_turbines_to_move = round(x)
    
    @property
    def best_layouts_to_keep_rate(self):
        return self.__best_layouts_to_keep_rate
    
    @best_layouts_to_keep_rate.setter
    def best_layouts_to_keep_rate(self,x):
        if x<0 or x>1:
            raise ParameterError("The best layouts to keep rate should be between 0 and 1.")
        self.__best_layouts_to_keep_rate = x
        if x == 0:
            self.__n_best_layouts_to_keep         = 0
        else:
            self.__n_best_layouts_to_keep         = max(1, round(self.pop_size * x))
    
    @property
    def n_best_layouts_to_keep(self):
        return self.__n_best_layouts_to_keep
    
    @property
    def dynamic_mutation_step(self):
        return self.__dynamic_mutation_step
    
    @dynamic_mutation_step.setter
    def dynamic_mutation_step(self,x):
        if x<0: 
            raise ParameterError("Dynamic mutation should be an integer of at least 0.")
        elif x==0:
            self.__dynamic_mutation_step = self.max_generations + 1
        else:
            self.__dynamic_mutation_step = round(x)
        
    @property
    def select_method(self):
        return self.__select_method
    
    @select_method.setter
    def select_method(self,x):
        if x not in ["elitist_random", "rank", "tournament"]:
            raise ParameterError("The select method should be elitist_random, rank or tournament.")
        self.__select_method = x
    
    @property
    def rng(self):
        return self.__rng
    
    @property
    def n_wind_speed_bins(self):
        return self.__n_wind_speed_bins
    
    @n_wind_speed_bins.setter
    def n_wind_speed_bins(self,x):
        if int(x)<=0:
            raise ParameterError("Number of wind speed bins should be greater than 0.")
        self.__n_wind_speed_bins                    = int(x)
        wind_data                                   = Wflop_ga.__generate_wind_rose(self, self.n_wind_direction_bins, int(x))
        self.__wind_speeds                          = wind_data[0]
        self.__wind_directions                      = wind_data[1] 
        self.__frequencies                          = wind_data[2].transpose()
        self.__fi.reinitialize(wind_directions = self.__wind_directions,\
                               wind_speeds     = self.__wind_speeds  )
        self.__rotation_matrix                      = Wflop_ga.__generate_rotation_matrix(self.__wind_directions)
        self.power_per_layout_windbin_turbine       = np.zeros((self.pop_size, self.n_wind_direction_bins, int(x), self.n_turbines))

    @property
    def n_wind_direction_bins(self):
        return self.__n_wind_direction_bins
    
    @n_wind_direction_bins.setter
    def n_wind_direction_bins(self,x):
        if int(x)<=0:
            raise ParameterError("Number of wind direction bins should be greater than 0.")
        self.__n_wind_direction_bins                = int(x)
        wind_data                                   = Wflop_ga.__generate_wind_rose(self, int(x), self.n_wind_speed_bins)
        self.__wind_speeds                          = wind_data[0]
        self.__wind_directions                      = wind_data[1] 
        self.__frequencies                          = wind_data[2].transpose()
        self.__fi.reinitialize(wind_directions = self.__wind_directions,\
                               wind_speeds     = self.__wind_speeds  )
        self.__rotation_matrix                      = Wflop_ga.__generate_rotation_matrix(self.__wind_directions)
        self.power_per_layout_windbin_turbine       = np.zeros((self.pop_size, int(x), self.n_wind_speed_bins, self.n_turbines))

    @property
    def cut_in_speed(self):
        return self.__cut_in_speed
    
    @property
    def cut_out_speed(self):
        return self.__cut_out_speed
    
    @property
    def wind_rose(self):
        return self.__wind_rose
    
    @property
    def wind_speeds(self):
        return self.__wind_speeds
    
    @property
    def wind_directions(self):
        return self.__wind_directions
    
    @property
    def frequencies(self):
        return self.__frequencies
    
    @property
    def wind_speeds_complete_wind_rose(self):
        return self.__wind_speeds_complete_wind_rose
    
    @property
    def wind_directions_complete_wind_rose(self):
        return self.__wind_directions_complete_wind_rose
    
    @property
    def frequencies_complete_wind_rose(self):
        return self.__frequencies_complete_wind_rose
    
    @property
    def floris_wind_farm(self):
        return self.__floris_wind_farm
    
    @property
    def fi(self):
        return self.__fi
    
    @property
    def minimal_spacing(self):
        return self.__minimal_spacing
    
    @property
    def domain(self):
        return self.__domain
    
    @domain.setter
    def domain(self,x):
        if type(x) == np.ndarray:
            domain_data                             = Wflop_ga.__generate_domain(self, x)
        else:
            try:
                x                                   = float(x)
                domain_data                         = Wflop_ga.__generate_simple_domain(self, x)
            except TypeError:
                raise InputError("Domain should be of type np.ndarray, float or int.")           
        domain_data                                 = Wflop_ga.__generate_simple_domain(self, x)
        self.__x_positions                          = domain_data[0]
        self.__y_positions                          = domain_data[1]
        self.__z_positions                          = domain_data[2]
        self.__initial_pop_positions_ind            = domain_data[3]
        self.__n_positions                          = len(self.x_positions)
        self.__positions                            = range(self.n_positions)

    @property
    def x_positions(self):
        return self.__x_positions
    
    @property
    def y_positions(self):
        return self.__y_positions
    
    @property
    def z_positions(self):
        return self.__z_positions
    
    @property
    def initial_pop_positions_ind(self):
        return self.__initial_pop_positions_ind
    
    @property
    def n_positions(self):
        return self.__n_positions
    
    @property
    def positions(self):
        return self.__positions
    
    @property
    def substation(self):
        return self.__substation
    
    @property
    def fi_complete_wind_rose(self):
        return self.__fi_complete_wind_rose
    
    @property
    def rotation_matrix(self):
        return self.__rotation_matrix
    
    @property
    def max_generations(self):
        return self.__max_generations
    
    @property
    def stagnation_generations_stop(self):
        return self.__stagnation_generations_stop
    
    # =============================== Hidden functions (only used for inizialization) ===================================
    
    @staticmethod
    def __check_parameters(parameters: dict):
        """ 
        Checks if the given parameters are valid and raises a ParameterError if this is not the case.

        Args:
            parameters (dict): A dictionary of parameters
        """ 
        if parameters["n_turbines"]<=0:
            raise ParameterError("Number of turbines should be greater than 0.")     
        if parameters["objective"] not in ["AEP", "LCOE"]:
            raise ParameterError("The objective can be AEP or LCOE.")  
        if parameters["yaw_optimizer"] not in ["yaw_optimizer_floris", "geometric_yaw_Stanley", "geometric_yaw_Jong", "None"]:
            raise ParameterError("No yaw optimizer was recognized, please choose yaw_optimizer_floris, geometric_yaw_Stanley, geometric_yaw_Jong or None.")
        if parameters["max_yaw"]<0 or parameters["max_yaw"]>45:
            raise ParameterError("The max yaw should be between 0 and 45 degrees.")
        if parameters["pop_size"]<2:
            raise ParameterError("Population size should be at least 2.")
        if parameters["parent_rate"]<0 or parameters["parent_rate"]>1:
            raise ParameterError("The parent rate should be between 0 and 1.")
        if parameters["mutation_rate"]<0 or parameters["mutation_rate"]>1:
            raise ParameterError("The mutation rate should be between 0 and 1.")
        if parameters["worst_turbines_to_move_rate"]<0 or parameters["worst_turbines_to_move_rate"]>1:
            raise ParameterError("The worst turbines to move rate schould be between 0 and 1.")
        if parameters["best_layouts_to_keep_rate"]<0 or parameters["best_layouts_to_keep_rate"]>1:
            raise ParameterError("The best layouts to keep rate schould be between 0 and 1.")
        if parameters["dynamic_mutation_step"]<0:
            raise ParameterError("Dynamic mutation should be an integer of at least 0.")
        if parameters["select_method"] not in ["elitist_random", "rank", "tournament"]:
            raise ParameterError("The select method should be elitist_random, rank or tournament.")
        if parameters["n_wind_speed_bins"]<=0:
            raise ParameterError("Number of wind speed bins should be greater than 0.")
        if parameters["n_wind_direction_bins"]<=0:
            raise ParameterError("Number of wind direction bins should be greater than 0.")
        if parameters["cut_in_speed"]<0:
            raise ParameterError("Cut in speed should be greater or equal than 0.")
        if parameters["cut_out_speed"]<=parameters["cut_in_speed"]:
            raise ParameterError("Cut out speed should be greater than the cut in speed")
        if parameters["minimal_spacing"]<1:
            raise ParameterError("The minimal spacing beteween turbines should be greater or equal than 1.")
        if parameters["velocity_wake_model"] not in ["jensen", "gauss"]:
            raise ParameterError("The velocity wake model should be jensen or gauss.")
        if parameters["stagnation_generations_stop"]<0:
            raise ParameterError("The stagnation generations stop should be greater or equal than 0.")
        return
    
    def __generate_wind_rose(self, n_wind_direction_bins: int, n_wind_speed_bins: int)->list:
        """ Generates the wind speeds, wind directions and frequencies of the chosen wind rose bins and of the complete wind rose bins.
        The chosen wind rose contains a number of wind speed and direction bins as specified in the given parameters.
        The complete wind rose containts a wind speed bin for every 1 m/s and wind direction bin for every 1 degree.

        Args:
            n_wind_direction_bins (int): Number of wind direction bins
            n_wind_speed_bins (int): Number of wind speed bins

        Returns:
            list: [wind_speeds, wind_directions, frequencies,\
                   wind_speeds_complete_wind_rose, wind_directions_complete_wind_rose, frequencies_complete_wind_rose]
        """
        
        # Wind_rose has the columns wind speed, wind direction and frequencie
        # Get only the wind speeds for which the farm needs to be optimized --> [cut in speed, cut out speed]
        wr = self.wind_rose[(self.wind_rose[:,0]>=self.cut_in_speed ) & (self.wind_rose[:,0]<self.cut_out_speed)]
        
        if len(np.unique(wr[:,0])) < n_wind_speed_bins:
            raise ParameterError("The number of wind speed bins should be smaller or equeal to the number of wind speeds in the wind rose.")
        if len(np.unique(wr[:,1])) < n_wind_direction_bins:
            raise ParameterError("The number of wind direction bins should be smaller or equeal to the number of wind directions in the wind rose.")

        # Devide the wind rose over the given number of bins --> n_wind_speed_bins and n_wind_direction_bins
        ws_edges = np.arange(self.cut_in_speed-0.5,self.cut_out_speed+1.5, ((self.cut_out_speed+1-self.cut_in_speed)/n_wind_speed_bins))
        wd_edges = np.arange(-0.5,360.5, (360/n_wind_direction_bins))
        (frequencies, ws_bins, wd_bins) = np.histogram2d(wr[:,0],\
                                                        wr[:,1],\
                                                        bins=[ws_edges, wd_edges],\
                                                        weights=wr[:,2])
        # Get the wind speed and directin of the middle of each bin
        wind_speeds     = 0.5*(ws_bins[0:-1]+ws_bins[1:])
        wind_directions = 0.5*(wd_bins[0:-1]+wd_bins[1:])
        
        # If number of wind bins is 1, then value is set to default value
        if n_wind_speed_bins == 1:
            wind_speeds = [8.0]
        if n_wind_direction_bins == 1:
            wind_directions = [270.0]
        
        # Devide the wind rose over wind bins of 1 m/s of wind speeds and 1 degree of wind direction
        ws_edges = np.arange(self.cut_in_speed-0.5,self.cut_out_speed+1.5)
        wd_edges = np.arange(-0.5,360.5)  
        (frequencies_complete_wind_rose, ws_bins, wd_bins) = np.histogram2d(wr[:,0],\
                                                                           wr[:,1],\
                                                                           bins=[ws_edges, wd_edges],\
                                                                           weights=wr[:,2])
        wind_speeds_complete_wind_rose     = 0.5*(ws_bins[0:-1]+ws_bins[1:])
        wind_directions_complete_wind_rose = 0.5*(wd_bins[0:-1]+wd_bins[1:])
        
        return [wind_speeds,\
                wind_directions,\
                frequencies,\
                wind_speeds_complete_wind_rose,\
                wind_directions_complete_wind_rose,\
                frequencies_complete_wind_rose]
    
    def __generate_domain(self, domain: np.ndarray)->list:
        """
        Generates the domain information consisting of the x coordinates, y coordinates and depth.
        Also selects indices of a grid adhearing to the minimum spacing in order to later generate the initial population.

        Args:
            domain (np.ndarray): Numpy array of the domain containing the (x,y,z) coordinates
            
        Returns:
            list: [x_positions, y_positions, z_positions, initial_pop_positions_ind]
        """   
        
        # Split the domain information in x, y and z    
        x_positions            = domain[:,0]
        y_positions            = domain[:,1]
        z_positions            = domain[:,2]
                    
        # Find the smallest step size between two points where the minimal spacing is respected
        x_bin_size = math.gcd(*x_positions.astype(int))
        x_bin_size_spaced = math.ceil(self.minimal_spacing/x_bin_size)*x_bin_size
        y_bin_size = math.gcd(*y_positions.astype(int))
        y_bin_size_spaced = math.ceil(self.minimal_spacing/y_bin_size)*y_bin_size
        
        # Create domain where all positions comply with the minimal spacing
        initial_pop_positions_ind = [ind for ind in range(len(domain)) if \
                                     domain[ind,0] % x_bin_size_spaced == 0 and \
                                     domain[ind,1] % y_bin_size_spaced == 0]
        
        if math.comb(len(initial_pop_positions_ind), self.n_turbines) < self.pop_size:
            raise ParameterError("A grid complying with the minimal spacing and where the postions are a subset of the given domain coordinates does not contain enough positions to place the turbines.\
                Make the minimal spacing, the number of turbines or the population size smaller. Changing the given domain can also be a solution, but is less straigt forward.")
        
        return [x_positions,\
                y_positions,\
                z_positions,
                initial_pop_positions_ind]
        
    def __generate_simple_domain(self, side_length: float)->list:
        """
        Produces a square with the given side lengths as domain
        where the distance between points is around a tenth of the minimal spacing distance 
        and the water depth is 20m everywhere.
        Also selects indices of a grid adhearing to the minimum spacing in order to later generate the initial population.

        Args:
            side_length (float): side length of the square domain in meters

        Returns:
            list: [x_positions, y_positions, z_positions, initial_pop_positions_ind]
        """

        n = 10*math.floor(side_length/self.minimal_spacing)
        x = np.linspace(0,side_length,n+1) 
        y = np.linspace(0,side_length,n+1) 
        xv, yv = np.meshgrid(x,y)
         
        x_positions                 = xv.flatten()
        y_positions                 = yv.flatten()
        z_positions                 = 20*np.ones(len(x_positions))
        initial_pop_positions_ind   = np.array([list(range(i*(n+1),(i+1)*(n+1),10)) for i in range(0,n+1,10)]).flatten()
        
        return [x_positions,\
                y_positions,\
                z_positions,
                initial_pop_positions_ind]
    
    @staticmethod    
    def __generate_rotation_matrix(n_wind_directions: int)->np.ndarray:
        """
        Generates the rotation matrix that can be used ot rotate the layout such that the wind comes from the west. 

        Args:
            n_wind_directions (int): Number of wind directions

        Returns:
            np.ndarray: Matrix containing the 2x2 rotation matrix for each wind direction
        """
        
        R = np.zeros((len(n_wind_directions)*2, 2))
        for i in range(len(n_wind_directions)):      
            theta = -(n_wind_directions[i] - 270.)*(np.pi/180.) # 270 degrees is wind from the west
            R[2*i:2*(i+1),:] = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        
        return R
    
    # ========================== Power computation ============================  
        
    def turbine_powers(self, farm: FlorisInterface, layout_x: np.ndarray, layout_y: np.ndarray, i: int = None)->np.ndarray:
        """ 
        Computes the turbine powers using wake calculations in FLORIS. 
        If a yaw optimizer is given in the parameters, the optimal yaw angles are calculated and used for the power computations.

        Args:
            farm (FlorisInterface): FLORIS wind farm
            layout_x (np.ndarray): x coordinates of the wind farm
            layout_y (np.ndarray): y coordinates of the wind farm

        Returns:
            np.ndarray: An array conaining the power of each turbine given a wind direction and speed.
                        The size of the array is the number of wind directions times the number of 
                        wind speeds times the number of turbines.
        """        
       
        farm.reinitialize(layout_x=layout_x, 
                          layout_y=layout_y)
        
        # Get the yaw angles
        if self.yaw_optimizer == "yaw_optimizer_floris":  
            yaw_angles = yaw_optimizer_floris(farm, self.max_yaw)
        elif self.yaw_optimizer == "geometric_yaw_Stanley":  
            yaw_angles = geometric_yaw_Stanley(farm, max_yaw=self.max_yaw)
        elif self.yaw_optimizer == "geometric_yaw_Jong":  
            yaw_angles = geometric_yaw_Jong(farm, self.rotation_matrix, self.max_yaw) 
        else:
            yaw_angles = np.zeros((farm.floris.flow_field.n_wind_directions, farm.floris.flow_field.n_wind_speeds, len(farm.floris.farm.layout_x)))
        
        # Get powers of each turbine for each wind bin with the given yaw angles
        farm.calculate_wake(yaw_angles=yaw_angles)
        turbine_powers = farm.get_turbine_powers() #size num_wind_dir x num_wind_speeds x num_of_turbines        
        
        # Save results if i is indicated
        if type(i) is int:
            #pop_size x n_wind_direction_bins x n_wind_speed_bins x n_turbines
            self.power_per_layout_windbin_turbine[i] = turbine_powers
            
        return turbine_powers     
    
    def lcoe(self, power_per_layout_turbine: np.array,
             discount_rate = 0.06,
             lifetime = 27,
             buildtime = 5,
             decommissiontime = 1,
             turbine_size = 15,
             independend_building_costs = 2183000*1.15,
             turbine_building_costs = 100000,
             cable_building_costs = 25,
             decommissioning_costs = 330000*1.15,
             maintenance_per_year_costs = 75000*1.15):
        """
        Compute the LCOE of each wind farm in the population.

        Args:
            power_per_layout_turbine (np.ndarray): Power per turbine for each layout in the population (size = pop_size x n_turbines)
            
        Returns:
            np.ndarray: The lcoe per layout in the population per turbine in euro/MWh
            np.ndarray: The lcoe per layout in the population in euro/MWh
        """
        ##########################
        # Numbers based on https://guidetoanoffshorewindfarm.com/wind-farm-costs:
        # Cost of tower under the tansition piece missing (100.000 + 20.000 + 10.000 != 280.000), so taken as 150.000
        # Cost of tower under the tansition piece, array cable and cable protection substracted
        # Pricing from the source are in brittish pounds, so converted to euro by multiplying with 1.15
        # independend_building_costs = 120.000 + 1.000.000 + (600.000 - 37.000 - 150.000) + 650.000 £/MW
        # maintenance_per_year = 75.000 £/MW/year
	    # decommissioning = 330.000 £/MW
        #
        # Numbers based on experts in the field:
        # turbine_building_costs = 100.000 euro per meter water depth per 15MW tubine
        # cable_building_costs = 25 euro per meter distance to the substation per 15MW tubine
        ###########################
        
        if turbine_size != 15:
            print("Note that the lcoe function is made for the 15MW turbine. Number may not be realistic for other turbine types.")
        
        # pop_size x num_of_turbines
        turbine_costs = turbine_building_costs*abs(self.z_positions[self.pop_ind]) #euro/turbine
        cable_costs = cable_building_costs*np.sqrt((self.x_positions[self.pop_ind] - self.substation[0])**2 + (self.y_positions[self.pop_ind] - self.substation[1])**2) #euro/turbine
        building_costs = (turbine_size*independend_building_costs + turbine_costs + cable_costs )*np.ones((self.pop_size, self.n_turbines)) #euro/turbine
        decommission_costs =  turbine_size*decommissioning_costs*np.ones((self.pop_size, self.n_turbines)) #euro/turbine
        maintenance_costs =  turbine_size*maintenance_per_year_costs*np.ones((self.pop_size, self.n_turbines)) #euro/turbine/year
        
        # project_time x pop_size x num_of_turbines
        I = np.zeros((buildtime + lifetime + decommissiontime, self.pop_size, self.n_turbines))
        M = np.zeros((buildtime + lifetime + decommissiontime, self.pop_size, self.n_turbines))
        E = np.zeros((buildtime + lifetime + decommissiontime, self.pop_size, self.n_turbines))
        for i in np.arange(0,buildtime):
            I[i,:,:] = ((building_costs/buildtime)/((1+discount_rate)**i))
        for i in np.arange(buildtime+lifetime,buildtime+lifetime+decommissiontime):
            I[i,:,:] = ((decommission_costs/decommissiontime)/((1+discount_rate)**i)) 
        for i in np.arange(buildtime,buildtime+lifetime):
            M[i,:,:] = (maintenance_costs/((1+discount_rate)**i))  
        C = I+M #euro/turbine/year
        for i in np.arange(buildtime, buildtime+lifetime):
            # power_per_layout_turbine is in W/turbine, convert to power production in MWh/year/turbine
            E[i,:,:] = (power_per_layout_turbine*24*365/1e6)/((1+discount_rate)**i) #MWh/year/turbine

        C_per_pop_per_turbine = np.sum(C, axis=0) #euro/turbine
        C_per_pop = np.sum(C_per_pop_per_turbine, axis=-1) #euro

        E_per_pop_per_turbine = np.sum(E, axis=0) #MWh/turbine
        E_per_pop = np.sum(E_per_pop_per_turbine, axis=-1) #MWh

        lcoe_per_pop = C_per_pop/E_per_pop #euro/MWh
        lcoe_per_pop_per_turbine = C_per_pop_per_turbine/E_per_pop_per_turbine
        
        return -lcoe_per_pop_per_turbine, -lcoe_per_pop
         
    def fitness(self):
        """ 
        Computes the objective of each layout in the population and orders pop_ind.
        Pop_ind is ordered from best to worst layout and from worst to best turbine.
        The best found objective and corresponding layout of this generation are saved to self.best_obj_gen and self.best_layout_gen.
        The time needed for the wake calculations saved to self.best_obj_gen.
        """        
        
        # Compute turbine powers, results are saved to power_per_turbine_and_wind_bin 
        # of size pop_size x num_wind_dir x num_wind_speeds x num_of_turbines        
        for i in range(self.pop_size):
            self.turbine_powers(farm   = copy.deepcopy(self.fi),
                                layout_x = self.x_positions[self.pop_ind[i]], 
                                layout_y = self.y_positions[self.pop_ind[i]],
                                i = i)
        
        # axis are switched, power_per_layout_turbine_windbin is of size pop_size x num_of_turbines x num_wind_dir x num_wind_speeds
        power_per_layout_turbine_windbin = np.moveaxis(self.power_per_layout_windbin_turbine,3,1) 
        # scale power_per_layout_turbine_windbin with the frequency of each wind bin
        power_per_layout_turbine_windbin_scaled = np.multiply(power_per_layout_turbine_windbin, self.frequencies)
        # sum all the wind bins together to get the average power per turbine per layout
        power_per_layout_turbine = np.sum(power_per_layout_turbine_windbin_scaled, axis=(-1,-2)) #pop_size x num_of_turbines
        
        # Compute fitness value and sort each layout
        if self.objective == "AEP":
            # Every layout in pop_ind is ordered from worst to best turbine.
            self.pop_ind         = np.take_along_axis(self.pop_ind, np.argsort(power_per_layout_turbine, axis=-1), axis=1)
            # The fitness is the power produced in a yaer in GWh for each layout.
            fitness              = np.sum(power_per_layout_turbine, axis=1)*24*365/1e9
        elif self.objective == "LCOE":
            lcoe_per_layout_turbine, lcoe_per_layout = self.lcoe(power_per_layout_turbine)
            # Every layout in pop_ind is ordered from worst to best turbine.
            self.pop_ind         = np.take_along_axis(self.pop_ind, np.argsort(lcoe_per_layout_turbine, axis=-1), axis=1)
            # The fitness is the average power produced for each layout.
            fitness              = lcoe_per_layout
        
        # Pop_ind is ordered from best to worst layout
        sorted_index         = np.argsort(-fitness)
        self.pop_ind         = self.pop_ind[sorted_index, :]
        fitness              = fitness[sorted_index] 
        
        # Replace worst n layouts with best n layout last generation
        if len(self.best_obj_gen)>0:            
            self.pop_ind[-self.n_best_layouts_to_keep:,:]           = self.best_layouts_to_keep
            fitness[-self.n_best_layouts_to_keep:]      	        = self.best_layouts_to_keep_fitness
            
            # Pop_ind is ordered from best to worst layout
            sorted_index         = np.argsort(-fitness)
            self.pop_ind         = self.pop_ind[sorted_index, :] 
            fitness              = fitness[sorted_index] 
        
        # Save best objective and corresponding layout of this generation
        self.best_obj_gen.append(fitness[0])
        self.best_layout_gen.append(np.sort(self.pop_ind[0, :]))
        
        # Save best n layouts to keep and their corresponding fitness value
        self.best_layouts_to_keep         = copy.deepcopy(self.pop_ind[:self.n_best_layouts_to_keep, :])
        self.best_layouts_to_keep_fitness = copy.deepcopy(fitness[:self.n_best_layouts_to_keep])
        
        return
    
    def robust_objective(self, 
                         layout: np.ndarray, 
                         optimizer: str = "yaw_optimizer_floris", 
                         complete_wind_rose: bool = True)->float:
        """
        Computes the aep, lcoe and power per turbine per windbin of the given layout
        using fully optimized yaw angles and the most complete wind rose unless specified differently.

        Args:
            layout (np.ndarray): An array containing the indices of the layout. It has the number of turbines as length.
            optimizer (str, optional): Optimizer to be used for the yaw angles. Defaults to "yaw_optimizer_floris".
            complete_wind_rose (bool, optional): Uses simplified wind rose if False and complete wind rose if True. Defaults to True.

        Returns:
            float: The aep in GWh
            float: The lcoe in euro/MWh
            float: The power per turbine per windbin in W
        """
        # Set yaw optimizer and wind rose. 
        self.yaw_optimizer = optimizer
        if complete_wind_rose:
            farm     = self.fi_complete_wind_rose
            freq     = self.frequencies_complete_wind_rose
        else:
            farm     = self.fi
            freq     = self.frequencies
        
        # Compute turbine powers and returns power_per_turbine_and_wind_bin of size num_wind_dir x num_wind_speeds x num_of_turbines
        power_per_windbin_turbine = self.turbine_powers(farm     = farm,
                                                        layout_x = self.x_positions[layout], 
                                                        layout_y = self.y_positions[layout])                                               

        # axis are switched, power_per_turbine_windbin is of size num_of_turbines x num_wind_dir x num_wind_speeds
        power_per_turbine_windbin = np.moveaxis(power_per_windbin_turbine,2,0) 
        # scale power_per_turbine_windbin with the frequency of each wind bin
        power_per_turbine_windbin_scaled = np.multiply(power_per_turbine_windbin, freq)
        # sum all the wind bins together to get the average power per turbine per layout
        power_per_turbine = np.sum(power_per_turbine_windbin_scaled, axis=(-1,-2)) #pop_size x num_of_turbines
        
        # Compute th eaep and lcoe
        aep                                      = np.sum(power_per_turbine, axis=-1)*24*365/1e9
        lcoe_per_layout_turbine, lcoe_per_layout = self.lcoe(np.resize(power_per_turbine,(1, self.n_turbines)))
        lcoe                                     = lcoe_per_layout[0]
        
        return aep, lcoe, power_per_turbine_windbin

    # ==================== Genetic algorithm ============================== 

    def genetic_alg(self, results_file: str = None,
                          initial_pop: np.ndarray = None):
        """
        Goes trough the genetic algorithm steps untill there is no improvement 
        for self.stagnation_generations_stop number of generations
        or if self.max_generations number of generations is reached.

        Args:
            results_file (str, optional): Directory of the file in which the results can be saved.\
                                          Defaults to None, in which case the results are not saved.
            initial_pop (np.ndarray, optional): Initial population.\
                                          Defaults to None, in which case a random population is used.
        """
        
        # Initialize the GA
        self.pop_ind                                = np.zeros((self.pop_size, self.n_turbines), dtype=np.int32)
        self.parent_pop_ind                         = np.zeros((self.n_parents, self.n_turbines), dtype=np.int32)
        self.best_obj_gen                           = []
        self.best_layout_gen                        = []
        self.generation_time                        = []
        self.power_per_layout_windbin_turbine       = np.zeros((self.pop_size, self.n_wind_direction_bins, self.n_wind_speed_bins, self.n_turbines))
        GA_start_time                               = datetime.now()
        
        # Generate the initial population and compute their fitness        
        if type(initial_pop) is not np.ndarray:
            self.pop_ind = self.generate_random_pop()  
        else:
            self.pop_ind = initial_pop
        self.fitness()
        self.generation_time.append((datetime.now()-GA_start_time).total_seconds())
        self.save_results(results_file)
        
        # Start the genertic algorithm
        no_improvement = 0
        for generation in range(1,self.max_generations+1):
            try:
                t = datetime.now()
                
                # Go through the steps of the genetic algorithm
                self.select()
                self.move_worst()
                self.crossover()
                self.mutation()
                self.fitness()
                
                # Save results and generation time
                self.generation_time.append((datetime.now()-t).total_seconds())
                self.save_results(results_file)
                
                # Stop the genetic algorithm when there is no improvement between generations for three generations in a row
                if self.best_obj_gen[-1] <= np.max(self.best_obj_gen[:-1]):
                    no_improvement += 1
                    if no_improvement == self.stagnation_generations_stop:
                        break
                else:
                    no_improvement = 0
                
                # Adjust the mutation rate or worst turbines to move if dynamic mutation is used
                if (generation % self.dynamic_mutation_step) == 0:
                    if self.mutation_rate < 0.11:
                        self.n_worst_turbines_to_move = np.max([0,round(self.n_worst_turbines_to_move - 1)])
                    self.mutation_rate = np.max([0.1,self.mutation_rate-0.1])
                
                # Check if all layouts in the population adhear to the minimum spacing    
                if np.min(self.distances_between_turbines(self.pop_ind))<self.minimal_spacing:
                    raise ImplementationError("The minimum distance between turbines is smaller than the minimal spacing.")
                
            except ValueError:
                print("A ValueError has occured. This iteration is skipped.")
                    
        # Take the best found layouta and compute its objective
        self.best_layout_gen.append(self.best_layout_gen[np.argmax(self.best_obj_gen)]) 
        if self.robust == 0 and self.yaw_optimizer == "None":
            aep, lcoe, power_per_turbine_windbin = self.robust_objective(layout = self.best_layout_gen[-1], 
                                                optimizer = "geometric_yaw_Jong", 
                                                complete_wind_rose = False)[0]
        elif self.robust == 0:
            objective = self.best_obj_gen[np.argmax(self.best_obj_gen)]
            aep = objective
            lcoe = objective
        elif self.robust == 1:
            aep, lcoe, power_per_turbine_windbin = self.robust_objective(self.best_layout_gen[-1], 
                                                optimizer = "yaw_optimizer_floris", 
                                                complete_wind_rose = False)[0]
        elif self.robust == 2:
            aep, lcoe, power_per_turbine_windbine = self.robust_objective(self.best_layout_gen[-1], 
                                                optimizer = "yaw_optimizer_floris", 
                                                complete_wind_rose = True)[0]
        # Save the results    
        if self.objective == 'AEP':
            self.best_obj_gen.append(aep)
        elif self.objective == 'LCOE':
            self.best_obj_gen.append(lcoe)
        self.generation_time.append((datetime.now()-GA_start_time).total_seconds())
        self.save_results(results_file)
        
        return
    
    def generate_random_pop(self):
        """
        Generates a random population.
        
        Returns:
            np.ndarray: A numpy array of size population size times the number of turbines containing the indices of the positions.
        """
                
        for i in range(10):
            # Make a numpy.ndarray with pop_size number of rows each containing n_positions number of elements
            # of which n_turbines elements are ones and the rest are zeros.
            empty_positions = np.zeros((self.pop_size,len(self.initial_pop_positions_ind)-self.n_turbines))
            turbine_positions = np.ones((self.pop_size,self.n_turbines))
            positions_bin = np.concatenate((empty_positions, turbine_positions), axis=1)
            
            # Shuffle each rows independently
            positions_bin_shuffled = self.rng.permuted(positions_bin, axis=1)
            # Get the indexes of each turbine in the course domain (with minimal spacing distance between each position)
            positions_ind_course = np.nonzero(positions_bin_shuffled)
            # Get the indexes of each turbine in the original domain
            positions_ind_fine = np.repeat([self.initial_pop_positions_ind], self.pop_size, axis=0)[positions_ind_course]   
            
            # Reshape the indexes to an nd.array of size population size times number of turbines.
            pop_ind = positions_ind_fine.reshape((self.pop_size,self.n_turbines))
            
            if i != 0:
                pop_ind = np.concatenate((pop_ind, pop_ind_unique), axis=0)

            # Check if all the randomly found layouts are unique
            pop_ind_unique = np.unique(pop_ind, axis=0)
            if len(pop_ind_unique) == self.pop_size:
                return pop_ind   
        
        raise ParameterError("The algorithm was not able to find a population size number of unique layouts\
                             on a grid complying with the minimal spacing and where the postions are a devider of the set positions bin size.\
                             Make the positions bin size, the minimal spacing or the number of turbines smaller.")
    
    def select(self):      
        """ 
        Selects parents from the population.
        """
        # Note that the pop_ind is ordered in the function fitness from best to worst layout and from worst to best turbine.
        
        if self.select_method == "elitist_random":
            
            # Add elite parents to parent population
            self.parent_pop_ind[:0.5*self.n_parents] = self.pop_ind[:0.5*self.n_parents]
            
            # Add random parents to parent population
            self.parent_pop_ind[0.5*self.n_parents:]  = self.rng.choice(self.pop_ind[0.5*self.n_parents:],
                                                                    size    = 0.5*self.n_parents,
                                                                    replace = False)
            
        if self.select_method == "rank":
            
            # Probabilities go lineair to zero, summing up to 1.
            probabilities = (self.pop_size - np.arange(self.pop_size)) / (0.5*self.pop_size*(self.pop_size+1))
            
            # Parents are choosen randomly with the given distribution of probabilities.
            self.parent_pop_ind  = self.rng.choice(self.pop_ind,
                                                   size    = self.n_parents,
                                                   replace = False,
                                                   p       = probabilities )
            
        elif self.select_method == "tournament":
            
            # Get the index number to each individual in the population
            pop_numbers = self.rng.permutation(self.pop_size)
            # Fill up the index numbers such that it can be reshapen into n_parents times something.
            fill_up_numbers = np.arange(self.pop_size, self.n_parents*math.ceil(self.pop_size/self.n_parents))
            numbers = np.concatenate((pop_numbers, fill_up_numbers), axis=None)
            # Reshape the index number into an array of n_parents rows. Each row represents an tournament
            tournament_numbers = numbers.reshape((self.n_parents, -1), order="F")
            # Get the minimum value of each row, this is the winner of the tournament.
            parents = np.min(tournament_numbers, axis=1)
            # Assign the chosen parents to the parent population
            self.parent_pop_ind = self.pop_ind[parents,:]
        
        return
    
    def distances_between_turbines(self, pop_ind: np.ndarray):
        """
        Computes the distances between each combination of turbines for each layout in the population.

        Args:
            pop_ind (np.ndarray): Population

        Returns:
            np.ndarray: Distances
        """
        
        if len(np.shape(pop_ind)) == 1:
            pop_ind = np.array([pop_ind])
       
        # Get x position of each layout in the population
        layout_x = np.array(self.x_positions[pop_ind])
        A = np.repeat(layout_x, self.n_turbines, axis=1)
        B = np.repeat(layout_x, self.n_turbines, axis=0).reshape(len(pop_ind),self.n_turbines**2)
        # dx has a row for each layout in the population with on this row the dx of each combination:
        # [x_1-x_1, x_1-x_2, x_1-x_3, ..., x_2-x_1, x_2-x_2, x_2-x_3, ..., x_n-x_n]
        dx = np.subtract(A, B)

        # Do the same to get dy
        layout_y = np.array(self.y_positions[pop_ind])
        A = np.repeat(layout_y, self.n_turbines, axis=1)
        B = np.repeat(layout_y, self.n_turbines, axis=0).reshape(len(pop_ind),self.n_turbines**2)
        dy = np.subtract(A, B)

        # Get the Euclidian distance between each combination of positions
        distance = np.sqrt(np.square(dx) + np.square(dy))
        # Remove the zeros, these are not relevant since they are the distance of a point to itself
        distance = distance[np.nonzero(distance)].reshape(len(pop_ind),self.n_turbines**2-self.n_turbines)
        
        # Return array of size n_turbines if input is one layout and pop_size x n_turbines otherwise.
        if len(np.shape(pop_ind)) == 1:
            return distance[0]
        else:
            return distance
    
    def move_worst(self):
        """ 
        Move worst n turibnes of each layout in the parent population to a random empty spots.
        """
        #Note that the pop_ind is ordered in the function fitness from best to worst layout and from worst to best turbine and this order is kept in parent_pop_ind.
        
        # Go through each parent layout
        for i in range(len(self.parent_pop_ind)):
            # Select a layout in the parent population
            layout = copy.deepcopy(self.parent_pop_ind[i,:])   
            # Select the empty spots
            empty_spots = np.setdiff1d(self.positions, layout)
            
            for attempts in range(10):
                
                # Move the worst n turbine to the randomly chosen empty spots
                layout[0:self.n_worst_turbines_to_move] = self.rng.choice(empty_spots, size=self.n_worst_turbines_to_move)  
                # Check if the new layout complies with the minimal spacing
                if np.min(self.distances_between_turbines(layout))>=self.minimal_spacing:
                    self.parent_pop_ind[i, :] = layout
                    break
                # If this is not the case no changes are implemented and a new atempt is made (max 10 times).
                
        return

    def crossover(self):
        """ 
        Generates new population based on the selected parents.
        For each child a crossover is made between two randomly selected distinct parents.
        If both parents have a turbine or not a turbine on the same spot, this is transfered to the child.
        The other turbine positions are randomly selected from the parents.
        The number of turbines stays the same.
        """
        
        children_made = 0
        attempts = 0
        
        # Make a pop_size amount of children to form the new population.
        while children_made < self.pop_size:

            if attempts == 10:
                # For the remaining layouts in the population the previous generation is used.
                break
            
            # Select two random parents out of the parent population
            parents              = self.rng.choice(self.parent_pop_ind, size = 2, replace = False)
            # Find common turbines
            set_turbines         = np.intersect1d(parents[0], parents[1], assume_unique=True)
            # Find different turbine
            choice_turbines      = np.setxor1d(parents[0], parents[1], assume_unique=True)
            # Randomly chose enough turbines out of the different turbines
            chosen_turbines      = self.rng.choice(choice_turbines, 
                                                   size    = self.n_turbines-len(set_turbines), 
                                                   replace = False)
            # Combine the common and randomly chosen turbines to a new layout.
            layout  = np.concatenate((set_turbines, chosen_turbines))
            
            # Check if the new layout complies with the minimal spacing and if the new layout is not already in the next generation
            if np.min(self.distances_between_turbines(layout))>=self.minimal_spacing and \
                layout not in self.pop_ind[:children_made, :]:
                # Add the the new layout to the next generation
                self.pop_ind[children_made, :] = layout
                children_made = children_made + 1
                attempts = 0
            else:
                attempts = attempts + 1
                
        return
    
    def mutation(self):
        """ 
        Selects some mutating layouts based on the mutation rate.
        Moves one random turbine to a random empty spot in each of these layouts.
        """
        
        # Shuffle the population
        self.rng.shuffle(self.pop_ind)
        
        mutations = 0
        # Go randomly through the population
        for i in range(len(self.pop_ind)):
            
            # Select a layout in the population to be mutated
            layout                   = copy.deepcopy(self.pop_ind[i,:])
            # Randomly chose a turbine in the mutating layout
            turbine                  = self.rng.choice(range(self.n_turbines))
            # Randomly chose an empty spot in the mutating layout
            empty_spots              = np.setdiff1d(self.positions, layout)
            # Move the chosen turbine to the chosen empty spot
            layout[turbine]          = self.rng.choice(empty_spots)

            # Check if the mutated layout complies with the minimal spacing
            if np.min(self.distances_between_turbines(layout))>=self.minimal_spacing and \
                layout not in self.pop_ind:
                self.pop_ind[i,:] = layout 
                mutations += 1
            
            # Stop if enough individuals are mutated according to the mutation rate. 
            # If this is not the case when tried once on each individual, less mutations are applied.
            if mutations == self.n_mutations+1:
                break
                    
        return        
    
    # ==================== Save and show results ==============================
    
    def save_results(self, results_file: str = None):
        """
        Saves the results as a new line to the results file.
        The results which are saved are the best objective value, corresponding layout and generation time.

        Args:
            results_file (str): Directory of the file in which the results can be saved.\
                                Defaults to None, in which case the results are not saved.
        """     
        
        # Generate string to be added to the result file   
        obj = self.best_obj_gen[-1]        
        results = [str(obj),\
                   str(self.best_layout_gen[-1]),\
                   str(self.generation_time[-1])]
        result_string = ','.join(results).replace('\n', '')

        # Add string to the result file
        if results_file is not None:
            if os.path.exists(results_file):
                f = open(results_file, 'r') 
                content = f.read()
                f.close()
            else:
                content = ''
            f = open(results_file, 'w')
            f.write(content) 
            f.write(result_string + '\n')
            f.close()
        
        # Print progress if indicated
        if self.print_progress:
            print(result_string)

        return

# ==================== Error classes ==============================

class ParameterError(Exception):
    pass

class InputError(Exception):
    pass

class ImplementationError(Exception):
    pass