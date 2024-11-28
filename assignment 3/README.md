# README
General Instructions
===
> Use python version $\geq$ 3.10.0
### Required Imports
- `numpy`
- `matplotlib`
- `toml` : This code uses a TOML file for storing helicopter design configuration in a python dictionary format.
- `scipy` : uses `fsolve()` from `scipy.optimise` module
- `pandas` : To load airfoil characteristics, engine performance and atmosphere model as `DataFrame`.

### Design Parameters
Please look at `data/params.toml` for design details of the **Team Helicopter**. The user can edit this file to incorporate custom designs or alternatively pass in their own `.toml` file by simply adding `--params` followed by the path to the `.toml` file to the below python commands.

### Mission Parameters
Please look at `data/mission_A.toml` for an example of mission details. The mission segments are defined by assigning specific <i>types</i>. Users can modify their own mission profile and run it by adding `--mission-params` followed by the path to the `.toml` file, and can also choose to store the output plots in a separate directory by adding `--output-dir`. 

### Execution

1. Running missions 
   
   ```console
   > python3 mission.py --mission_params "data/mission_A.toml" --output_dir "./images/mission_A"
	#----------------------------------#
	#-----Starting takeoff segment-----#
	#----------------------------------#
	#-----------------------------------------#
	#-----Starting vertical_climb segment-----#
	#-----------------------------------------#
	#---------------------------------------#
	#-----Starting steady_climb segment-----#
	#---------------------------------------#
	#---------------------------------------#
	#-----Starting level_flight segment-----#
	#---------------------------------------#
	#-----------------------------------------#
	#-----Starting steady_descent segment-----#
	#-----------------------------------------#
	#--------------------------------#
	#-----Starting hover segment-----#
	#--------------------------------#
	#-------------------------------------------#
	#-----Starting vertical_descent segment-----#
	#-------------------------------------------#
	#-----------------------------------------#
	#-----Starting change_payload segment-----#
	#-----------------------------------------#
	#-----------------------------------------#
	#-----Starting vertical_climb segment-----#
	#-----------------------------------------#
	#---------------------------------------#
	#-----Starting steady_climb segment-----#
	#---------------------------------------#
	#---------------------------------------#
	#-----Starting level_flight segment-----#
	#---------------------------------------#
	#-----------------------------------------#
	#-----Starting steady_descent segment-----#
	#-----------------------------------------#
	#-------------------------------------------#
	#-----Starting vertical_descent segment-----#
	#-------------------------------------------#
	
	#-------------------------------#
	#-----Completed the Mission-----#
	#-------------------------------#
   ```

2. Trimming the helicopter : For manual trim, GUI is not the best option due to slow reaction. Hence, in the file `helicopter.py` one must manually edit the following snippet section to vary the velocity $V_{\infty}$ and control inputs $\theta_{0,m} ,\theta_{1c},\theta_{1s}$ and $\theta_{0,t}$
   
   ```python
   # values are in degrees
    main_rotor_collective = 0.792 # Enter your value here
    lateral_cyclic = 0.0	       # Enter your value here
    longitudinal_cyclic = 0.5     # Enter your value here
    tail_rotor_collective = 0.9   # Enter your value here

    forward_velocity = 13.88      # change from default of 50 * 5/18 m/s
   ```

   After putting in the control inputs, run the file using the following bash command to get the net moments and forces along with parameters like coning angles $\beta_0,\beta_{1s},\beta_{1c}$ and $\alpha_{TPP}$.
   
   ```console
   python3 helicopter.py
   ```

   Adjust values and run again until the moment and force residues are minimized to a satisfactory level.

### Sample Outputs


```console
> python helicopter.py

Main Rotor Collective:           2.274
Main Rotor Lateral Cyclic:       1.378
Main Rotor Longitudinal Cyclic:  2.3
Tail Rotor Collective:           1.912
Forward Velocity (m/s):          13.88888888888889


Net force (Hub Frame):    [-0.82676 -9.97956 -0.39148]
Net moment (Hub Frame):   [-1.40503  0.69189 -0.82953]

Net force (Body Frame):   [-0.82676  9.97956  0.39148]
Net moment (Body Frame):  [-1.40503 -0.69189  0.82953]
Main Rotor Thrust:        [   3.74181   21.67574 1963.44964]
Tail Rotor Thrust:        31.65530
Fuselage Drag:            38.82890
Mean of Max Alpha:        3.40522
Alpha TPP:                1.13360
beta_o:                   0.30308
beta_1c:                  0.12968
beta_1s:                  0.59529
Power:                    15320.92814
Power Available:          77040.00000
```

### Code Structure
We have the following Structure of our code

`Ass3/`
- `data/`  
	- `NACA0012.csv` : Airfoil characteristics
	- `isa.csv` : Atmosphere data
	- `turboprop_data.csv` : Engine data
	- `trim_curve.csv` : True and empirical data collected at various trimmed flight conditions using our code.
	- `params.toml` : Here is where you input the helicopter **design parameters**
   - `mission_A.toml` : Default **mission parameters** with rates and distances for each flight segment
   - `mission_B.toml`, `mission_C.toml`, `mission_D.toml` : Here is where you can input custom mission parameters.
- `images/` : contains all the generated plots and figures for presentation
   - `mission_A/`, `mission_B/`, `mission_C/`, `mission_D/` : contains respective mission plots of various performance characteristics.
- `presentation/` 
	- `pres.pdf` : Team slides
	- Remaining are latex code
- `README.md` : You are here :smiley:
- `helicopter.py`: File defining helicopter class build upon <i>rotorblade</i> class and <i>wing</i> class.
- `mission.py` : File defining <i>mission</i> class, and uses helicopter object to perform various missions and create plots for performance analysis.
- `optimize_mission.py` : File used to find an optimized mission profile to perform a given task.
- `trim_curve.ipynb` : Contains the curve fitting code for the data in `data/trim_curve.csv` which estimates the empirical power coefficient.
- `utilities.py` : Contains definition for <i>rotorblade</i> class that use different theories, <i>wing</i> class that uses linear aerodynamic theory, and other utility functions

### Remarks
1. A `data/mission.toml` can be created for any required mission. The user needs to specify the "type" of each mission segment, which determines which theory is used. All the altitudes, rates and distances are also user inputs, with no limit on the number of misison segments.
2. Upon running `mission.py`, the script will print whether the mission was successful or not (with reason). You will have to change either mission or design parameters iteratively until it works.
3. The empirical and true data (aerodynamic/performance parameters) are to be retrieved for several different trim conditions, and stored in `data/trim_curve.csv`.
4. The true data (generated from our code) is used to fit the empirical relations to obtain few specific parameters in `trim_curve.ipynb`.
5. These parameters are then to be put back in `data/params.toml`, which will be used to estimate the power.

## Credits and Acknowledgements
This code is designed for computing flight performance and mission of standard helicopters using low fidelity models like _BEMT_. We approach the problem in a geometric framework as described in the team slides :  `presentation/pres.pdf`. The code is a combined effort of 

 1. Ravi Kumar               (Roll no. 210010052)
 2. Vighnesh J.R.            (Roll no. 210010073)
 3. Shreyas N.B.             (Roll no. 210010061)
 4. Mayank Ghritlahre        (Roll no. 210010040)

Furthermore, we would like to convey special acknowledgements to professor Dhwanil Shukla of Department of Aerospace Engineering, IIT Bombay, for providing us with the theoretical knowledge and opportunity to build this tool as an assignment for his course AE 667, rotary wing aerodynamics.
