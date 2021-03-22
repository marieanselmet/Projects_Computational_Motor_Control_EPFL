"""Exercise 8c"""

import pickle
import numpy as np
from simulation import simulation
from simulation_parameters import SimulationParameters
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import farms_pylog as pylog
from salamandra_simulation.data import AnimatData
from plot_results import plot_2d, plot_s_shape


def exercise_8c(timestep):
    """Exercise 8c"""
     # Parameters
     
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            drive=4,  # An example of parameter part of the grid search
            amplitude_gradient = [Rhead, Rtail],
            freqs = 1 # intrinsic frequencies
        )
        for Rhead in np.linspace(.1, 1, 5)
        for Rtail in np.linspace(.1, 1, 5)
    ]

    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        print(simulation_i)
        filename = './logs/ex_8c/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground' or 'amphibious'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)

        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
        # Save video
        if sim.options.record:
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.mp4',
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.html',
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))
        
        
def ex_8c_simulation(timestep, amplitude_gradient):
    
    sim_parameters = SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            drive=4,  # An example of parameter part of the grid search
            amplitude_gradient =  amplitude_gradient,
            freqs = 1
        )
    sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground' or 'amphibious'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video, see below for saving
        )
    # Log robot data
    data.to_file('./logs/ex_8c/simulation_gradient.h5', sim.iteration)
    # Log simulation parameters
    with open('./logs/ex_8c/simulation_gradient.pickle', 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
    if sim.options.record:
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation_max.mp4',
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.html',
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))


def ex_8c_heatmap(timestep):
    # Load data
    filename = './logs/ex_8c/simulation_{}.{}' # grid search 0.1-1
    # filename = './logs/ex_8c/simulation_closer_{}.{}' # grid search 0.1-0.6
    # filename = './logs/ex_8c/simulation_closer2_{}.{}' # grid search 0.25-0.55

    # contains 25 combinations of Rhead & Rtail and ... 
    energy_2d = np.zeros((25,3)) # ... the corresponding energy
    speed_2d =  np.zeros((25,3)) # ... the corresponding speed
    inv_COT_2d = np.zeros((25,3)) # ... the corresponding 1/COT = d/E (cost of transport)
    speed_vs_E_2d = np.zeros((25,3)) # ... the corresponding v/E (sort of cost of transport)
    distance_2d = np.zeros((25,3)) # ... the corresponding distance
    max_inv_COT = 0 # max of 1/COT 
    max_speed_vs_E = 0 # max of v/E
    opti_R = [] # optimized gradient of amplitude that induces max of 1/COT
    
    
    for i in range(25):
        data = AnimatData.from_file(filename.format(i, 'h5'), 2*14)
        joints_velocities = data.sensors.proprioception.velocities_all()
        joints_torques = data.sensors.proprioception.motor_torques()
        
        with open(filename.format(i, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
            
        gradient = parameters.amplitude_gradient # [Rhead, Rtail]
        # integration method : np.trapz, by default, integrate along axis = -1 (here, body joints)
        energy = np.sum(np.trapz(np.array(joints_velocities[2000:,:])*np.array(joints_torques[2000:,:]),dx = timestep, axis = 0))
        energy_2d[i] = [gradient[0], gradient[1], energy]

        links_positions = data.sensors.gps.urdf_positions()
        velocity = np.diff(links_positions[2000:,:,0], axis = 0) / timestep 
        speed = np.abs(np.mean(velocity))
        speed_2d[i] = [gradient[0], gradient[1], speed]
        
        ###
        # Cost of transport
        ###
        # "pseudo" COT = E/d => inv_COT = d/E
        # links_positions[-1,0,0] = x position of the head at the end of the simulation 
        inv_COT = (links_positions[-1,0,0]-links_positions[2000,0,0])/energy 
        
        ###
        # Interaction speed/Energy
        ###
        speed_vs_E = speed/energy
        
        # Search the max of inv_COT, because we search for the max of 1/E and the max of distance
        if max_inv_COT < inv_COT:
            max_inv_COT = inv_COT
            opti_R = [gradient[0], gradient[1]]
        
        # Search the max of speed_vs_E, because we search for the max of 1/E and the max of speed
        if max_speed_vs_E < speed_vs_E:
            max_speed_vs_E = speed_vs_E
            opti_R = [gradient[0], gradient[1]]
            
        inv_COT_2d[i] = [gradient[0], gradient[1], inv_COT]
        speed_vs_E_2d[i] = [gradient[0], gradient[1], speed_vs_E]
        distance_2d[i] = [gradient[0], gradient[1], (links_positions[-1,0,0]-links_positions[2000,0,0])]
        
    print('The most appropriate gradient of amplitude is : ',opti_R)
    
    ###
    # Plots
    ###
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15)
    
    ###
    # Energy
    ###
    plt.subplot(2,2,1)
    plot_2d(energy_2d, ['Rhead', 'Rtail', 'Energy [J]'], n_data = 5, log=False)
    
    ###
    # Speed
    ###
    plt.subplot(2,2,2)
    plot_2d(speed_2d, ['Rhead', 'Rtail', 'Speed [m/s]'], n_data = 5, log=False)
    
    ###
    # Distance
    ###
    plt.subplot(2,2,3)
    plot_2d(distance_2d, ['Rhead', 'Rtail', 'Distance [m]'], n_data = 5, log=False)
    
    ###
    # COT
    ###
    plt.subplot(2,2,4)
    plot_2d(inv_COT_2d, ['Rhead', 'Rtail', '1/COT = d/E [s$^2$/(kg*m)]' ], n_data = 5, log=False)
    
    plt.figure()
    plot_2d(speed_vs_E_2d, ['Rhead', 'Rtail', 'v/E [s/(kg*m)]'], n_data = 5, log=False)
    
    
if __name__ == '__main__':
    timestep = 1e-2
    
    exercise_8c(timestep)
    ex_8c_heatmap(timestep)

    ###
    # Test the simulation for the best amplitude gradient:  
    ###
    amplitude_gradient = [.2, .4]
    ex_8c_simulation(timestep, amplitude_gradient)

    ###
    # Plot the curvature of the salamander body:
    ###
    data = AnimatData.from_file('./logs/ex_8c/simulation_gradient.h5', 2*14)
    plot_s_shape(data.times, data.sensors.gps.urdf_positions(), int(0.133/timestep), walk = False, number_subplot = 8, amplitude_gradient)
        
