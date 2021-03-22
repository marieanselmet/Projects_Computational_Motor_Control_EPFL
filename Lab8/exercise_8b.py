"""Exercise 8b"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import AnimatData
from plot_results import plot_2d
from save_figures import save_figures
import farms_pylog as pylog


def exercise_8b_grid_search(timestep):
    """Exercise 8b"""
    
    # Define the set of parameters for the grid search
    phase_lags = [np.pi/10, 2*np.pi/10, 3*np.pi/10, 2*np.pi/5, np.pi/2]
    nominal_amps = list(np.linspace(0.2, 0.6, 5))
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            drive=4.0, # For swimming
            phase_lag=phi,
            nominal_amp=R,
        )
        
        for R in nominal_amps
        for phi in phase_lags
    ]
    
    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_8b/simulation_{}.{}'
        pylog.info('Simulation {}: '.format(simulation_i))
        sim, data = simulation(
             sim_parameters=sim_parameters,  # Simulation parameters, see above
             #arena='water',  # Can also be 'ground' or 'amphibious'
             fast=True,  # For fast mode (not real-time)
             headless=True,  # For headless mode (No GUI, could be faster)
             #record=True,  # Record video, see below for saving
             #video_distance=1.5,  # Set distance of camera to robot
             #video_yaw=0,  # Set camera yaw for recording
             #video_pitch=-45,  # Set camera pitch for recording
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



def exercise_8b_plot_grid_search(timestep):
    
    # Load the results of the grid search
    filename = './logs/ex_8b/simulation_{}.{}'
    
    # Load and plot data
    # Contains 25 combinations of nominal_amps, phase_lags and the resulting characteristic measured:
    energies = np.zeros((25,3)) # energy
    speeds =  np.zeros((25,3)) # forward speed 
    traveled_distances = np.zeros((25,3)) # traveled distance
    inverse_CoTs = np.zeros((25,3)) # inverses of the cost of transports (1/CoT)
    max_inv_CoT = 0 # max(1/CoT)
    best_params = [] # optimized parameters leading to max(1/CoT)
    
    for sim in range(25):
        
        data = AnimatData.from_file(filename.format(sim, 'h5'), 24)
        with open(filename.format(sim, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)
        
        nominal_amp = parameters.nominal_amp
        phase_lag = parameters.phase_lag
        
        ## ENERGY
        joints_velocities = data.sensors.proprioception.velocities_all()
        joints_torques = data.sensors.proprioception.motor_torques()
        # Numerical integration with the trapeze method : np.trapz
        energy = np.sum(np.trapz(np.array(joints_velocities[2000:,:])*np.array(joints_torques[2000:,:]), dx = timestep))
        energies[sim] = [nominal_amp, phase_lag, energy]
        
        ## SPEED
        links_positions = data.sensors.gps.urdf_positions()
        forward_speed = np.mean(np.diff(links_positions[2000:,:,0], axis = 0) / timestep) # negative forward speed when the salamander swimms backwards
        speeds[sim] = [nominal_amp, phase_lag, forward_speed]
        
        ## TRAVELED DISTANCE (in absolute value)
        displacement = np.asarray(links_positions[-1,0,0]-links_positions[2000,0,0]) # links_positions[-1,0,0] = x position of the head at the end of the simulation
        traveled_distances[sim] = [nominal_amp, phase_lag, displacement]
        
        ## 1/CoT : defined here as (traveled distance)/energy
        inv_CoT = displacement/energy
        inverse_CoTs[sim] = [nominal_amp, phase_lag, inv_CoT]
        # Check if this set of parameters is better than the previous ones for maximizing 1/CoT
        if max_inv_CoT < inv_CoT:
            max_inv_CoT = inv_CoT
            best_params = [nominal_amp, phase_lag]
    
    # Plot the heat maps of the results of the grid search
    plt.figure("Exercise 8b - grid search results", figsize=(14, 12))

    ## SPEED
    plt.subplot(2,2,1)
    plot_2d(speeds, ['Nominal amplitude [rad]', 'Phase lag [rad]', 'Speed [m/s]'], n_data=5, log=False)
    
    ## TRAVELED DISTANCE
    plt.subplot(2,2,2)
    plot_2d(traveled_distances, ['Nominal amplitude [rad]', 'Phase lag [rad]', 'Distance [m]'], n_data=5, log=False)
    
    ## ENERGY
    plt.subplot(2,2,3)
    plot_2d(energies, ['Nominal amplitude [rad]', 'Phase lag [rad]', 'Energy [J]'], n_data=5, log=False)

    ## 1/CoT
    plt.subplot(2,2,4)
    plot_2d(inverse_CoTs, ['Nominal amplitude [rad]', 'Phase lag [rad]', '1/(CoT) = d/E'], n_data=5, log=False)
    # Parameters maximizing 1/CoT
    pylog.info('Best parameters for salamander locomotion: Nominal amp = {} [rad], Phase lag = {} [rad]'.format(best_params[0], best_params[1]))



if __name__ == '__main__':
    exercise_8b_grid_search(timestep=1e-2)
    exercise_8b_plot_grid_search(timestep=1e-2)


