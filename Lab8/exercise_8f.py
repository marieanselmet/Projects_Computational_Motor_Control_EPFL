"""Exercise 8f"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from simulation import simulation
import matplotlib.animation as manimation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import AnimatData
from plot_results import *
from save_figures import save_figures
import farms_pylog as pylog


def exercise_8f1_2(timestep, drive = 2):
    """Exercise 8f 1) and 2)"""
    
    # Parameters
    sim_parameters = SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0.5, 0, 0.1], # Robot position in [m]
            spawn_orientation=[0, 0, np.pi/2], # Orientation in Euler angles [rad]
            drive=drive
            )

    # Simulate
    if(drive < 3): arena = 'amphibious'
    else: arena = 'water'
    sim, data = simulation(
             sim_parameters = sim_parameters,  # Simulation parameters, see above
             arena=arena,  # Can also be 'ground' or 'amphibious'
             fast=True,  # For fast mode (not real-time)
             headless=True,  # For headless mode (No GUI, could be faster)
             #record=True,  # Record video, see below for saving
             #video_distance=1.5,  # Set distance of camera to robot
             #video_yaw=0,  # Set camera yaw for recording
             #video_pitch=-45,  # Set camera pitch for recording
        )
    
    filename = './logs/ex_8f1_2/sim_gaits.{}'
    # Log robot data
    data.to_file(filename.format('h5'))
    # Log simulation parameters
    with open(filename.format('pickle'), 'wb') as param_file:
        pickle.dump(sim_parameters, param_file)
    
    # Record video
    # Save video
    if sim.options.record:
        if 'ffmpeg' in manimation.writers.avail:
            sim.interface.video.save(
                filename='./logs/ex_8f1_2/salamandra_robotica_simulation.mp4',
                iteration=sim.iteration,
                writer='ffmpeg',
            )
        elif 'html' in manimation.writers.avail:
            # FFmpeg might not be installed, use html instead
            sim.interface.video.save(
                filename='./logs/ex_8f1_2/salamandra_robotica_simulation.html',
                iteration=sim.iteration,
                writer='html',
            )
        else:
            pylog.error('No known writers, maybe you can use: {}'.format(
                manimation.writers.avail
            ))
    
    
    # Load data
    data = AnimatData.from_file(filename.format('h5'), 24)
    with open(filename.format('pickle'), 'rb') as param_file:
        parameters = pickle.load(param_file)
        
    times = data.times
    
    osc_phases = np.asarray(data.state.phases_all())
    # Compute the phase lags along the spine (downwards here)
    phase_lags = np.diff(osc_phases[:,:20], axis = 1)
    # Remove the phase difference between oscillators 10-11 that are not coupled
    phase_lags = np.concatenate((phase_lags[:,:9], phase_lags[:,10:]), axis = 1)
    
    joints_positions = np.asanyarray(data.sensors.proprioception.positions_all()) 
    
    
    ## PLOT RESULTS
    plt.figure("Exercise 8f - Spine movement ", figsize=(16, 14)) 
    # Plot the spine angles 
    plt.subplot(2, 2, 1)
    plot_spine_angles(times, joints_positions, vspace=2)
    
    # Plot the phase differences along the spine 
    plt.subplot(2, 2, 2)
    for i in range(4):
        if(i==0): plt.plot(times, phase_lags[:,i], color='blue', label='Trunk')
        else: plt.plot(times, phase_lags[:,i], color='blue')
    plt.plot(times, phase_lags[:,4], color='red', label='Trunk/Tail transition')
    for i in range(5,9):
        if(i==5): plt.plot(times, phase_lags[:,i], color='green', label='Tail')
        else: plt.plot(times,  phase_lags[:,i], color='green')
    for i in range(9,13):
        plt.plot(times, phase_lags[:,i], color='blue')
    plt.plot(times, phase_lags[:,13], color='red')
    for i in range(14,18):
        plt.plot(times, phase_lags[:,i], color='green')
    plt.xlabel("Time [s]")
    plt.ylabel("Phase differences")
    plt.grid()
    plt.legend(loc=1)
    
    # Plot the stable phases differences along the spine oscillators 
    oscillators = np.arange(1, 19, 1)
    stable_phase_lags = np.mean(phase_lags[-200:,:], axis = 0)
    # Left spine oscillators 
    plt.subplot(2, 2, 3)
    plt.plot(oscillators[:9], stable_phase_lags[:9], marker='o', label = 'Left side')
    plt.xticks(oscillators[:9], [str(oscillator) for oscillator in oscillators[:9]])
    plt.xlabel("Oscillators coupling index (vertical coupling between 2 body oscillators)")
    plt.ylabel("Stable phase differences")
    plt.grid()
    plt.legend(loc=2)
    plt.subplot(2, 2, 4)
    # Right spine oscillators
    plt.plot(oscillators[9:], stable_phase_lags[9:], marker='o', label = 'Right side')
    plt.xticks(oscillators[9:], [str(oscillator) for oscillator in oscillators[9:]])
    plt.xlabel("Oscillators coupling index (vertical coupling between 2 body oscillators)")
    plt.ylabel("Stable phase differences")
    plt.grid()
    plt.legend(loc=2)
    


def exercise_8f3(timestep):
    """Exercise 8f3"""
    
    # Parameters list
    phase_offsets = list(np.arange(-np.pi, np.pi+np.pi/9, np.pi/9))
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s] 
            timestep=timestep,  # Simulation timestep in [s]
            drive=2.0, # To have the same walking gait as in the previous questions
            offset=phi,
            nominal_amp=0.3
        )
        for phi in phase_offsets
    ]
    
    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_8f3/simulation_{}.{}'
        sim, data = simulation(
             sim_parameters=sim_parameters,  # Simulation parameters, see above
             #arena='amphibious',  # Can also be 'ground' or 'amphibious'
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
    
    # Load and plot results
    phase_offsets = []
    speed = np.zeros(len(parameter_set))
    for simulation_i in range(len(parameter_set)):
    
        data = AnimatData.from_file(filename.format(simulation_i, 'h5'), 24)
        with open(filename.format(simulation_i, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)

        phase_offsets.append(parameters.offset)
        
        links_positions = data.sensors.gps.urdf_positions()
        # Velocity estimated only after 20s of simulation (for stabilization)
        velocity = np.sqrt((np.diff(links_positions[2000:,0,0], axis = 0) / timestep)**2 + (np.diff(links_positions[2000:,0,1], axis = 0) / timestep)**2 + (np.diff(links_positions[2000:,0,2], axis = 0) / timestep)**2)
        speed[simulation_i] = np.abs(np.mean(velocity))
        
    # Plot grid search results
    plt.figure("Grid search phase offsets")
    plt.plot(phase_offsets, speed)
    plt.xlabel('Phase offset [rad]')
    plt.ylabel('Mean velocity [m/s]')
    plt.title('Mean velocity depending on the phase offset')
    plt.grid()
    plt.show()
    
    

def exercise_8f4(timestep, freq = None):
    """Exercise 8f4"""

    # Parameters
    nominal_amps = list(np.linspace(0.0, 0.5, 20)) # To stay in the range of the walking gait
    
    parameter_set = [
        SimulationParameters(
            duration=30,  # Simulation duration in [s] 
            timestep=timestep,  # Simulation timestep in [s]
            drive=2.0, # For walking
            phase_lag=2*np.pi/10,
            freqs=freq,
            nominal_amp=R
        )
        for R in nominal_amps
    ]
    
    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/ex_8f4/simulation_{}.{}'
        sim, data = simulation(
             sim_parameters=sim_parameters,  # Simulation parameters, see above
             #arena='amphibious',  # Can also be 'ground' or 'amphibious'
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
            
    # Load and plot results
    nominal_amps = []
    speed = np.zeros(len(parameter_set))
    for simulation_i in range(len(parameter_set)):
    
        data = AnimatData.from_file(filename.format(simulation_i, 'h5'), 24)
        with open(filename.format(simulation_i, 'pickle'), 'rb') as param_file:
            parameters = pickle.load(param_file)

        nominal_amps.append(parameters.nominal_amp)
        
        links_positions = data.sensors.gps.urdf_positions()
        # Velocity estimated only after 20s of simulation (for stabilization)
        velocity = np.sqrt((np.diff(links_positions[2000:,0,0], axis = 0) / timestep)**2 + (np.diff(links_positions[2000:,0,1], axis = 0) / timestep)**2 + (np.diff(links_positions[2000:,0,2], axis = 0) / timestep)**2)
        speed[simulation_i] = np.abs(np.mean(velocity))
        
    
    # Plot grid search results
    plt.figure("Grid search nominal amplitudes")
    plt.plot(nominal_amps, speed)
    plt.xlabel('Nominal radius [rad]')
    plt.ylabel('Mean velocity [m/s]')
    plt.title('Mean velocity depending on the nominal radius')
    plt.grid()
    plt.show()



if __name__ == '__main__':
    exercise_8f1_2(timestep=1e-2, drive=2) # drive=2 for walking
    exercise_8f1_2(timestep=1e-2, drive=4) # drive=4 for swimming
    exercise_8f3(timestep=1e-2)
    exercise_8f4(timestep=1e-2, freq=0.6) # intrisinc frequency: for the plots, the following values were implemented: 0.4 and 0.6


