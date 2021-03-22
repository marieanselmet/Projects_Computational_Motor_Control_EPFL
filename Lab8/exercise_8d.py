"""Exercise 8d"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from simulation import simulation
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import AnimatData
from plot_results import plot_spine_angles
from save_figures import save_figures
import farms_pylog as pylog 



def exercise_8d1(timestep, plot=True):
    """Exercise 8d1"""
    sim_parameters = SimulationParameters(
                    duration=15,
                    timestep=timestep,
                    amplitude_gradient = [0.2, 0.4],
                    drive = 4,
                )
    start_turn = 7.5/1e-2
    turn = [[1, 1], [1.2, 0.8]]
    
    # Grid search
    filename = './logs/ex_8d1/simulation_{}.{}'
    for simulation_i in range(2):
        sim, data = simulation(
             sim_parameters=sim_parameters, 
             start_turning = start_turn,
             turning = turn[simulation_i],
             arena='water',
             fast=True,
        )

        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    Head_positions = []
    Times = []
    Joints_angles = []
    for sim in range(2):
        data = AnimatData.from_file(filename.format(sim, 'h5'), 24)
            
        times = data.times
        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        joints_angles = data.sensors.proprioception.positions_all()
        
        Head_positions.append(head_positions)
        Times.append(times)        
        Joints_angles.append(np.asarray(joints_angles))
  
    # Plot of the trajectory  
    plt.figure('Exercise 8d1 - Trajectory') 
    plt.plot(Head_positions[0][:, 0], Head_positions[0][:, 1], label='No turn')
    plt.plot(Head_positions[1][:, 0], Head_positions[1][:, 1], label='Right turn')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(loc=4)
    plt.grid(True)
    
    # Plot of the spine angles
    plt.figure('Exercise 8d1 - Spine angles')
    plt.subplot(1, 2, 1)
    plot_spine_angles(Times[0], Joints_angles[0])
    plt.axhline(-9.3, color = 'black', linewidth = 0.5)
    plt.title('No turn')
    plt.subplot(1, 2, 2)
    plot_spine_angles(Times[1], Joints_angles[1])
    plt.axhline(-9.3, 0, 0.5, color = 'black', linewidth = 0.5)
    plt.axvline(7.5, 0.047, 0.055, color = 'black', linewidth = 0.5)
    plt.axhline(-9.2, 0.5, 1, color = 'black', linewidth = 0.5)
    plt.title('Right turn')
    
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
    


def exercise_8d2(timestep, plot=True):
    """Exercise 8d2"""
    parameter_set = [
        SimulationParameters(
            duration=15,
            timestep=timestep,
            amplitude_gradient = [0.2, 0.4],
            drive = 4,
            phase_lag = p
        )
        for p in [2*np.pi/10, -2*np.pi/10]
    ]

    # Grid search
    filename = './logs/ex_8d2/simulation_{}.{}'
    for simulation_i, sim_parameters in enumerate(parameter_set):
        sim, data = simulation(
             sim_parameters=sim_parameters,
             arena='water',
             fast=True,
        )

        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    
    Head_positions = []
    Times = []
    Joints_angles = []
    for sim in range(len(parameter_set)):
        
        data = AnimatData.from_file(filename.format(sim, 'h5'), 2*14)
            
        times = data.times
        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        joints_positions = data.sensors.proprioception.positions_all()
        
        Head_positions.append(head_positions)
        Times.append(times)        
        Joints_angles.append(np.asarray(joints_positions))
  
    # Plot of the trajectory  
    plt.figure('Exercise 8d2 - Trajectory') 
    plt.plot(Head_positions[0][:, 0], Head_positions[0][:, 1], label='Forwards')
    plt.plot(Head_positions[1][:, 0], Head_positions[1][:, 1], label='Backwards')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.grid(True)
    
    # Plot of the spine angles
    plt.figure('Exercise 8d2 - Spine angles')
    plt.subplot(1, 2, 1)
    plot_spine_angles(Times[0], Joints_angles[0])
    plt.title('Forwards')
    plt.subplot(1, 2, 2)
    plot_spine_angles(Times[1], Joints_angles[1])
    plt.title('Backwards')
    
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()



if __name__ == '__main__':
    exercise_8d1(timestep=1e-2)
    exercise_8d2(timestep=1e-2)

