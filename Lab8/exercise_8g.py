"""Exercise 8g"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from simulation_amphibious import simulation_amphibious
from simulation_parameters import SimulationParameters
from salamandra_simulation.data import AnimatData
from plot_results import plot_spine_angles, plot_limbs_angles
from save_figures import save_figures
import farms_pylog as pylog

        
def exercise_8g(timestep, plot=True):
    """Exercise 8g"""
    # Parameters
    drive_ground = 2.5
    drive_water = 4.5
    xlim = [-2.0, -1.2]
    drives = [drive_ground, drive_water]
    spawn_positions = [[-0.5, 0, 0.1], [-2.5, 0, 0.1]]
    spawn_orientations = [[np.pi, np.pi, 0], [0, 0, 0]]
    titles = ['Land to water', 'Water to land']
    
    for i in range(2):
        sim_parameters = SimulationParameters(
                duration = 20,
                timestep = timestep,
                amplitude_gradient = [0.2, 0.4],
                spawn_position = spawn_positions[i], # Robot position in [m]
                spawn_orientation = spawn_orientations[i], # Orientation in Euler angles [rad]
                drive = drives[i],
            )
    
        filename = './logs/exercise_8g/simulation_{}.{}'
        sim, data = simulation_amphibious(
            sim_parameters = sim_parameters,
            drive_ground = drive_ground,
            drive_water = drive_water,
            x_lim = xlim[i],
            fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            record=True,  # Record video, see below for saving
        )
        # Log robot data
        data.to_file(filename.format(i, 'h5'), sim.iteration)
        # Save video
        if sim.options.record:
            filename='./logs/exercise_8g/salamandra_robotica_simulation_{}.{}'
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename=filename.format(i, 'mp4'),
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename=filename.format(i, 'html'),
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))
              
        filename = './logs/exercise_8g/simulation_{}.{}'        
        data = AnimatData.from_file(filename.format(i, 'h5'), 24)        
        times = data.times
        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        joints_angles = np.asarray(data.sensors.proprioception.positions_all())
        osc_amplitudes = data.state.amplitudes_all()
            
        ### PLOTS
        plt.figure('Exercise 8g - {}'.format(titles[i]), figsize=(6, 8)) 
        
        # Plot of the spine angles
        plt.subplot(3, 1, 1)
        plot_spine_angles(times, joints_angles, 2)
        
        # Plot of the limbs angles
        plt.subplot(3, 1, 2)
        plot_limbs_angles(times, joints_angles, osc_amplitudes)
        
        # Plot of the x-position of the head
        plt.subplot(3, 1, 3)
        plt.plot(times, head_positions[:, 0], color='black')
        if i==0: plt.axhline(-2.0, color = 'red', linewidth = 1)
        if i==1: plt.axhline(-1.2, color = 'red', linewidth = 1)
        plt.axhline(-1.6, color = 'blue', linewidth = 1)
        plt.xlabel("Time [s]")
        plt.ylabel("Head x-position [m]")
        plt.grid(True)
        plt.tight_layout()    
    
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()



if __name__ == '__main__':
    exercise_8g(timestep=1e-2)

