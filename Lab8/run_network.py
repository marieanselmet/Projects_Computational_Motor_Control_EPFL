"""Run network without Pybullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog
from network import SalamandraNetwork
from save_figures import save_figures
from parse_args import save_plots
from simulation_parameters import SimulationParameters
from plot_results import plot_spine_angles, plot_limbs_angles, plot_freq


def run_network(duration, update=True, drive=0): 
    """Run network without Pybullet and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        description
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    timestep = 1e-2
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    sim_parameters = SimulationParameters(drive=drive)
    network = SalamandraNetwork(sim_parameters, n_iterations)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs
    nominal_amplitudes_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.nominal_amplitudes)
    ])
    nominal_amplitudes_log[0, :] = network.robot_parameters.nominal_amplitudes
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)
    
    inst_freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
   
    # Run network ODE and log data
    tic = time.time()
    drives = np.linspace(0, 6, n_iterations)
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive = drives[i]
                    # amplitude_gradient=None,
                    # phase_lag=None
                )
            )
            
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
        nominal_amplitudes_log[i+1, :] = network.robot_parameters.nominal_amplitudes
        inst_freqs_log[i+1,:] = (network.get_state(i+1)-network.get_state(i))/(2*np.pi*timestep)
        
    # Alternative option
    # phases_log[:, :] = network.state.phases()
    # amplitudes_log[:, :] = network.state.amplitudes()
    # outputs_log[:, :] = network.get_motor_position_output()
    toc = time.time()

    # Network performance
    pylog.info("Time to run simulation for {} steps: {} [s]".format(
        n_iterations,
        toc - tic
    ))
    
    # Implement plots of network results  
    xcoords = [20/3, 20, 100/3]
    plt.figure('Exercise 8a - Output', figsize=(6, 8)) 
    plt.subplot(4, 1, 1)
    plot_spine_angles(times, outputs_log, 2)
    for xc in xcoords:
        plt.axvline(x=xc, color = 'black', linestyle = ':', linewidth = 1)
    
    plt.subplot(4, 1, 2)
    plot_limbs_angles(times, outputs_log, amplitudes_log, 2)
    for xc in xcoords:
        plt.axvline(x=xc, color = 'black', linestyle = ':', linewidth = 1)
    
    plt.subplot(4, 1, 3)
    plot_freq(times, inst_freqs_log)
    for xc in xcoords:
        plt.axvline(x=xc, color = 'black', linestyle = ':', linewidth = 1)
    
    plt.subplot(4, 1, 4)
    plt.plot(times, drives, color='black')
    plt.yticks([1, 3, 5])
    drive_limits = [1, 3, 5]
    for xc in xcoords:
        plt.axvline(x=xc, color = 'black', linestyle = ':', linewidth = 1)
    for yc in drive_limits:
        plt.axhline(yc, color = 'red', linewidth = 0.5)
    plt.text(20/3+0.2, 5.2, 'Walking')
    plt.text(20+0.2, 5.2, 'Swimming')       
    plt.xlabel("Time [s]")
    plt.ylabel("Drive")
    plt.tight_layout()
    
    plt.figure('Exercise 8a - Parameters', figsize=(10, 4)) 
    plt.subplot(1, 2, 1)
    plt.plot(drives, freqs_log[:,0], color='black', label='Body')
    plt.plot(drives, freqs_log[:,20], color='black', linestyle=':', label='Limb')
    plt.xlabel("Drive")
    plt.ylabel("f [Hz]")
    plt.legend(loc=2)
    
    plt.subplot(1, 2, 2)
    plt.plot(drives, nominal_amplitudes_log[:,0], color='black', label='Body')
    plt.plot(drives, nominal_amplitudes_log[:,20], color='black', linestyle=':', label='Limb')
    plt.xlabel("Drive")
    plt.ylabel("R")
    plt.legend(loc=2)
    

def main(plot):
    """Main"""
    run_network(duration=40)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

