"""Plot results"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from save_figures import save_figures
from parse_args import save_plots
from salamandra_simulation.data import AnimatData


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot trajectory"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_spine_angles(times, joint_angles, vspace=1):
    """Plot angles of the salamander spine"""
    for i in range(5):
        if(i==0): 
            plt.plot(times, joint_angles[:, i]-vspace*i, color='blue', label='Trunk')
        else: plt.plot(times, joint_angles[:, i]-vspace*i, color='blue')
    for i in range(5,10):
        if(i==5): plt.plot(times, joint_angles[:, i]-vspace*i, color='green', label='Tail')
        else: plt.plot(times, joint_angles[:, i]-vspace*i, color='green')
    plt.yticks([])
    plt.xlabel("Time [s]")
    plt.ylabel("Spine angles")
    plt.legend(loc=1)


def plot_limbs_angles(times, joint_angles, osc_amplitudes, vspace=1):
    """Plot angles of the salamander limbs"""
    plt.plot(times, osc_amplitudes[:,10]*(1+np.cos(joint_angles[:,10])), color='blue', label='Upper')
    plt.plot(times, osc_amplitudes[:,11]*(1+np.cos(joint_angles[:,11]))-vspace, color='blue')
    plt.plot(times, osc_amplitudes[:,12]*(1+np.cos(joint_angles[:,12]))-2*vspace, color='green', label='Lower')
    plt.plot(times, osc_amplitudes[:,13]*(1+np.cos(joint_angles[:,13]))-3*vspace, color='green')
    plt.yticks([])
    plt.xlabel("Time [s]")
    plt.ylabel("Limbs angles")
    plt.legend(loc=1)
    
    
def plot_freq(times, freq):
    """Plot instantaneous frequencies"""
    for i in range(24):
        plt.plot(times, freq[:, i])
    plt.xlabel("Time [s]")
    plt.ylabel("Instantaneous frequencies")
    
    
def plot_s_shape(times, link_data, steps, walk, number_subplot, amplitude_gradient):
    """ Plot position of the salamander seen from above
    - steps = number of steps between each subplot
    - walk = True => limbs needed. Walk = False => no limbs plotted
    - number_subplot = number of different subplots. From one subplot to the next, time inscreased of steps
    - amplitude_gradient = [Rhead, Rtail]
    """
    
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    fig, axes = plt.subplots(nrows=number_subplot, ncols=1, sharex=True)
    
    for j in range(number_subplot):
        x_body = []
        y_body = []
        x_limb = []
        y_limb = []
        
        for i in range(15):
            # link_data[iteration, link_id, xyz]
            data = np.array(link_data[:, i, :])
            
            if i < 11: #limbs
                x_body.append(data[2000+j*steps,0])
                y_body.append(data[2000+j*steps,1])       
            else:
                x_limb.append(data[2000+j*steps,0])
                y_limb.append(data[2000+j*steps,1]) 
        
        min_x = min(min(x_body),min(x_limb)) # min_x for each subplot
        max_x = max(max(x_body),max(x_limb)) # max_x for each subplot
 
        # Plot a line for the mean for each subplot
        x_mean = np.linspace(min_x,max_x,50)
        mean = np.mean(link_data[2000+j*steps, :, 1])*np.ones(x_mean.size) 
        axes[j].plot(x_mean, mean, linewidth=.8)
 
        # Plot the body
        axes[j].plot(x_body,y_body, '.-', color = 'black')
        # Don't need the limbs if swimming
        if walk:
            axes[j].plot(x_limb,y_limb, 'o', color = 'red')
        
        # Define the limit of y axis for each subplot
        y_lim_max = max(max(y_body),max(y_limb))
        y_lim_min = min(min(y_body),min(y_limb))
        axes[j].set_ylim([y_lim_min-.1,y_lim_max+.1])         

    axes[-1].set_xlabel("x [m]", fontsize = 20)
    axes[4].set_ylabel("y [m]", fontsize = 20)
    axes[0].set_title("Amplitude gradient = [{} - {}]".format(amplitude_gradient[0],amplitude_gradient[1]), fontsize = 20)
    fig.subplots_adjust(hspace=.4)
    
    
def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0], fontsize = 15)
    plt.ylabel(labels[1], fontsize = 15)
    plt.title('')
    cbar = plt.colorbar()
    cbar.set_label(labels[2], fontsize = 15)


def main(plot=True):
    """Main"""
    # Load data
    data = AnimatData.from_file('logs/example/simulation_0.h5', 2*14)
    with open('logs/example/simulation_0.pickle', 'rb') as param_file:
        parameters = pickle.load(param_file)
    times = data.times
    timestep = times[1] - times[0]  # Or parameters.timestep
    amplitudes = parameters.amplitudes
    phase_lag = parameters.phase_lag
    osc_phases = data.state.phases_all()
    osc_amplitudes = data.state.amplitudes_all()
    links_positions = data.sensors.gps.urdf_positions()
    head_positions = links_positions[:, 0, :]
    tail_positions = links_positions[:, 10, :]
    joints_positions = data.sensors.proprioception.positions_all()
    joints_velocities = data.sensors.proprioception.velocities_all()
    joints_torques = data.sensors.proprioception.motor_torques()

    # Notes:
    # For the gps arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    # plt.figure("Positions")
    # plot_positions(times, head_positions)
    # plt.figure("Trajectory")
    # plot_trajectory(head_positions)
   

        # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

