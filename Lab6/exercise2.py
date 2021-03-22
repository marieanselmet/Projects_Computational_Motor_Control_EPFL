""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt

import farms_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSystem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation

import warnings
warnings.filterwarnings("ignore")


# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True



def system_init(t_max = 2):
    """Initialize default system."""
    ########## PENDULUM ##########
    # Define and Setup your pendulum model here
    # Check Pendulum.py for more details on Pendulum class
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    P_params.L = 1.0  # To change the default length of the pendulum
    P_params.m = 0.25  # To change the default mass of the pendulum
    pendulum = PendulumSystem(P_params)  # Instantiate Pendulum object
    #### CHECK OUT Pendulum.py to ADD PERTURBATIONS TO THE MODEL #####
    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    ########## MUSCLES ##########
    # Define and Setup your muscle model here
    # Check MuscleSystem.py for more details on MuscleSystem class
    m1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    m1_param.f_max = 200.  # To change Muscle 1 max force
    m1_param.l_opt = 0.4
    m1_param.l_slack = 0.45
    m2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    m2_param.f_max = 200.  # To change Muscle 2 max force
    m2_param.l_opt = 0.4
    m2_param.l_slack = 0.45
    m1 = Muscle('m1', m1_param)  # Instantiate Muscle 1 object
    m2 = Muscle('m2', m2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    # Instantiate Muscle System with two muscles
    muscles = MuscleSystem(m1, m2)
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        m1.parameters.showParameters(),
        m2.parameters.showParameters()))
    # Define Muscle Attachment points
    m1_origin = np.asarray([0.0, 0.9])  # Origin of Muscle 1
    m1_insertion = np.asarray([0.0, 0.15])  # Insertion of Muscle 1

    m2_origin = np.asarray([0.0, 0.8])  # Origin of Muscle 2
    m2_insertion = np.asarray([0.0, -0.3])  # Insertion of Muscle 2
    # Attach the muscles
    muscles.attach(np.asarray([m1_origin, m1_insertion]),
                   np.asarray([m2_origin, m2_insertion]))

    ########## ADD SYSTEMS ##########
    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ########## INITIALIZATION ##########
    time = np.arange(0., t_max, 0.001)  # Time vector
    ##### Model Initial Conditions #####
    x0_P = np.asarray([np.pi/2, 0.0])  # Pendulum initial condition
    # Muscle Model initial condition
    l_ce_0 = sys.muscle_sys.initialize_muscle_length(np.pi/2)
    x0_M = np.asarray([0.05, l_ce_0[0], 0.05, l_ce_0[1]])
    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ########## System Simulation ##########
    sim = SystemSimulation(sys)  # Instantiate Simulation object
    # Simulate the system for given time
    sim.initalize_system(x0, time)  # Initialize the system state
    return sim



def poincare_crossings(res, threshold, crossing_index, figure):
    """ Study poincaré crossings """
    ci = crossing_index

    # Extract state of first trajectory
    state1 = np.array(res[:, 1])
    state2 = np.array(res[:, 2])
    state = np.column_stack((state1, state2)) 
    
    # Crossing index (index corresponds to last point before crossing)
    idx = np.argwhere(np.diff(np.sign(state[:, ci] - threshold)) < 0)

    # Linear interpolation to find crossing position on threshold
    # Position before crossing
    pos_pre = np.array([state[index[0], :] for index in idx])
    # Position after crossing
    pos_post = np.array([state[index[0]+1, :] for index in idx])
    # Position on threshold
    pos_threshold = [
        (
            (threshold - pos_pre[i, 1])/(pos_post[i, 1] - pos_pre[i, 1])
        )*(
            pos_post[i, 0] - pos_pre[i, 0]
        ) + pos_pre[i, 0]
        for i, _ in enumerate(idx)
    ] 
    val_min = np.sort(pos_threshold)[1]
    val_max = np.sort(pos_threshold)[-1]
    bnd = 0.3*(val_max - val_min)
    
    # Plot
    # Zoom on limit cycle
    plt.figure(figure)
    plt.subplot(1, 2, 1)
    plt.plot(res[:, 1], res[:, 2])
    for pos in pos_threshold:
        plt.plot(pos, threshold, "ro")
    plt.xlim([val_min-bnd, val_max+bnd])
    plt.ylim([threshold-1e-7, threshold+1e-7])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid(True)
    
    # Figure limit cycle variance
    plt.subplot(1, 2, 2)
    plt.plot(pos_threshold, "o-")
    plt.xlabel("Number of Poincaré section crossings")
    plt.ylabel("Position [rad] (Velocity = {} rad.s)".format(threshold))
    plt.grid(True)
    
    plt.subplots_adjust(wspace = 0.3)




def exercise2a():
    thetas = np.linspace(np.pi/4, 3*np.pi/4, 20)
    
    # Default set of attachment points:
    # m1_origin = [0.0, 0.9]  # origin of muscle 1
    # m1_insertion = [0.0, 0.15]  # insertion of muscle 1
    # m2_origin = [0.0, 0.8]  # origin of muscle 2
    # m2_insertion = [0.0, -0.3]  # insertion of muscle 2
    
    # distances between the muscles attachment point and the pendulum origin = [0.0, 0.0]
    a_muscle1 = [0.9, 0.15]
    a_muscle2 = [0.8, 0.3]
    
    plt.figure('Exercise 2a')
    
    for a in [a_muscle1, a_muscle2]:
        lengths = []
        moments = []
        
        for theta in thetas:
            a1 = a[0]
            a2 = a[1]
            
            length = sqrt(a1**2 + a2**2 + 2*a1*a2*np.cos(theta))
            moment = (a1 * a2 * np.sin(theta)) / length
            
            lengths.append(length)
            moments.append(moment)
        
        plt.subplot(1, 2, 1)
        plt.plot(thetas, lengths)
        
        plt.subplot(1, 2, 2)
        plt.plot(thetas, moments)
        
    plt.subplot(1, 2, 1)
    plt.xlabel('Pendulum angular position [rad]')
    plt.ylabel('Muscle length [m]')
    plt.legend(('Muscle 1', 'Muscle 2'))
    plt.grid() 
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Pendulum angular position [rad]')
    plt.ylabel('Moment arm [m]')
    plt.legend(('Muscle 1', 'Muscle 2'))
    plt.grid()  
    
    
    
def exercise2b():
    sim = system_init(t_max = 5)

    # Add muscle activations to the simulation
    act1 = np.array([0.45 * np.sin(10*sim.time) + 0.55]).T
    act2 = np.array([0.45 * np.sin(10*sim.time+np.pi/2) + 0.55]).T
    
    activations = np.hstack((act1, act2))
    sim.add_muscle_stimulations(activations)

    # Plotting the activation wave forms used
    plt.figure('Exercise 2b - Activation Wave Forms')
    plt.plot(sim.time, act1)
    plt.plot(sim.time, act2)
    plt.xlabel('Time [s]')
    plt.ylabel('Activation')
    plt.legend(('Muscle 1','Muscle 2'))
    plt.grid
    
    plt.figure('Exercise 2b - Pendulum Phase')
    
    # NO PERTURBATION
    # Integrate the system for the above initialized state and time
    sim.simulate()
    # Obtain the states of the system after integration
    res = sim.results()
    # Plotting the results
    plt.subplot(1, 2, 1)
    plt.plot(res[:, 1], res[:, 2])
    plt.plot(1.916, 2, "ro")
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.title('No perturbation')
    plt.grid()
    
    # PERTURBATION
    # Perturb the pendulum model by setting the state of the pendulum model to zeros between time interval 1.2 < t < 1.3. 
    sim.sys.pendulum_sys.parameters.PERTURBATION = True
    # Integrate the system for the above initialized state and time
    sim.simulate()
    # Obtain the states of the system after integration
    res = sim.results()
    # Plotting the results
    plt.subplot(1, 2, 2)
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.title('Perturbation')
    plt.grid()
    
    # POINTCARÉ MAP
    sim.sys.pendulum_sys.parameters.PERTURBATION = False
    # Integrate the system for the above initialized state and time
    sim.simulate()
    # Obtain the states of the system after integration
    res = sim.results()
    poincare_crossings(res, 2, 1, "Exercise 2b - Pointcaré Map") 
    
    

def exercise2c():
    sim = system_init()
    
    frequencies = np.arange(0,10,2)
    
    plt.figure('Exercise 2c')
    legend = []
    
    for freq in frequencies:
        # Add muscle activations to the simulation
        act1 = np.array([0.45 * np.sin(freq*sim.time) + 0.55]).T
        act2 = np.array([0.45 * np.sin(freq*sim.time+np.pi/2) + 0.55]).T
        activations = np.hstack((act1, act2))
        sim.add_muscle_stimulations(activations)
        
        # NO PERTURBATION
        # Integrate the system for the above initialized state and time
        sim.simulate()
        # Obtain the states of the system after integration
        res = sim.results()
    
        plt.plot(res[:, 1], res[:, 2])
        legend += [str(freq)+' Hz']
    
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.title('Different stimulation frequencies')
    plt.legend(legend, loc = 'lower right')
    plt.grid()



def exercise2():
    """ Main function to run for Exercise 2.
    """
    exercise2a()
    exercise2b()
    exercise2c()
    
    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()

