""" Lab 6 Exercise 3
This file implements the pendulum system with two muscles attached driven
by a neural network
"""

import numpy as np
from matplotlib import pyplot as plt
import farms_pylog as pylog
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

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=11.0)     # fontsize of the axes title
plt.rc('axes', labelsize=12.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12.0)    # fontsize of the tick labels


def system_init():
    """ Use this function to create a new default system. """
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

    ########## Network ##########
    # The network consists of four neurons
    N_params = NetworkParameters()  # Instantiate default network parameters
    N_params.D = 2.  # To change a network parameter
    # weights from Brown Symmetric four-neuron oscillator, L4P85
    N_params.w = [[0,-5,-5,0],
                  [-5,0,0,-5],
                  [5,-5,0,0],
                  [-5,5,0,0]] 
    # Similarly to change w -> N_params.w = (4x4) array

    # Create a new neural network with above parameters
    neural_network = NeuralSystem(N_params)
    pylog.info('Neural system initialized \n {}'.format(
        N_params.showParameters()))

    ########## ADD SYSTEMS ##########
    # Create system of Pendulum, Muscles and neural network using SystemClass
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system
    # Add the neural network to the system
    sys.add_neural_system(neural_network)

    ##### Time #####
    t_max = 2.5  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.asarray([np.pi/2, 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    l_ce_0 = sys.muscle_sys.initialize_muscle_length(np.pi/2)
    x0_M = np.asarray([0.05, l_ce_0[0], 0.05, l_ce_0[1]])

    x0_N = np.asarray([-0.5, 1, 0.5, 1])  # Neural Network Initial Conditions

    x0 = np.concatenate((x0_P, x0_M, x0_N))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time
    sim = SystemSimulation(sys)  # Instantiate Simulation object
    sim.initalize_system(x0, time)  # Initialize the system state
    return sim


def exercise3a():
    """ Simulate the system """

    # Create system
    sim = system_init()

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.asarray [time, states], states vector is in the same order as x0
    res = sim.results()

    # Plot the results 
    plt.figure('Exercise 3a')
    
    # Phase plot
    plt.subplot(1, 2, 1)
    plt.plot(res[:, 1], res[:, 2])
    plt.title('Pendulum Phase')
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    
    # Mean activity of the neurons
    plt.subplot(1, 2, 2)
    plt.plot(res[:, 0], res[:, 3])
    plt.plot(res[:, 0], res[:, 5])
    plt.title('Activation wave forms without an external drive')
    plt.xlabel('Time [s]')
    plt.ylabel('Activation')
    plt.legend(('Muscle 1', 'Muscle 2'), loc = 'upper right')
    plt.grid()
    
    plt.show()
    


def exercise3b():
    """ Simulate the system by applying different external drives"""

    # Definite low, intermediate and high external drives
    drives = np.linspace(0, 1, 3) 

    for drive in drives:
        # Create system
        sim = system_init()

        # Add external inputs to neural network
        sim.add_external_inputs_to_network(np.ones((len(sim.time), 4))*drive)
        
        # Integrate the system for the above initialized state and time
        sim.simulate()
        
        # Obtain the states of the system after integration
        res = sim.results()

        # Plot the results
        plt.figure('Exercise 3b - Drive '+str(drive))
        
        plt.subplot(1, 3, 1)
        plt.plot(res[:, 1], res[:, 2])
        plt.title('Pendulum phase for an external drive of ' + str(drive))
        plt.xlabel('Position [rad]')
        plt.ylabel('Velocity [rad.s]')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.plot(res[:, 0], res[:, 1])
        plt.title('Pendulum state for an external drive of ' + str(drive))
        plt.xlabel('Time [s]')
        plt.ylabel('Position [rad]')
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.plot(res[:, 0], res[:, 3])
        plt.plot(res[:, 0], res[:, 5])
        plt.title('Activation wave forms for an external drive of ' + str(drive))
        plt.xlabel('Time [s]')
        plt.ylabel('Activation')
        plt.legend(('Muscle 1', 'Muscle 2'), loc = 'upper right')
        plt.grid()
        
        plt.show()
        

def exercise3():
    """ Main function to run for Exercise 3. 
    """    
    exercise3a()
    exercise3b()
    
    if DEFAULT["save_figures"] is False:
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
    exercise3()

