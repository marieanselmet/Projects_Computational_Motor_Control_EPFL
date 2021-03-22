""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import farms_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")
    
    # Create muscle object
    muscle = Muscle(parameters)
    
    # Instantiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)
        
    # Evaluate for several muscle stretches
    muscle_stretches = np.arange(0,0.31,0.01)
    
    # Evaluate for a single muscle stimulation
    muscle_stimulation = 1
    
    # Set the initial condition
    x0 = [0.0, sys.muscle.l_opt]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contractile length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)

    # Initialize empty arrays to record the forces and length values
    active_forces = []
    passive_forces = []
    total_forces = [] 
    lengths = []
    
    # Run the experiment for each muscle stretch
    for stretch in muscle_stretches:
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=muscle_stimulation,
                               muscle_length=stretch)
        # Record the forces and length values at the last time point of the stimulation
        active_forces.append(result.active_force[-1])
        passive_forces.append(result.passive_force[-1])
        total_forces.append(result.active_force[-1]+result.passive_force[-1])
        lengths.append(result.l_ce[-1])
    
    # Plot the forces in function of the length
    plt.figure('Isometric muscle experiment 1a')
    plt.plot(lengths, active_forces)
    plt.plot(lengths, passive_forces)
    plt.plot(lengths, total_forces)
    plt.xlabel('Muscle contractile length [m]')
    plt.ylabel('Muscle force [N]')
    plt.legend(('Active','Passive','Total'))
    plt.grid()
    plt.show()
    
    
def exercise1b():
    """ Exercise 1b
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")
    
    # Create muscle object
    muscle = Muscle(parameters)
    
    # Instantiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)
        
    # Evaluate for several muscle stretches
    muscle_stretches = np.arange(0,0.31,0.01)
    
    # Evaluate for several muscle stimulation
    muscle_stimulations = np.arange(0,1.1,0.2)
    
    # Set the initial condition
    x0 = [0.0, sys.muscle.l_opt]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contractile length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    # Initialize empty arrays to record the force and length values for all the stimulations
    Active_forces = []
    Lengths = []
    
    # Run the experiment for each muscle stimulation and each muscle stretch
    for stimulation in muscle_stimulations:
        # Initialize empty arrays to record the force and length values for each stimulation
        active_forces = []
        lengths = []
        for stretch in muscle_stretches:
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=stimulation,
                                   muscle_length=stretch)
            # Record the forces and length values at the last time point of the stimulation
            active_forces.append(result.active_force[-1])
            lengths.append(result.l_ce[-1])
        Active_forces.append(active_forces)
        Lengths.append(lengths)
    
    # Plot the forces in function of the length
    plt.figure('Isometric muscle experiment 1b')
    for i in range(len(muscle_stimulations)):
        plt.plot(Lengths[i], Active_forces[i])
    plt.xlabel('Muscle contractile length [m]')
    plt.ylabel('Muscle force [N]')
    plt.legend(np.round(muscle_stimulations, 2), title='Muscle stimulation')
    plt.grid()
    plt.show()


def exercise1c():
    """ Exercise 1c
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""
    
    l_opt = [0.1, 0.5]
    muscle_stretch_max = [0.31, 0.96]
    plt.figure('Isometric muscle experiment 1c')
    
    for i in range(2):
        # Definition of muscles
        parameters = MuscleParameters(l_opt = l_opt[i])
        pylog.warning("Loading default muscle parameters")
        pylog.info(parameters.showParameters())
        pylog.info("Use the parameters object to change the muscle parameters")
        
        # Create muscle object
        muscle = Muscle(parameters)
        
        # Instantiate isometric muscle system
        sys = IsometricMuscleSystem()
    
        # Add the muscle to the system
        sys.add_muscle(muscle)
    
        # You can still access the muscle inside the system by doing
        # >>> sys.muscle.l_opt # To get the muscle optimal length
            
        # Evaluate for several muscle stretches
        muscle_stretches = np.arange(0, muscle_stretch_max[i], 0.01)
        
        # Evaluate for one muscle stimulation
        muscle_stimulation = 1
        
        # Set the initial condition
        x0 = [0.0, sys.muscle.l_opt]
        # x0[0] --> muscle stimulation intial value
        # x0[1] --> muscle contractile length initial value
    
        # Set the time for integration
        t_start = 0.0
        t_stop = 0.2
        time_step = 0.001
        time = np.arange(t_start, t_stop, time_step)
        
        # Initialize empty arrays to record the force and length values for each stimulation
        active_forces = []
        passive_forces = []
        total_forces = []
        lengths = []
        
        # Run the experiment for each muscle stimulation and each muscle stretch
        for stretch in muscle_stretches:
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=muscle_stimulation,
                                   muscle_length=stretch)
            # Record the forces and length values at the last time point of the stimulation
            active_forces.append(result.active_force[-1])
            passive_forces.append(result.passive_force[-1])
            total_forces.append(result.active_force[-1]+result.passive_force[-1])
            lengths.append(result.l_ce[-1])
        
        # Plot the forces in function of the length
        plt.subplot(1, 2, i+1)
        plt.plot(lengths, active_forces)
        plt.plot(lengths, passive_forces)
        plt.plot(lengths, total_forces)
        plt.title('Muscle optimal fiber length = %.1f m' %(l_opt[i]))
        plt.xlabel('Muscle contractile length [m]')
        plt.ylabel('Muscle force [N]')
        plt.ylim(0,3000)
        plt.legend(('Active','Passive','Total'))    
        plt.grid()
    plt.show()   



def exercise1d():
    """ Exercise 1d
    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest. """

    # Definition of muscles
    muscle_parameters = MuscleParameters()
    pylog.info(muscle_parameters.showParameters())
    
    # Definition of the mass
    mass_parameters = MassParameters()
    pylog.info(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)
    # Create mass object
    mass = Mass(mass_parameters)

    # Instantiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)
    # Add the mass to the system
    sys.add_mass(mass)

    # Evaluate for several loads
    loads = 1./9.81*np.arange(10, 2*muscle.f_max, 100)
    # Evaluate for a single muscle stimulation
    muscle_stimulation = 1.0

    # Set the initial conditions
    x0 = [0.0, sys.muscle.l_opt, sys.muscle.l_opt + sys.muscle.l_slack, 0.0]
    # x0[0] --> activation
    # x0[1] --> contractile length (l_ce)
    # x0[2] --> position of the mass/load
    # x0[3] --> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.5
    time_step = 0.001
    time_stabilize = 0.05
    time = np.arange(t_start, t_stop, time_step)
    
    # Stores the maximal V_ce and the tendon force for each load
    max_V_ce = []
    tendon_F = []
    # Run the experiment for each load
    for load in loads:
        result = sys.integrate(x0=x0,
                                time=time,
                                time_step=time_step,
                                time_stabilize=time_stabilize,
                                stimulation=muscle_stimulation,
                                load=load) 
        # Record the maximal muscle contractile velocity and the tendon F
        max_vce_local = max_V_CE(result, muscle)
        max_V_ce.append(max_vce_local)
        tendon_F.append(result.tendon_force[-1])
    
    ####### TENSION ##########
    # Plot the tension in function of the maximal muscle contractile velocity
    plt.figure("Isotonic muscle experiment 1d")
    plt.plot(max_V_ce, 9.81*loads, color='blue', label = 'Tension')
    plt.xlabel('Muscle contractile velocity [lopts/s]')
    plt.ylabel('Tension, Tendon Force [N]')
   
    ####### TENDON FORCE ##########
    # Plot the tendon force in function of the maximal muscle contractile velocity
    plt.plot(max_V_ce, tendon_F, color='orange', label = 'Tendon force')
    plt.legend()
    plt.grid()
    
    # Show the shortening region on the plot
    # Draw horizontal line from (min(max_v_ce),300) to (0, 300)
    plt.text((min(max_V_ce)-0.2)/2, 1555, 'Shortening', color='green')
    plt.annotate("",
              xy=(min(max_V_ce), 1500), xycoords='data',
              xytext=(0, 1500), textcoords='data',
              arrowprops=dict(arrowstyle="-",
                              connectionstyle="arc3,rad=0.", color = 'green'))
    
    # Show the lengthening region on the plot
    # Draw horizontal line from (0, 300) to (max(max_v_ce),300)
    plt.text((max(max_V_ce)-0.2)/2, 1555, 'Lengthening', color ='red')
    plt.annotate("",
              xy=(0, 1500), xycoords='data',
              xytext=(max(max_V_ce), 1500), textcoords='data',
              arrowprops=dict(arrowstyle="-",
                              connectionstyle="arc3,rad=0.", color='red'))


def max_V_CE(result, muscle) : 
    
    # Return the maximum value of V_CE (maximum negative or maximum positive)
    # V_CE < 0 if shortening => take the min
    # V_CE > 0 if lengthening => take the max
    if result.l_mtu[-1] < (muscle.l_opt + muscle.l_slack):
        return min(result.v_ce) # shortening
    else:
        return max(result.v_ce) # lengthening
    
    
def exercise1f():
    """ Exercise 1f
    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest. """

    # Definition of muscles
    muscle_parameters = MuscleParameters()
    pylog.info(muscle_parameters.showParameters())
    # Definition of the mass
    mass_parameters = MassParameters()
    pylog.info(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)
    # Create mass object
    mass = Mass(mass_parameters)

    # Instantiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)
    # Add the mass to the system
    sys.add_mass(mass)

    # Evaluate for several loads
    loads = 1./9.81*np.arange(10, 2*muscle.f_max, 100)
    # Evaluate for several stimulations
    muscle_stimulations = np.arange(0,1.1,0.2)

    # Set the initial conditions
    x0 = [0.0, sys.muscle.l_opt, sys.muscle.l_opt + sys.muscle.l_slack, 0.0]
    # x0[0] --> activation
    # x0[1] --> contractile length (l_ce)
    # x0[2] --> position of the mass/load
    # x0[3] --> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.5
    time_step = 0.001
    time_stabilize = 0.05
    time = np.arange(t_start, t_stop, time_step)
    
    # Stores the maximal V_ce for each load
    Max_V_ce = []
    
    # Run the experiment for each muscle stimulation and each load
    for stimulation in muscle_stimulations:
        max_V_ce = []
        for load in loads:
            result = sys.integrate(x0=x0,
                                    time=time,
                                    time_step=time_step,
                                    time_stabilize=time_stabilize,
                                    stimulation=stimulation,
                                    load=load) 
            # Record the maximal muscle contractile velocity for each load
            max_vce_local = max_V_CE(result, muscle)
            max_V_ce.append(max_vce_local)
        # Record the maximal muscle contractile velocity for each stimulation
        Max_V_ce.append(max_V_ce)
    
    # Plot the tension in function of the maximal muscle contractile velocity for each stimulation
    plt.figure('Isotonic muscle experiment 1f')
    for i in range(len(muscle_stimulations)):
        plt.plot(Max_V_ce[i], 9.81*loads)
    plt.xlabel('Muscle contractile velocity [lopts/s]')
    plt.ylabel('Tension [N]')
    plt.legend(np.round(muscle_stimulations, 2), title='Muscle stimulation')
    plt.grid()



def exercise1():
    exercise1a()
    exercise1b()
    exercise1c()
    exercise1d()
    exercise1f()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

