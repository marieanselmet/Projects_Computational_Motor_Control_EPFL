"""Simulation parameters"""

import numpy as np

class SimulationParameters(dict):
    """Simulation parameters"""
    

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.duration = 10
        self.timestep = 1e-2
        self.phase_lag = 2*np.pi/10
        self.offset = np.pi
        self.amplitudes = [1, 2, 3]
        self.freqs = None # intrinsic frequencies
        self.nominal_amp = None
        self.drive = 3.5
        self.amplitude_gradient = None # [Rhead, Rtail]
        self.turn = [1, 1] # [f_left, f_right]
        self.spawn_position = [0, 0, 0.1] # Robot position in [m]
        self.spawn_orientation = [0, 0, 0] # Orientation in Euler angles [rad]
        
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations
        if 'duration' in kwargs:
            self.duration = kwargs['duration']
        if 'drive' in kwargs:
            self.drive = kwargs['drive']
        if 'phase_lag' in kwargs:
            self.phase_lag = kwargs['phase_lag']
        if 'offset' in kwargs:
            self.offset = kwargs['offset']
        if 'nominal_amp' in kwargs:
            self.nominal_amp = kwargs['nominal_amp']
        if 'amplitude_gradient' in kwargs:
            self.amplitude_gradient = kwargs['amplitude_gradient']
        if 'turn' in kwargs:
            self.turn = kwargs['turn']
        if 'phase_lag' in kwargs:
            self.phase_lag = kwargs['phase_lag']
        if 'spawn_position' in kwargs:
            self.spawn_position = kwargs['spawn_position']
        if 'spawn_orientation' in kwargs:
            self.spawn_orientation = kwargs['spawn_orientation']
        if 'freqs' in kwargs:
            self.freqs = kwargs['freqs']
            