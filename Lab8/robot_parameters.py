"""Robot parameters"""

import numpy as np
import farms_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)
        

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        freqs = np.ones(self.n_oscillators)
        
        body_freq = 0
        limbs_freq = 0
        d_low = 1.0
        d_high_body = 5.0
        d_high_limbs = 3.0
        
        # We want to set the intrinsic frequencies to 1 Hz for ex 8c
        if parameters.freqs is not None:
            if parameters.drive >= d_low and parameters.drive <= d_high_body:
                body_freq = parameters.freqs
                
            if parameters.drive >= d_low and parameters.drive <= d_high_limbs:
                limbs_freq = parameters.freqs
        # Normal case
        else :
            if parameters.drive >= d_low and parameters.drive <= d_high_body:
                body_freq = 0.2 * parameters.drive + 0.3
            
            if parameters.drive >= d_low and parameters.drive <= d_high_limbs:
                limbs_freq = 0.2 * parameters.drive
        
        freqs[:20] = body_freq
        freqs[20:] = limbs_freq
        
        self.freqs = freqs

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        weights_matrix = np.zeros([24,24])
        
        #upwards in body CPG
        np.fill_diagonal(weights_matrix[1:20], 10)
        #downwards in body CPG
        np.fill_diagonal(weights_matrix[:,1:20], 10)
        #the oscillators 10 and 11 are not coupled in either direction
        weights_matrix[9,10] = 0
        weights_matrix[10,9] = 0
        
        #from right to left in body CPG
        np.fill_diagonal(weights_matrix[10:20], 10)
        #from left to right in body CPG
        np.fill_diagonal(weights_matrix[:,10:20], 10)
        
        #whithin the limb CPG
        weights_matrix[20,21:23] = 10
        weights_matrix[21:23,20] = 10
        weights_matrix[23,21:23] = 10
        weights_matrix[21:23,23] = 10
        
        #from limb to body CPG
        weights_matrix[1:5,20] = 30
        weights_matrix[11:15,21] = 30
        weights_matrix[5:10,22] = 30
        weights_matrix[15:20,23] = 30
        
        self.coupling_weights = weights_matrix

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        bias_matrix = np.zeros([24,24])

        #downwards in body CPG
        np.fill_diagonal(bias_matrix[1:20], parameters.phase_lag)
        # upwards in body CPG
        np.fill_diagonal(bias_matrix[:,1:20], -parameters.phase_lag)
        #the oscillators 10 and 11 are not coupled in either direction
        bias_matrix[9,10] = 0
        bias_matrix[10,9] = 0
        
        #from right to left in body CPG
        np.fill_diagonal(bias_matrix[10:20], np.pi)
        #from left to right in body CPG
        np.fill_diagonal(bias_matrix[:,10:20], np.pi)

        #whithin the limb CPG
        bias_matrix[20,21:23] = np.pi
        bias_matrix[21:23,20] = np.pi
        bias_matrix[23,21:23] = np.pi
        bias_matrix[21:23,23] = np.pi
        
        #from limb to body CPG
        bias_matrix[1:5,20] = parameters.offset
        bias_matrix[11:15,21] = parameters.offset
        bias_matrix[5:10,22] = parameters.offset
        bias_matrix[15:20,23] = parameters.offset
        
        self.phase_bias = bias_matrix

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates = self.n_oscillators * np.ones(self.n_oscillators)

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        nominal_amplitudes = np.zeros(self.n_oscillators)
        
        body_amp = 0
        limbs_amp = 0
        
        if parameters.nominal_amp is not None:
            body_amp = parameters.nominal_amp
            limbs_amp = parameters.nominal_amp
        
        else:
            d_low = 1.0
            d_high_body = 5.0
            d_high_limbs = 3.0
            
            if parameters.drive >= d_low and parameters.drive <= d_high_body:
                body_amp = 0.065 * parameters.drive + 0.196
            
            if parameters.drive >= d_low and parameters.drive <= d_high_limbs:
                limbs_amp = 0.131 * parameters.drive + 0.131
            
        # parameters.turn = [f_left, f_right]
        body_amp_left, body_amp_right = body_amp, body_amp 
        body_amp_left *= parameters.turn[0]
        body_amp_right *= parameters.turn[1]
        
        # parameters.amplitude_gradient = [Rhead, Rtail]
        if parameters.amplitude_gradient is not None:
            R_gradient = np.linspace(parameters.amplitude_gradient[0], parameters.amplitude_gradient[1], self.n_body_joints)
        else:
            R_gradient = 1
        
        nominal_amplitudes[:self.n_body_joints] = body_amp_left*R_gradient
        nominal_amplitudes[self.n_body_joints:2*self.n_body_joints] = body_amp_right*R_gradient
        nominal_amplitudes[2*self.n_body_joints:] = limbs_amp
        
        self.nominal_amplitudes = nominal_amplitudes