"""Simulation for the amphibious arena"""

import numpy as np
from farms_bullet.model.model import (
    SimulationModels,
    GroundModel,
    DescriptionFormatModel,
)
from farms_bullet.simulation.options import SimulationOptions
from salamandra_simulation.simulation import simulation_setup
from salamandra_simulation.options import SalamandraOptions
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork
import farms_pylog as pylog


def simulation_amphibious(
        sim_parameters,
        drive_ground,
        drive_water,
        x_lim,
        **kwargs
):
    """Main"""
    # Simulation options
    arena = 'amphibious'
    pylog.info('Creating simulation')
    n_iterations = int(sim_parameters.duration/sim_parameters.timestep)
    simulation_options = SimulationOptions.with_clargs(
        timestep=sim_parameters.timestep,
        n_iterations=n_iterations,
        **kwargs,
    )

    # Arena
    water_surface = -0.1
    arena = SimulationModels([
        DescriptionFormatModel(
            path='arena_amphibious.sdf',
            visual_options={
                'path': 'BIOROB2_blue.png',
                'rgbaColor': [1, 1, 1, 1],
                'specularColor': [1, 1, 1],
            }
        ),
        DescriptionFormatModel(
            path='arena_water.sdf',
            spawn_options={
                'posObj': [0, 0, water_surface],
                'ornObj': [0, 0, 0, 1],
            }
        ),
    ])

    # Robot
    network = SalamandraNetwork(
        sim_parameters=sim_parameters,
        n_iterations=n_iterations
    )
    sim, data = simulation_setup(
        animat_sdf='salamandra_robotica.sdf',
        arena=arena,
        animat_options=SalamandraOptions.from_options(dict(
            water_surface=water_surface,
            spawn_position=sim_parameters.spawn_position,
            spawn_orientation=sim_parameters.spawn_orientation,
        )),
        simulation_options=simulation_options,
        network=network,
    )

    # Run simulation
    pylog.info('Running simulation')
    gps = data.sensors.gps
    for iteration in sim.iterator(show_progress=True):
        head_position = np.asarray(gps.urdf_position(iteration=iteration, link_i=0))
        if(x_lim < head_position[0]):
           sim_parameters.drive = drive_ground 
           network.robot_parameters.update(sim_parameters)
        else:
           sim_parameters.drive = drive_water
           network.robot_parameters.update(sim_parameters)
        assert iteration >= 0

    # Terminate simulation
    pylog.info('Terminating simulation')
    sim.end()
    return sim, data

