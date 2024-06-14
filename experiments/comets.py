"""
COMETS Vivarium Composite Simulation
"""

import numpy as np
from processes.diffusion_field import DiffusionField
from processes.spatial_dfba import SpatialDFBA
from plots.field import plot_objective_flux, plot_fields_temporal
from vivarium.core.engine import Engine

# Define comets_config 
comets_config = {
    'total_time': 10,
    'bounds': [5, 5],
    'nbins': [2, 2],
}

# Parameters shared by both processes
shared_params = {
    'bounds': comets_config['bounds'],
    'nbins': comets_config['nbins'],
    'molecules': ['glucose', 'oxygen'],
}

species_info = [
    {
        "name": "Alteromonas",
        "model": '../data/Alteromonas_Model.xml',
        "flux_id_map": {
            "glucose": "EX_cpd00027_e0",
            "oxygen": "EX_cpd00007_e0"
        },
        "kinetic_params": {
            "glucose": (0.5, 0.0005),
            "oxygen": (0.3, 0.0005),
        },
        "fixed_bounds": {
            'EX_cpd00149_e0': (-10, 10)
        }
    },
    {
        "name": "ecoli",
        "model": '../data/iECW_1372.xml',
        "flux_id_map": {
            "glucose": "EX_glc__D_e",
            "oxygen": "EX_o2_e"
        },
        "kinetic_params": {
            "glucose": (0.4, 0.6),
            "oxygen": (0.25, 0.6),
        },
        "fixed_bounds": {
            'EX_fe3dhbzs_e': (0, 10)
        }
    }
]

# Specific parameters for Diffusion Field
diffusion_field_params = {
    **shared_params,
    'default_diffusion_dt': 0.001,
    'default_diffusion_rate': 2E-5,
    'diffusion': {
        'glucose': 6.7E-1,
        'oxygen': 2.0E-2,
    },
    'advection': {
        'glucose': (0.01, 0.02),
        'oxygen': (0.01, 0.01),
    }
}

# Specific parameters for Spatial DFBA
spatial_dfba_params = {
    **shared_params,
    'species_info': species_info
}

initial_state_config = {
    'random': {
        'glucose': 200.0,  # Max random value for glucose
        'oxygen': 200.0,   # Max random value for oxygen
        'species': {
            'ecoli': 0.5,   # Max random value for E. coli biomass
            'Alteromonas': 0.5   # Max random value for Alteromonas biomass
        }
    }
}

def run_comets(comets_config, diffusion_field_params, spatial_dfba_params, initial_state_config):

    # create the two processes
    diffusion_field = DiffusionField(parameters=diffusion_field_params)
    spatial_dfba = SpatialDFBA(parameters=spatial_dfba_params)

    # set the initial state for diffusion field
    initial_state_diffusion_field = diffusion_field.initial_state({'random': 1.0})

    # set the initial state for spatial dfba
    initial_state_spatial_dfba = spatial_dfba.initial_state(initial_state_config)

    # Merge the initial states
    initial_state = {
        'fields': initial_state_diffusion_field['fields'],
        'species': initial_state_spatial_dfba['species'],
    }

    # make the composite simulation and run it
    sim = Engine(
        processes={
            'diffusion_process': diffusion_field,
            'fba_process': spatial_dfba
        },
        topology={
            'diffusion_process': {
                'fields': ('fields',),
                'species': ('species',),
                'dimensions': ('dimensions',),
            },
            'fba_process': {
                'fields': ('fields',),
                'species': ('species',),
                'exchange_fluxes': ('exchange_fluxes',),
                'dimensions': ('dimensions',),
            }
        },
        initial_state=initial_state
    )
    sim.update(comets_config['total_time'])

    # retrieve the results and plot them
    data = sim.emitter.get_timeseries()
    desired_time_points = [1, 3, comets_config['total_time'] - 1]

    plot_objective_flux(
        data,
        time_points=desired_time_points,
        species_names=[species['name'] for species in species_info],
        out_dir='./out',
        filename='Comets_objective_flux_plot'
    )

    plot_fields_temporal(
        data['fields'],
        desired_time_points=desired_time_points,
        actual_time_points=data['time'],
        plot_fields=["glucose", "oxygen", "ecoli", "Alteromonas"],
        molecule_colormaps={"glucose": "Blues", "oxygen": "Greens", "ecoli": "Purples", "Alteromonas": "Oranges"},
        filename='comets_fields',
    )


if __name__ == '__main__':
    run_comets(comets_config, diffusion_field_params, spatial_dfba_params, initial_state_config)
