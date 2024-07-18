"""
COMETS Vivarium Composite Simulation
"""

import numpy as np
from processes.diffusion_field import DiffusionField
from processes.spatial_dfba import SpatialDFBA
from vivarium.core.engine import Engine

# Configuration
comets_config = {
    'total_time':10 ,
    'time_step': 1,
    'bounds': [10, 4],
    'nbins': [10, 4],
    'molecules': ['glucose', 'acetate', 'Maltose'],
    'species': ['Thermotoga', 'ecoli'],
}

# Parameters shared by both processes
shared_params = {
    'bounds': comets_config['bounds'],
    'nbins': comets_config['nbins'],
    'molecules': comets_config['molecules'],
    'species':comets_config['species']
}


# Specific parameters for Diffusion Field
diffusion_field_params = {
    **shared_params,
    'default_diffusion_dt': 0.001,
    'default_diffusion_rate': 2E-5,
    'diffusion': {
        'glucose': 2.0E-2,
        'acetate':2.0E-2,
        'Maltose':2.0E-2,
        'Thermotoga': 1.0E-2, 
        'ecoli': 1.0E-2 
    },
    'advection': {
        'glucose': (0.02, -0.03),
        'acetate': (0.02, -0.03),
        'Maltose': (0.02, -0.03),
        'Thermotoga': (0.02, -0.02), # advection vector 
        'ecoli': (0.02, -0.05)
    },
    'clamp_edges': {
        'glucose': 0.5, 
        'acetate': 0.5,
        'Maltose': 0.5,

    }
}

# Specific parameters for Spatial DFBA
spatial_dfba_params = {
    **shared_params,
    'species_info': [
        {
            'name': 'Thermotoga',
            'model': '../data/iLJ478.xml',  # Path to FBA model file 
            'flux_id_map': {
                'glucose': 'EX_glc__D_e',
                'acetate': "EX_ac_e",
                'Maltose': "EX_malt_e"
            },
            'kinetic_params': {
                'glucose': (0.1, 1),  # Km, Vmax for glucose
                'Maltose': (0.1, 0.8)
            },
            # "fixed_bounds": {
            #     'EX_cpd00058_e0': {'lower': -1, 'upper': 1},
            # }
        },
        {
            'name': 'ecoli',
            'model': '../data/iECW_1372.xml',  # Path to E. coli model file
            'flux_id_map': {
                'glucose': 'EX_glc__D_e',
                'acetate': 'EX_ac_e',       # Exchange reaction ID for acetate
                "Maltose": "EX_malt_e"
            },
            'kinetic_params': {
                'glucose': (0.1, 1.0)  #(0.4, 5),  # Km, Vmax for glucose
                # 'acetate': (0.1, 1.0),  #(0.4, 5),  # Km, Vmax for glucose
            },
            'fixed_bounds': {
                'EX_o2_e': {'lower': -2},  # Setting fixed bounds for E. coli
                'ATPM': {'lower': 1, 'upper': 1}  # Setting fixed bounds for E. coli
            #         'EX_fe3dhbzs_e': (0, 10)  # Setting fixed bounds for E. coli
            }
        }
    ]
    
}


# Initial state configuration
initial_field_config = {
    'uniform': {
        'glucose': 200.0,
        'Maltose': 200.0,
    }}
initial_species_config = {
    'uniform': {
        'species': {
            'ecoli': 0.5,
            'Thermotoga': 0.5
        }}}

def run_comets(comets_config, diffusion_field_params, spatial_dfba_params, initial_field_config, initial_species_config):
    # Create the two processes
    diffusion_field = DiffusionField(parameters=diffusion_field_params)
    spatial_dfba = SpatialDFBA(parameters=spatial_dfba_params)
    
    # Set the initial state for diffusion field
    initial_state_diffusion_field = diffusion_field.initial_state(initial_field_config)
    
    # Set the initial state for spatial dfba
    initial_species = spatial_dfba.initial_state(initial_species_config)

    # make top row 0 for the species
    for sid, array in initial_species['species'].items():
        initial_species['species'][sid][0,:] = 0

    
    # Merge the initial states
    initial_state = {
        'fields': initial_state_diffusion_field['fields'],
        'species': initial_species['species'],
    }
    
    # Make the composite simulation and run it
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
    
    # Run the simulation
    sim.update(comets_config['total_time'])
    
    # Retrieve the results
    data = sim.emitter.get_timeseries()
    
    return data


if __name__ == '__main__':
    run_comets(comets_config, diffusion_field_params, spatial_dfba_params, initial_field_config, initial_species_config)
