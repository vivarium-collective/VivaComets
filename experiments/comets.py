"""
COMETS Vivarium Composite Simulation
"""

import numpy as np
from processes.diffusion_field import DiffusionField
from processes.spatial_dfba import SpatialDFBA
from plots.field import plot_objective_flux, plot_fields_temporal
from vivarium.core.engine import Engine


def test_comets():
    total_time = 10
    bounds = [5, 5]
    nbins = [5, 5]
    molecules = ['glucose', 'oxygen']
    species_info = [{
        "name": "ecoli",
        "model": '../data/iECW_1372.xml',
        "flux_id_map": {
            "glucose": "EX_glc__D_e",
            "oxygen": "EX_o2_e"
        },
        "kinetic_params": {
            "glucose": (0.4, 1.5),
            "oxygen": (0.25, 4.5)
        }
    }]

    # Parameters shared by both processes
    shared_params = {
        'bounds': bounds,
        'nbins': nbins,
        'molecules': molecules,
    }

    # create the two processes
    diffusion_field = DiffusionField(parameters=shared_params)
    spatial_dfba = SpatialDFBA(parameters={**shared_params, 'species_info': species_info})

    # set the initial state
    initial_state = {
        'fields': {mol: np.ones(nbins) * 5.0 for mol in molecules},
        'species': {'ecoli': np.ones(nbins) * 1.0},
    }

    # make the composite simulation and run it
    sim = Engine(
        processes={
            'diffusion_field': diffusion_field,
            'spatial_dfba': spatial_dfba
        },
        topology={
            'diffusion_field': {
                'fields': ('fields',),
                'species': ('species',),
            },
            'spatial_dfba': {
                'fields': ('fields',),
                'species': ('species',),
            }
        },
        initial_state=initial_state
    )
    sim.update(total_time)

    # retrieve the results and plot them
    data = sim.emitter.get_timeseries()
    desired_time_points = [1, 3, total_time-1]
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
        plot_fields=["glucose", "oxygen"],
        filename='comets_fields',
    )


if __name__ == '__main__':
    test_comets()
