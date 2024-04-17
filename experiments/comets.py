import sys
sys.path.append('/Users/amin/Desktop/VivaComets')
import numpy as np
from processes.diffusion_field import DiffusionField
from processes.spatial_dfba import SpatialDFBA
from vivarium.core.engine import Engine



def test_comets():
  
    bounds = [3, 3]
    nbins = [3, 3]  
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

    # processes
    diffusion_field = DiffusionField(parameters=shared_params)
    spatial_dfba = SpatialDFBA(parameters={**shared_params, 'species_info': species_info})

    initial_state = {
        'fields': {mol: np.ones(nbins) * 5.0 for mol in molecules},
        'species': {'ecoli': np.ones(nbins) * 1.0},
    }


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

   
    total_time = 10 
    sim.update(total_time)

    data = sim.emitter.get_timeseries()

if __name__ == '__main__':
    test_comets()
