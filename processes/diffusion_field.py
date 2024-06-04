"""
===============
Diffusion Field 
===============

Diffuses and decays molecular concentrations in a 2D field.
"""

import copy
import numpy as np
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from scipy.ndimage import convolve
import os

from plots.field import plot_fields_temporal

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


def get_bin_site(location, n_bins, bounds):
    bin_site_no_rounding = np.array([
        location[0] * n_bins[0] / bounds[0], 
        location[1] * n_bins[1] / bounds[1]
    ])
    bin_site = tuple(np.floor(bin_site_no_rounding).astype(int) % n_bins)
    return bin_site  # address of the bin


def get_bin_volume(bin_size, depth=1):
    dimensions = bin_size + [depth]
    return np.prod(dimensions)


class DiffusionField(Process):
    """
    DiffusionField Process
    """
    defaults = {
        'bounds': [20, 20],
        'nbins': [20, 20],
        "depth": 1, #TODO make this adjustable
        'molecules': ['glucose', 'oxygen'],
        'species': ['Alteromonas', 'ecoli'],
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

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.species_ids = self.parameters['species']
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters['nbins']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        assert len(self.bounds) == 2, "Process only supports 2D"
        diffusion_rate = self.parameters['default_diffusion_rate']
        dx, dy = self.bin_size  
        dx2_dy2 = get_bin_volume(self.bin_size)
        self.diffusion_rate = diffusion_rate / dx2_dy2
        self.molecule_specific_diffusion = {
            mol_id: diff_rate / dx2_dy2
            for mol_id, diff_rate in self.parameters['diffusion'].items()
        }
        self.advection_vectors = {
            mol_id: self.parameters['advection'].get(mol_id, (0, 0))
            for mol_id in self.molecule_ids
        }

        diffusion_dt = 0.5 * min(dx**2, dy**2) / (2 * diffusion_rate)
        self.diffusion_dt = min(diffusion_dt, self.parameters['default_diffusion_dt'])
        self.bin_volume = get_bin_volume(self.bin_size, self.parameters["depth"])


    def initial_state(self, config=None):
        """get initial state of the fields
        Args:
            * config (dict): with optional keys "random" or "uniform".
                * "random" key maps to a maximum value for the field, which gets filled with values between [0, max].
                * "uniform" key maps to a value that will fill the entire field
        Returns:
            * fields (dict) with {mol_id: 3D np.array}
        """
        if config is None:
            config = {}
        fields = {}
        for mol in self.parameters['molecules']:
            shape = tuple(self.nbins)
            if 'random' in config:
                random_config = config['random']
                if isinstance(random_config, dict):
                    # If config['random'] is a dictionary, get the value for the current molecule
                    max_value = random_config.get(mol, 1)
                else:
                    # If config['random'] is directly a float (or int), use it as the max value for all molecules
                    max_value = random_config
                field = np.random.rand(*shape) * max_value
            elif 'uniform' in config:
                value = config.get('uniform', 1)
                field = np.ones(shape) * value
            else:
                field = np.ones(shape)
            fields[mol] = field
        species = {}
        for spec in self.parameters['species']:
            shape = tuple(self.nbins)
            if 'random' in config:
                random_config = config['random']
                if isinstance(random_config, dict):
                    max_value = random_config.get(spec, 1)
                else:
                    max_value = random_config
                field = np.random.rand(*shape) * max_value
            elif 'uniform' in config:
                value = config.get('uniform', 1)
                field = np.ones(shape) * value
            else:
                field = np.ones(shape)
            species[spec] = field
        return {'fields': fields, 'species': species}

    def ports_schema(self):
        schema = {
            'species': {},
            'fields': {},
        }
        for species in self.parameters['species']:
            schema['species'].update({
                species: {
                    '_default': np.ones(self.nbins),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True
                }
            })
        for mol in self.parameters['molecules']:
            schema['fields'].update({
                mol: {
                    '_default': np.ones(self.nbins),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True
                }
            })
        schema['dimensions'] = {
            'bounds': {
                '_value': self.bounds,
                '_updater': 'set',
                '_emit': True
            },
            'nbins': {
                '_value': self.nbins,
                '_updater': 'set',
                '_emit': True
            }
        }
        return schema

    def next_update(self, timestep, states):
        combined_dict = {**states['fields'], **states['species']}
        combined_new = copy.deepcopy(combined_dict)
        for mol_id, field in combined_dict.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            advection_vector = self.parameters['advection'].get(mol_id, (0, 0))
            if np.var(field) > 0:  # If field is not uniform
                combined_new[mol_id] = self.diffuse(field, timestep, diffusion_rate, advection_vector)
        delta_fields = {mol_id: combined_new[mol_id] - field for mol_id, field in states['fields'].items()}
        delta_species = {spec_id: combined_new[spec_id] - field for spec_id, field in states['species'].items()}
        return {'fields': delta_fields, 'species': delta_species}

    def get_bin_site(self, location):
        return get_bin_site(
            [loc for loc in location],
            self.nbins,
            self.bounds)

    def ones_field(self):
        return np.ones(self.nbins, dtype=np.float64)

    def random_field(self):
        return np.random.rand(*self.nbins)

    def diffuse(self, field, timestep, diffusion_rate, advection_vector):
        if field.ndim == 2:    
            laplacian_kernel = np.array([[0,  1, 0],
                                         [1, -4, 1],
                                         [0,  1, 0]])
            gradient_x_kernel = np.array([[-1, 0, 1]]) / 2.0
            gradient_y_kernel = np.array([[-1], [0], [1]]) / 2.0
            
        else:
            raise ValueError('Field must be 1D, 2D, or 3D')
        t = 0.0
        dt = min(timestep, self.diffusion_dt)
        while t < timestep:
            laplacian = convolve(field, laplacian_kernel, mode='reflect') * diffusion_rate   #TODO put bondary 
            grad_x = convolve(field, gradient_x_kernel, mode='reflect') * advection_vector[0]
            grad_y = convolve(field, gradient_y_kernel, mode='reflect') * advection_vector[1]
            field += dt * (laplacian - grad_x - grad_y)
            t += dt
        return field


def test_fields():
    total_time = 200
    config = {
        'bounds': [20, 20],
        'nbins': [20, 20],
        'molecules': ['glucose', 'oxygen'],
        'species': ['Alteromonas', 'ecoli'],
        'diffusion': {
            'glucose': 6.7E-1,     # 6.7E-6,  # cm^2/s
            'oxygen':  2.0E-2,     # 2.0E-5,  # cm^2/s
        },
        'advection': {
            'glucose': (0.01, 0.02),  # Advection vector for glucose
            'oxygen': (0.01, 0.01),   # Advection vector for oxygen
        }
    }

    # create the process and make a simulation
    diffusion_field = DiffusionField(config)
    initial_state = diffusion_field.initial_state({'random': 1.0})
    sim = Engine(
        initial_state=initial_state,
        processes={'diffusion_process': diffusion_field},
        topology={'diffusion_process': {
            'fields': ('fields',),
            'species': ('species',),
            'dimensions': ('dimensions',),
        }}
    )
    # Run the simulation
    sim.update(total_time)
    # Get the results
    data = sim.emitter.get_timeseries()
    time_list = [0, 1, int(total_time/4), int(total_time/2), total_time]

    # plot the results
    plot_fields_temporal(
        fields_data=data['fields'],
        desired_time_points=time_list,
        actual_time_points=data['time'],
        out_dir='./out',
        filename='Diffusion_test')

    # plot the species
    plot_fields_temporal(
        fields_data=data['species'],
        desired_time_points=time_list,
        actual_time_points=data['time'],
        out_dir='./out',
        filename='Species_test')


if __name__ == '__main__':
    test_fields()
