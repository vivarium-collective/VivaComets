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
    volume = np.prod(dimensions)
    assert volume > 0, f'Volume of bin is {volume}'
    return volume


class DiffusionField(Process):
    """
    DiffusionField Process
    """
    defaults = {
        'bounds': [10, 4],
        'nbins': [10, 4],
        'depth': 1,
        'molecules': [],  # ['glucose', 'oxygen'],
        'species': [],  # ['Alteromonas', 'ecoli'],
        'default_diffusion_dt': 0.001,
        'default_diffusion_rate': 2E-5,
        'diffusion': {
            # 'glucose': 6.7E-1,
            # 'oxygen': 2.0E-2,
            # 'Alteromonas': 1.0E-2,
            # 'ecoli': 1.0E-2
        },
        'advection': {
            # 'glucose': (0.0, 0.0),
            # 'oxygen': (0.01, 0.01),
            # 'Alteromonas': (0.01, 0.01),
            # 'ecoli': (0.01, 0.01)
        },
        'clamp_edges': {

        },
        'initial_edge_values': {
            # 'glucose': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
            # 'ecoli': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
        },
        'constant_edges': False,  # determine if initial edges remain constant during the simulation
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
        self.constant_edges = self.parameters.get('constant_edges', False)  # Get the constant_edges parameter

        self.advection_vectors = {
            mol_id: self.parameters['advection'].get(mol_id, (0, 0))
            for mol_id in self.molecule_ids + self.species_ids
        }


        diffusion_dt = 0.5 * min(dx**2, dy**2) / (2 * diffusion_rate)
        self.diffusion_dt = min(diffusion_dt, self.parameters['default_diffusion_dt'])
        self.bin_volume = get_bin_volume(self.bin_size, self.parameters["depth"])

        self.clamp_edges = self.parameters['clamp_edges']
        self.initial_edge_values = self.parameters['initial_edge_values']
        # Check that edge clamp values are provided for all molecules
        if isinstance(self.parameters['clamp_edges'], dict):
            for key in self.parameters['clamp_edges'].keys():
                assert (key in self.molecule_ids or key in self.species_ids), f'clamp edge key {key} not in molecules or species'

    def apply_initial_edge_values(self, field, edge_values):
        if 'top' in edge_values:
            field[0, :] = edge_values['top']
        if 'bottom' in edge_values:
            field[-1, :] = edge_values['bottom']
        if 'left' in edge_values:
            field[:, 0] = edge_values['left']
        if 'right' in edge_values:
            field[:, -1] = edge_values['right']
        return field
    
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
                    max_value = random_config.get(mol, 1)
                else:
                    max_value = random_config
                field = np.random.rand(*shape) * max_value
            elif 'uniform' in config:
                uniform_config = config['uniform']
                if isinstance(uniform_config, dict):
                    value = uniform_config.get(mol, 1)
                else:
                    value = uniform_config
                field = np.ones(shape) * value
            else:
                field = np.ones(shape)
            if mol in self.initial_edge_values:
                field = self.apply_initial_edge_values(field, self.initial_edge_values[mol])
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
                uniform_config = config['uniform']
                if isinstance(uniform_config, dict):
                    value = uniform_config.get(spec, 1)
                else:
                    value = uniform_config
                field = np.ones(shape) * value
            else:
                field = np.ones(shape)
            if spec in self.initial_edge_values:
                field = self.apply_initial_edge_values(field, self.initial_edge_values[spec])
            species[spec] = field
        return {'fields': fields, 'species': species}

    def ports_schema(self):
        schema = {
            'species': {},
            'fields': {},
            'dimensions': {
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
        }

        # fill in the species
        for species in self.parameters['species']:
            schema['species'].update({
                species: {
                    '_default': np.ones(self.nbins),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True
                }})

        # fill in the fields
        for mol in self.parameters['molecules']:
            schema['fields'].update({
                mol: {
                    '_default': np.ones(self.nbins),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True
                }
            })
        return schema

    def next_update(self, timestep, states):
        # get the input states
        combined_dict = {**states['fields'], **states['species']}
        combined_new = copy.deepcopy(combined_dict)

        # update the fields and species
        for mol_id, field in combined_dict.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            advection_vector = self.advection_vectors.get(mol_id, (0, 0))

            # Get the clamp value for the molecule
            clamp_value = self.parameters['clamp_edges'].get(mol_id, 0.0)

            # Apply diffusion if the field is not uniform
            if np.var(field) > 0:
                field = self.diffuse(field, timestep, diffusion_rate, constant_value=clamp_value)

            # Always apply advection
            field = self.advect(field, timestep, advection_vector, constant_value=clamp_value)
            # Apply constant edge values if constant_edges is True
            if self.constant_edges and mol_id in self.initial_edge_values:
                field = self.apply_initial_edge_values(field, self.initial_edge_values[mol_id])

            combined_new[mol_id] = field

        # get deltas for fields and species
        delta_fields = {
            mol_id: combined_new[mol_id] - field
            for mol_id, field in states['fields'].items()
        }
        delta_species = {
            spec_id: combined_new[spec_id] - field
            for spec_id, field in states['species'].items()
        }


        #return the update
        return {
            'fields': delta_fields,
            'species': delta_species
        }

    def diffuse(self, field, timestep, diffusion_rate, constant_value=None):
        if field.ndim == 2:    
            laplacian_kernel = np.array([[0,  1, 0],
                                         [1, -4, 1],
                                         [0,  1, 0]])
        else:
            raise ValueError('Field must be 1D, 2D, or 3D')

        t = 0.0
        dt = min(timestep, self.diffusion_dt)
        while t < timestep:
            if constant_value is not None:
                laplacian = convolve(field, laplacian_kernel, mode='constant', cval=constant_value) * diffusion_rate
            else:
                laplacian = convolve(field, laplacian_kernel, mode='nearest') * diffusion_rate

            field += dt * laplacian
            t += dt
        return field
    
    
    def advect(self, field, timestep, advection_vector, constant_value=None):
        gradient_x_kernel = np.array([[-1, 0, 1]]) / 2.0
        gradient_y_kernel = np.array([[-1], [0], [1]]) / 2.0
        t = 0.0
        dt = min(timestep, self.diffusion_dt)
        while t < timestep:
            if constant_value is not None:
                grad_x = convolve(field, gradient_x_kernel, mode='nearest', cval=constant_value) * advection_vector[0]
                grad_y = convolve(field, gradient_y_kernel, mode='nearest', cval=constant_value) * advection_vector[1]
            else:
                grad_x = convolve(field, gradient_x_kernel, mode='nearest') * advection_vector[0]
                grad_y = convolve(field, gradient_y_kernel, mode='nearest') * advection_vector[1]

            field -= dt * (grad_x + grad_y)
            t += dt
        return field


    def get_bin_site(self, location):
        return get_bin_site(
            [loc for loc in location],
            self.nbins,
            self.bounds)

    def ones_field(self):
        return np.ones(self.nbins, dtype=np.float64)

    def random_field(self):
        return np.random.rand(*self.nbins)

def test_fields():
    total_time = 20
    config = {
        'bounds': [20, 20],
        'nbins': [20, 20],
        'molecules': ['glucose', 'Maltose'],
        'species': ['Alteromonas', 'ecoli'],
        'diffusion': {
            'glucose': 2.0E-2 ,    # cm^2/s
            'Maltose':  6.7E-1,     # cm^2/s
            'Alteromonas': 1.0E-2, # diffusion rate for Alteromonas
            'ecoli': 1.0E-2        # diffusion rate for ecoli
            
        },
        'advection': {
            'glucose': (0.01, 0.01),     # Advection vector for glucose (x, y)
            'Maltose': (0.01, 0.01),      # Advection vector for oxygen (x, y)
            'Alteromonas': (0.0, -0.05), # advection vector for Alteromonas (x, y)
            'ecoli': (0.0, -0.05)        # advection vector for ecoli
            
        },
        'clamp_edges': {
            'glucose': 0, 
            'Maltose': 0,
            'Alteromonas': 0.0,
            'ecoli': 0.0,
        },
        'initial_edge_values': {
            'glucose': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
            'ecoli': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0},
        },
        'constant_edges': True,  # Set this to True or False for constant edges

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
