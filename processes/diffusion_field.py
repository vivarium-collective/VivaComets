"""
===============
Diffusion Field 
===============

Diffuses and decays molecular concentrations in a 3D field.
"""

import copy
import numpy as np
from vivarium.core.serialize import Quantity
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from scipy.ndimage import convolve
from plots.field import plot_fields, plot_fields_temporal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



LAPLACIAN_3D = np.array([[[0, 0, 0],   [0, 1/6, 0],      [0, 0, 0]],
                         [[0, 1/6, 0], [1/6, -6/6, 1/6], [0, 1/6, 0]],
                         [[0, 0, 0],   [0, 1/6, 0],      [0, 0, 0]]])

def get_bin_site(location, n_bins, bounds): #compute the relative position of the point within the bounds. 
    bin_site_no_rounding = np.array([
        location[0] * n_bins[0] / bounds[0], 
        location[1] * n_bins[1] / bounds[1],
        location[2] * n_bins[2] / bounds[2]  
    ])
    bin_site = tuple(np.floor(bin_site_no_rounding).astype(int) % n_bins)
    return bin_site #address of the bin

def get_bin_volume(bin_size):
    total_volume = np.prod(bin_size) 
    return total_volume

class DiffusionField(Process):
    defaults = {
        'bounds': [10, 10, 10], # cm
        'nbins': [10, 10, 10],
        'molecules': ['glucose', 'oxygen'],
        'species': ["Alteromonas"],
        'default_diffusion_dt': 0.001,
        'default_diffusion_rate': 2E-5,  # cm^2/s, set to the highest diffusion coefficient (oxygen)
        'diffusion': {
            'glucose': 6.7E-6,  # cm^2/s  TODO should find the current rate for cm^3
            'oxygen': 2.0E-5,   # cm^2/s
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters["nbins"]
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        diffusion_rate = self.parameters['default_diffusion_rate']
        dx, dy, dz = self.bin_size  
        dx2_dy2_dz2 = get_bin_volume(self.bin_size)
        self.diffusion_rate = diffusion_rate / dx2_dy2_dz2
        self.molecule_specific_diffusion = {
            mol_id: diff_rate / dx2_dy2_dz2
            for mol_id, diff_rate in self.parameters['diffusion'].items()
        }

        diffusion_dt = 0.5 * min(dx**2, dy**2, dz**2) / (2 * diffusion_rate)
        self.diffusion_dt = min(diffusion_dt, self.parameters['default_diffusion_dt'])
        self.bin_volume = get_bin_volume(self.bin_size)


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
        return {'fields': fields, 'species': {}}


    def ports_schema(self):
        schema = {}
        for species in self.parameters['species']:
            schema['species'] = {
                species: {'_default': np.ones(self.nbins), '_updater': 'nonnegative_accumulate', '_emit': True}
            }
        for mol in self.parameters['molecules']:
            schema['fields'] = {
                mol: {'_default': np.ones(self.nbins), '_updater': 'nonnegative_accumulate', '_emit': True}
            }
        schema['dimensions'] = {'bounds': {'_value': self.bounds, '_updater': 'set', '_emit': True},
                                'nbins': {'_value': self.nbins, '_updater': 'set', '_emit': True}}
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        fields_new = copy.deepcopy(fields)
        for mol_id, field in fields.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            if np.var(field) > 0:  # If field is not uniform
                fields_new[mol_id] = self.diffuse(field, timestep, diffusion_rate)
        delta_fields = {mol_id: fields_new[mol_id] - field for mol_id, field in fields.items()}
        return {'fields': delta_fields}

    def get_bin_site(self, location):
        return get_bin_site(
            [loc for loc in location],
            self.nbins,
            self.bounds)

    def ones_field(self):
        return np.ones(self.nbins, dtype=np.float64)

    def random_field(self):
        return np.random.rand(*self.nbins)

    def diffuse(self, field, timestep, diffusion_rate):
        t = 0.0
        dt = min(timestep, self.diffusion_dt)
        while t < timestep:
            result = convolve(field, LAPLACIAN_3D, mode='constant', cval=0.0) #Now it works for 3D
            field += diffusion_rate * dt * result
            t += dt
        return field

    def diffuse_fields(self, fields, timestep):  #actual computing of diffusing of each molecule 
        """ diffuse fields in a fields dictionary """
        for mol_id, field in fields.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            # run diffusion if molecule field is not uniform
            if len(set(field.flatten())) != 1:  
                fields[mol_id] = self.diffuse(field, timestep, diffusion_rate)
        return fields


#  3D test_field
def test_fields():
    total_time = 30
    config = {
        "bounds": [10, 10, 10],
        "nbins": [10, 10, 10],
        "molecules": ["glucose", "oxygen"]
    }
    field = DiffusionField(config)

    sim = Engine(
        initial_state=field.initial_state({'random': 1.0}),
        processes={'diffusion_process': field},
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
    print(type(data['fields']['oxygen']))  # oxygen data type
    first_oxygen_data = data['fields']['oxygen'][0] if isinstance(data['fields']['oxygen'], list) else None
    print(type(first_oxygen_data))  # to see if its a list
    if isinstance(first_oxygen_data, np.ndarray):
        print(first_oxygen_data.shape)  # check the shape

    # Plot the results
    first_fields = {key: matrix[0] for key, matrix in data['fields'].items()}
    plot_fields(first_fields, out_dir='out', filename='initial_fields')

    plot_fields_temporal(data['fields'],data['species'], nth_timestep=5, out_dir='out', filename='fields_over_time')

if __name__ == '__main__':
    test_fields()
