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
#from plots.field import plot_fields, plot_fields_temporal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


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
        'bounds': [3, 3, 3], # cm
        'nbins': [3, 3, 3],
        'molecules': ['glucose', 'oxygen'],
        'species': ["Alteromonas"],
        'default_diffusion_dt': 0.001,
        'default_diffusion_rate': 2E-5,  # cm^2/s, set to the highest diffusion coefficient (oxygen)
        'diffusion': {
            'glucose':2.0E-5,    # 6.7E-6,  # cm^2/s  TODO should find the current rate for cm^3
            'oxygen': 2.0E-5,   # cm^2/s
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.timestep_counter = 0 
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
        self.cubic_dict = {}

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
        schema = {
            'species': {},
            'fields': {}
        }
        for species in self.parameters['species']:
            schema['species'].update({
                species: {'_default': np.ones(self.nbins), '_updater': 'nonnegative_accumulate', '_emit': True}
            })
        for mol in self.parameters['molecules']:
            schema['fields'].update({
                mol: {'_default': np.ones(self.nbins), '_updater': 'nonnegative_accumulate', '_emit': True}
            })
        schema['dimensions'] = {'bounds': {'_value': self.bounds, '_updater': 'set', '_emit': True},
                                'nbins': {'_value': self.nbins, '_updater': 'set', '_emit': True}}
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        cubic_dict = {} 
        fields_new = copy.deepcopy(fields)
        current_timestep = self.timestep_counter
        self.cubic_dict[current_timestep] = {}
        for mol_id, field in fields.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            if np.var(field) > 0:  # If field is not uniform
                fields_new[mol_id] = self.diffuse(field, timestep, diffusion_rate)
        # Update cubic_dict 
        for x in range(self.nbins[0]):
            for y in range(self.nbins[1]):
                for z in range(self.nbins[2]):
                    cubic_address = (x, y, z)
                    oxygen_level = fields_new['oxygen'][x, y, z]
                    glucose_level = fields_new['glucose'][x, y, z]
                    self.cubic_dict[current_timestep][cubic_address] = {
                        'oxygen': oxygen_level,
                        'glucose': glucose_level,
                        'biomass': 'NA'  # Assuming biomass remains 'NA' for now
                    }

        delta_fields = {mol_id: fields_new[mol_id] - field for mol_id, field in fields.items()}
        
        self.timestep_counter += 1
        return {'fields': delta_fields}, cubic_dict
        
     

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
    
def plot_fields_temporal(fields_dict, desired_time_points, actual_time_points, z=5, out_dir="/Users/amin/Desktop/VivaComet/processes/out/", filename='fields_at_z'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    z_index = z - 1
    num_molecules = len(fields_dict.keys())
    
    # Map desired time points to indices in the actual data
    time_indices = [actual_time_points.index(time) for time in desired_time_points if time in actual_time_points]
    num_times = len(time_indices)
    if num_molecules > 1 or num_times > 1:
        fig, axs = plt.subplots(num_times, num_molecules, figsize=(10, num_times * 5), squeeze=False)
    else:
        fig, axs = plt.subplots(num_times, num_molecules, figsize=(10, 5))
        axs = np.array([[axs]])  # Make sure axs is 2D for consistency
    molecule_names = list(fields_dict.keys())
    for i, time_idx in enumerate(time_indices):
        for j, molecule in enumerate(molecule_names):
            data_array = np.array(fields_dict[molecule][time_idx])
            data = data_array[..., z_index]  # Use the numpy array here
            ax = axs[i, j]
            cax = ax.imshow(data, cmap='viridis', interpolation='nearest')
            if i == 0:
                ax.set_title(molecule, fontsize=24)  
            ax.set_ylabel(f"Time {actual_time_points[time_idx]}", fontsize=22)  
            ax.set_xticks(np.arange(data.shape[1]), minor=False) 
            ax.set_yticks(np.arange(data.shape[0]), minor=False)  # Adjust y-ticks
            ax.set_xticklabels(np.arange(1, data.shape[1]+1))  # Adjust x-tick labels
            ax.set_yticklabels(np.arange(1, data.shape[0]+1)) 
            ax.tick_params(axis='both', which='major', labelsize=20)  
            if j == num_molecules - 1 and i == 0:
                cb = fig.colorbar(cax, ax=ax)
                cb.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}.png")


#  3D test_field
def test_fields():
    total_time = 12
    config = {
        "bounds": [10, 10, 10],
        "nbins": [10, 10, 10],
        "molecules": ["glucose", "oxygen"]
    }
    field = DiffusionField(config)
    initial_state = field.initial_state({'random': 1.0})
    sim = Engine(
        initial_state=initial_state,
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
    first_oxygen_data = data['fields']['oxygen'][0] if isinstance(data['fields']['oxygen'], list) else None

    # Plot the results
    first_fields = {key: matrix[0] for key, matrix in data['fields'].items()}
    time_list=[0,1,2, 50, 100]
    # Inside test_fields function
    actual_time_points = data['time']  # Extract actual time points from data
    plot_fields_temporal(data['fields'], time_list, actual_time_points, z=5, out_dir="/Users/amin/Desktop/VivaComet/processes/out", filename='fields_over_time')
    print(field.cubic_dict[0]keys())

if __name__ == '__main__':
    test_fields()
