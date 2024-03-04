"""
===============
Diffusion Field 
===============

Diffuses and decays molecular concentrations in a 3D field.
"""

import copy
import cv2
import numpy as np
from scipy import constants
from vivarium.core.serialize import Quantity
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from scipy.ndimage import convolve
from plots.field import plot_fields, plot_fields_temporal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




# laplacian kernel for diffusion
LAPLACIAN_3D = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                         [[1, 1, 1], [1, -6, 1], [1, 1, 1]],
                         [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])

def get_bin_site(location, n_bins, bounds):
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
        'bounds': [10, 10, 10],
        'nbins': [10, 10, 10],
        'molecules': ['glucose', 'oxygen'],
        'species': ["Alteromonas"],
        'default_diffusion_dt': 0.001,
        'default_diffusion_rate': 1e-1,
        'diffusion': {
            'glucose': 1E-1,
            'oxygen': 1E0,
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
        dx2_dy2_dz2 = dx * dy * dz
        self.diffusion_rate = diffusion_rate / dx2_dy2_dz2

        self.molecule_specific_diffusion = {
            mol_id: diff_rate / dx2_dy2_dz2
            for mol_id, diff_rate in self.parameters['diffusion'].items()
        }

        diffusion_dt = 0.5 * min(dx**2, dy**2, dz**2) / (2 * diffusion_rate)
        self.diffusion_dt = min(diffusion_dt, self.parameters['default_diffusion_dt'])
        self.bin_volume = get_bin_volume(self.bin_size)


    def initial_state(self, config=None):
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

    def get_single_local_environments(self, specs, fields):  #retrieve the info of each bin in a dict
        bin_site = self.get_bin_site(specs['location'])
        local_environment = {}
        for mol_id, field in fields.items():
            local_environment[mol_id] = field[bin_site]
        return local_environment

    def set_local_environments(self, species_populations, fields):
        local_environments = {}
        if species_populations:
            for species_id, specs in species_populations.items():
                bin_site = self.get_bin_site(specs['location'])
                local_environment = {}
                for mol_id, field in fields.items():
                    # Access the concentration of each molecule at the species location.
                    local_environment[mol_id] = field[bin_site]
                local_environments[species_id] = {
                    'boundary': {'external': local_environment}
                }
        return local_environments

    def ones_field(self):
        return np.ones(self.nbins, dtype=np.float64)

    def random_field(self):
        return np.random.rand(*self.nbins)

    def diffuse(self, field, timestep, diffusion_rate):
        t = 0.0
        dt = min(timestep, self.diffusion_dt)
        while t < timestep:
            result = convolve(field, LAPLACIAN_3D, mode='constant', cval=0.0)
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



def plot_fields(fields, out_dir='out', filename='fields'):
    for mol_id, field_list in fields.items():
        # Assuming the field data is the first element in the list
        field = field_list[0] if isinstance(field_list, list) and len(field_list) > 0 else None
        
        if field is not None and hasattr(field, 'shape'):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # Select the middle slice along z-axis for simplicity
            z_slice = field.shape[2] // 2
            slice_2d = field[:, :, z_slice]
            
            # Create a meshgrid to plot surface
            x, y = np.meshgrid(np.arange(slice_2d.shape[0]), np.arange(slice_2d.shape[1]))
            ax.plot_surface(x, y, slice_2d.T, cmap='viridis')
            
            ax.set_title(f'{mol_id} concentration (slice at z={z_slice})')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Concentration')
            
            plt.savefig(f"{out_dir}/{filename}_{mol_id}.png")
            plt.close()
        else:
            print(f"Warning: Field data for '{mol_id}' is not in the expected format.")


def plot_fields_temporal(fields_data, nth_timestep=5, out_dir='out', filename='fields_temporal'):
    # Assuming fields_data is a list of lists where each inner list represents the field data for a timestep
    if not fields_data or not isinstance(fields_data[0], list):
        print("Unexpected or empty data structure in 'fields_data'.")
        return

    # Convert the first element to numpy array to check its dimensions
    first_field = np.array(fields_data[0])
    if len(first_field.shape) != 3:
        print("Data does not have expected 3D structure.")
        return
    
    num_timesteps = len(fields_data)
    z_slice_index = first_field.shape[2] // 2  # Assuming we want the middle z-slice
    
    # Create subplots
    num_plots = (num_timesteps + nth_timestep - 1) // nth_timestep
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 4))
    
    for i, timestep_data in enumerate(fields_data[::nth_timestep]):
        field = np.array(timestep_data)
        slice_2d = field[:, :, z_slice_index]
        
        ax = axes[i] if num_plots > 1 else axes  # Handle the case of a single subplot differently
        cax = ax.imshow(slice_2d, cmap='viridis')
        ax.set_title(f'Timestep {i * nth_timestep}')
        fig.colorbar(cax, ax=ax)
    
    plt.suptitle(f'Concentration over time')
    plt.savefig(f"{out_dir}/{filename}.png")
    plt.close()




# Example of how you could implement the test_fields function with 3D plotting
def test_fields():
    # Example configuration for testing
    total_time = 30
    config = {
        "bounds": [10, 10, 10],
        "nbins": [10, 10, 10],
        "molecules": ["glucose", "oxygen"]
    }
    field = DiffusionField(config)

    # Initialize the simulation engine with the diffusion field process
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
    print(type(data['fields']['oxygen']))  # Check the type of the data for 'oxygen'
    first_oxygen_data = data['fields']['oxygen'][0] if isinstance(data['fields']['oxygen'], list) else None
    print(type(first_oxygen_data))  # Check the type of the first item if it's a list
    if isinstance(first_oxygen_data, np.ndarray):
        print(first_oxygen_data.shape)  # If it's an ndarray, check its shape



    # Plot the results
    first_fields = {key: matrix[0] for key, matrix in data['fields'].items()}
    plot_fields(first_fields, out_dir='out', filename='initial_fields')

    plot_fields_temporal(data['fields'], nth_timestep=5, out_dir='out', filename='fields_over_time')

if __name__ == '__main__':
    test_fields()
