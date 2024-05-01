"""
===============
Diffusion Field 
===============

Diffuses and decays molecular concentrations in a 3D field.
"""

import copy
import numpy as np
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


def get_bin_site(location, n_bins, bounds):
    bin_site_no_rounding = np.array([
        location[0] * n_bins[0] / bounds[0], 
        location[1] * n_bins[1] / bounds[1]
    ])
    bin_site = tuple(np.floor(bin_site_no_rounding).astype(int) % n_bins)
    return bin_site  # address of the bin


def get_bin_volume(bin_size):
    return np.prod(bin_size)


class DiffusionField(Process):
    defaults = {
        'bounds': [20, 20],
        'nbins': [20, 20],
        'molecules': ['glucose', 'oxygen'],
        'species': ['Alteromonas'],
        'default_diffusion_dt': 0.001,
        'default_diffusion_rate': 2E-5,
        'diffusion': {
            'glucose': 6.7E-1,
            'oxygen': 2.0E-2,
        },
        'advection': {
            'glucose': (0.01, 0.01),  
            'oxygen': (0.01, 0.01),   
        },
        'sinking': {
            'glucose': -0.01,  # if the value is positive it goes up
            'oxygen': -0.02,
            }
            }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters['nbins']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        assert len(self.bounds) == 2, "Proccess only support 2D"
        diffusion_rate = self.parameters['default_diffusion_rate']
        dx, dy = self.bin_size  
        dx2_dy2_dz2 = get_bin_volume(self.bin_size)
        self.diffusion_rate = diffusion_rate / dx2_dy2_dz2
        self.molecule_specific_diffusion = {
            mol_id: diff_rate / dx2_dy2_dz2
            for mol_id, diff_rate in self.parameters['diffusion'].items()
        }
        self.advection_vectors = {
            mol_id: self.parameters['advection'].get(mol_id, (0, 0))
            for mol_id in self.molecule_ids
        }
        self.sinking_rates = self.parameters['sinking']
    

        diffusion_dt = 0.5 * min(dx**2, dy**2) / (2 * diffusion_rate)
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
        fields = states['fields']
        fields_new = copy.deepcopy(fields)
        for mol_id, field in fields.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            advection_vector = self.parameters['advection'].get(mol_id, (0, 0))
            sinking_rate = self.parameters['sinking'].get(mol_id, 0)
            if np.var(field) > 0:  # If field is not uniform
                fields_new[mol_id] = self.diffuse(field, timestep, diffusion_rate, advection_vector, sinking_rate)
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

    def diffuse(self, field, timestep, diffusion_rate, advection_vector, sinking_rate):
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
            laplacian = convolve(field, laplacian_kernel, mode='reflect') * diffusion_rate
            grad_x = convolve(field, gradient_x_kernel, mode='reflect') * advection_vector[0]
            grad_y = convolve(field, gradient_y_kernel, mode='reflect') * advection_vector[1]
            if sinking_rate != 0:
                vertical_shift = int(sinking_rate * dt / self.bin_size[1])  # Vertical shift in bins
                if vertical_shift != 0:
                    field = np.roll(field, shift=vertical_shift, axis=0)

            field += dt * (laplacian - grad_x - grad_y)
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
    
    
def plot_fields_temporal(
        fields_data,
        desired_time_points,
        actual_time_points,
        out_dir='out',
        filename='fields_at_z', 
        molecule_colormaps = {},
        plot_fields = ["glucose" , "oxygen"]
):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
   
    # Convert desired and actual time points to float for accurate indexing
    desired_time_points = [float(time) for time in desired_time_points]
    actual_time_points = [float(time) for time in actual_time_points]
    num_molecules = len(plot_fields)
    num_times = len(desired_time_points)
    fig, axs = plt.subplots(num_times, num_molecules, figsize=(10, num_times * 5), squeeze=False)
    
    # Calculate global min/max for each molecule across all timepoints
    global_min_max = {}
    for molecule in fields_data.keys() :
        if molecule not in plot_fields:
            continue
        all_data = np.concatenate([np.array(times_data) for times_data in fields_data[molecule]], axis=0)
        global_min_max[molecule] = (np.min(all_data), np.max(all_data))
    

    for mol_idx, molecule in enumerate(fields_data.keys()):
        if molecule not in plot_fields:
            continue
        times_data = fields_data[molecule]
        for time_idx, desired_time in enumerate(desired_time_points):
            if desired_time in actual_time_points:
                actual_idx = actual_time_points.index(desired_time)
                data_array = np.array(times_data[actual_idx])  # Accessing the time-specific data
    
                ax = axs[time_idx, mol_idx]
                # Use the specified colormap for the molecule
                cmap = molecule_colormaps.get(molecule, 'Blues')  # Default to 'viridis' if molecule not in dict
                vmin, vmax = global_min_max[molecule]  # Use global min/max
                cax = ax.imshow(data_array, cmap=cmap, interpolation='nearest',  vmin=vmin, vmax=vmax)

                # molecule labels for top row
                if time_idx == 0:
                    ax.set_title(molecule, fontsize=24)

                # time label for leftmost column
                if mol_idx == 0:
                    ax.set_ylabel(f'Time {desired_time}', fontsize=22)

                ax.set_xticks([])
                ax.set_yticks([])
                if time_idx == 0:
                    cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                    cb.ax.tick_params(labelsize=10)
                
    plt.tight_layout()
    fig_path = os.path.join(out_dir, filename)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()


def test_fields():
    total_time = 200
    config = {
        'bounds': [20, 20],
        'nbins': [20, 20],
        'molecules': ['glucose', 'oxygen'], 
        'diffusion': {
            'glucose': 6.7E-1, #6.7E-6,  # cm^2/s  
            'oxygen':  2.0E-2,     #2.0E-5,   # cm^2/s 
        },
        'advection': {
            'glucose': (0.01, 0.01),  # Advection vector for glucose
            'oxygen': (0.01, 0.01),   # Advection vector for oxygen
        },
        'sinking': {
            'glucose': -0.02,  # Sinks at a rate of 0.02 units per timestep
            'oxygen': -0.01,   # Sinks at a rate of 0.01 units per timestep
        }
    }
    Diffusion_field = DiffusionField(config) 
    initial_state = Diffusion_field.initial_state({'random': 1.0})
    sim = Engine(
        initial_state=initial_state,
        processes={'diffusion_process': Diffusion_field},
        topology={'diffusion_process': {
            'fields': ('fields',),
            'species': ('species',),
            'dimensions': ('dimensions',),
        }}
    )
    # Run the simulation
    sim.update(total_time)
    # Get the results
    data = sim.emitter.get_timeseries()  #TODO add a line and make assert to check, like biomass should not decreased as the times go
    time_list = [0, 1, int(total_time/4), int(total_time/2), total_time-1]
    # Inside test_fields function
    plot_fields_temporal(
        fields_data=data['fields'],
        desired_time_points=time_list,
        actual_time_points=data['time'],
        out_dir='./out',
        filename='Diffusion_test')
    #test


if __name__ == '__main__':
    test_fields()
