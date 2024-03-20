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
        'bounds': [30, 30, 30], # cm
        'nbins': [3, 3, 3],
        'molecules': ['glucose', 'oxygen'],
        'species': ["Alteromonas"],
        'default_diffusion_dt': 0.001,
        'default_diffusion_rate': 2E-5,  # cm^2/s, set to the highest diffusion coefficient (oxygen)
        'diffusion': {
            'glucose': 6.7E-6,  # cm^2/s  TODO should find the current rate for cm^3
            'oxygen':  2.0E-5,   # cm^2/s
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
            'fields': {},
            'cubic_dict': {
                '_default': {},  
                '_updater': 'set',  
                '_emit': True, 
            }
        }
        for species in self.parameters['species']:
            schema['species'].update({
                species: {'_default': np.ones(self.nbins), '_updater': 'nonnegative_accumulate', '_emit': True}
            })
        for mol in self.parameters['molecules']:
            schema['fields'].update({
                mol: {'_default': np.ones(self.nbins), '_updater': 'nonnegative_accumulate', '_emit': True}
            })
        schema['dimensions'] = {
            'bounds': {'_value': self.bounds, '_updater': 'set', '_emit': True},
            'nbins': {'_value': self.nbins, '_updater': 'set', '_emit': True}
        }
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
    
def plot_fields_temporal(fields_data, desired_time_points, actual_time_points, z=2,
                         out_dir="/Users/amin/Desktop/VivaComet/processes/out/", filename='fields_at_z'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    z_index = z - 1

    # Convert desired and actual time points to float for accurate indexing
    desired_time_points = [float(time) for time in desired_time_points]
    actual_time_points = [float(time) for time in actual_time_points]
    num_molecules = len(fields_data)
    num_times = len(desired_time_points)
    fig, axs = plt.subplots(num_times, num_molecules, figsize=(10, num_times * 5), squeeze=False)
    if num_molecules == 1 or num_times == 1:
        axs = np.array([[axs]])

    # Define a colormap for each molecule
    molecule_colormaps = {
        'glucose': 'Blues',
        'oxygen': 'Greens',
    }

    for mol_idx, molecule in enumerate(fields_data.keys()):
        times_data = fields_data[molecule]
        for time_idx, desired_time in enumerate(desired_time_points):
            if desired_time in actual_time_points:
                actual_idx = actual_time_points.index(desired_time)
                data_array = np.array(times_data[actual_idx])  # Accessing the time-specific data
                data = data_array[..., z_index]
                ax = axs[time_idx, mol_idx]
                # Use the specified colormap for the molecule
                cmap = molecule_colormaps.get(molecule, 'viridis')  # Default to 'viridis' if molecule not in dict
                cax = ax.imshow(data, cmap=cmap, interpolation='nearest')
                if time_idx == 0:
                    ax.set_title(molecule, fontsize=24)
                ax.set_ylabel(f"Time {desired_time}", fontsize=22)
                ax.set_xticks([])
                ax.set_yticks([])
                if time_idx == 0:
                    cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                    cb.ax.tick_params(labelsize=10)

                
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}.png")
    plt.close()


class Spatial_FBA(Process):
    defaults = {
        'bounds': [3, 3, 3], # cm
        'nbins': [3, 3, 3],
        'molecules': ['glucose', 'oxygen'],
        'species': {"Alteromonas": "/Users/amin/Desktop/VivaComet/data/Alteromonas_Model.xml",
                     "ecoli": "/Users/amin/Desktop/VivaComet/data/e_coli_core.xml" } 
    }
     
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters["nbins"]
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        self.bin_volume = get_bin_volume(self.bin_size)
        #load FBA models
        self.models = {}
        for species, modelpath in self.parameters["species"].items():
            self.models[species] = read_sbml_model(self.parameters['model_file'])

    def ports_schema(self):
        return {
            'species': {},
            'fields': {},
            'cubic_dict': {
                '_default': {},
                '_updater': 'set',
            },
        }

    def next_update(self, timestep, states):
        species = states["species"]
        fields = states["fields"]
        updated_biomass = {species_id : np.zeros(self.nbins) for species_id in species.keys() }
        updated_fields = {field_id : np.zeros(self.nbins) for field_id in fields.keys()}
        
        # go to each small cubic and compute FBA for each species
        for species_id, species_array in species.items():
            for x in range(self.nbins[0]):
                for y in range(self.nbins[1]):
                    for z in range(self.nbins[2]):
                        local_fields = {field_id : field_array[x,y,z] for field_id, field_array in fields.items() }
                        species_biomass = species_array[x,y,z]
                        species_model = self.models[species_id]
                        #TODO make this more generic, dont hardcode names.
                        species_model.reactions.get_by_id('EX_glc__D_e').lower_bound = -local_fields['glucose'] # TODO make sure the units are compatible
                        species_model.reactions.get_by_id('EX_o2_e').lower_bound = -local_fields['oxygen'] 
                        solution = species_model.optimize()
                        objective_flux = solution.objective_value
                        updated_biomass[species_id][x,y,z] += objective_flux 

                        #TODO from the objective flux we need to get 
                        #TODO go through exchange fluxes in the field and remove it from the environment
                        #calculate FBA for this species in this location
        return {'species': updated_biomass , "fields": updated_fields}
   

#  3D test_field
def test_fields():
    total_time = 1000
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
    time_list=[0,1,5, 50, 100, 1000]
    # Inside test_fields function
    plot_fields_temporal(fields_data=data['fields'],
                     desired_time_points=time_list, 
                     actual_time_points=data["time"], 
                     z=2, 
                     out_dir="/Users/amin/Desktop/VivaComet/processes/out", 
                     filename='fields_over_time')





def test_fba():
    # Configuration for the spatial environment and simulation
    total_time = 2
    config = {
        "bounds": [3, 3, 3],  
        "nbins": [3, 3, 3],  
        "molecules": ["glucose"],  
        "species": {
            "Alteromonas": "/Users/amin/Desktop/VivaComet/data/Alteromonas_Model.xml",  
        }  
    }

    fba_process = Spatial_FBA(config)

    # Define the initial state with some initial concentrations for the molecules
    initial_state = {
        'fields': {
            'glucose': np.full((3, 3, 3), 5.0),  # Initial glucose concentration in each bin
          
        },
        'species': {
            "Alteromonas": np.full((3, 3, 3), 1.0),  # Initial biomass for Alteromonas in each bin
        }
    }

    sim = Engine(
        initial_state=initial_state,
        processes={'fba_process': fba_process},
        topology={'fba_process': {
            'fields': ('fields',),
            'species': ('species',),
            'cubic_dict': ('cubic_dict',)  
        }}
    )

    sim.update(total_time)
    data = sim.emitter.get_timeseries()


if __name__ == '__main__':
    test_fields()
    test_fba()
