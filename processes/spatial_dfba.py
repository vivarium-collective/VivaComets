"""
============
Spatial DFBA
============
"""
import os
import numpy as np
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from processes.diffusion_field import get_bin_volume
from cobra.io import read_sbml_model
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")

# Suppress logging warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Suppress specific cobra warnings
cobra_logger = logging.getLogger('cobra.medium.boundary_types')
cobra_logger.setLevel(logging.ERROR)

# Determine the absolute path to the data directory
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')


class SpatialDFBA(Process):
    """
    SpatialDFBA

    This process simulates the growth of multiple species in a spatial environment.

    config:
    - bounds: the size of the environment in each dimension
    - nbins: the number of bins in each dimension
    - molecules: the list of molecules in the environment
    - species_info: a list of dictionaries of species names and paths to their FBA models
    """
    defaults = {
        'bounds': [3, 3],  # cm
        'nbins': [3, 3],
        'depth': 0.01,
        'molecules': [
            'glucose',
            'acetate'
        ],
        'species_info': [ 
            {
                'name': 'Alteromonas',
                'model': os.path.join(data_dir, 'Alteromonas_Model.xml'),
                'flux_id_map': {},
                'kinetic_params': {},
                # 'biomass_reaction': 'BIOMASS',  # TODO -- check this.
                # 'fixed_bounds': {'reaction_name': {'lower': -100, 'upper': 100}},
            }
        ],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.species_ids = {
            info['name']: info['model']
            for info in self.parameters.get('species_info', [])
        }
        
        # spatial setting
        self.bounds = self.parameters['bounds']
        assert len(self.bounds) == 2, 'This process ONLY supports 2D, change the bounds parameters'
        self.nbins = self.parameters['nbins']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        self.bin_volume = get_bin_volume(self.bin_size, self.parameters['depth'])

        # load FBA Model
        self.models = {}
        self.flux_id_maps = {}
        self.kinetic_params = {}
        self.exchange_fluxes = {}

        # load a model for each species
        for species in self.parameters.get('species_info', []):
            model_path = species['model']
            species_name = species['name']
            
            self.models[species_name] = read_sbml_model(model_path)
            print(f'Loaded model for {species_name}')
            self.flux_id_maps[species_name] = species.get('flux_id_map', {})
            self.kinetic_params[species_name] = species.get('kinetic_params', {})

            # Initialize exchange fluxes 
            self.exchange_fluxes[species_name] = {
                reaction.id: np.zeros(self.nbins)
                for reaction in self.models[species_name].exchanges}

            # Apply fixed bounds
            if 'fixed_bounds' in species:
                for reaction_id, bounds in species['fixed_bounds'].items():
                    if reaction_id in self.models[species_name].reactions:
                        reaction = self.models[species_name].reactions.get_by_id(reaction_id)
                        lb = bounds.get('lower', None)
                        ub = bounds.get('upper', None)
                        if lb is not None:
                            reaction.lower_bound = lb
                        if ub is not None:
                            reaction.upper_bound = ub

    def initial_state(self, config=None):
        if config is None:
            config = {}
            print('Warning: No configuration provided, initializing to default zero values.')

        fields = {}
        species = {}
        shape = tuple(self.nbins)

        # Initialize fields (molecules)
        for molecule in self.molecule_ids:
            if 'random' in config and molecule in config['random']:
                max_value = config['random'][molecule]
                fields[molecule] = np.random.rand(*shape) * max_value
            elif 'uniform' in config and molecule in config['uniform']:
                value = config['uniform'][molecule]
                fields[molecule] = np.full(shape, value)
            else:
                print(f"No specific initialization for molecule '{molecule}', defaulting to zero.")
                fields[molecule] = np.zeros(shape)

        # Initialize species (biomass)
        for spec in self.species_ids:
            if 'random' in config and 'species' in config['random'] and spec in config['random']['species']:
                max_value = config['random']['species'][spec]
                species[spec] = np.random.rand(*shape) * max_value
            elif 'uniform' in config and 'species' in config['uniform'] and spec in config['uniform']['species']:
                value = config['uniform']['species'][spec]
                species[spec] = np.full(shape, value)
            else:
                print(f"No specific initialization for species '{spec}', defaulting to zero.")
                species[spec] = np.zeros(shape)

        return {'fields': fields, 'species': species}

    def ports_schema(self):
        schema = {
            'species': {},
            'fields': {},
            'exchange_fluxes': {},
            'dimensions': {
                'bounds': {
                    '_value': self.bounds,
                },
                'nbins': {
                    '_value': self.nbins,
                }
            }
        }

        # Define schema for each species based on species_info
        for species_info in self.parameters['species_info']:
            species_name = species_info['name']
            schema['species'][species_name] = {
                '_default': np.zeros(self.nbins),  # Initialize to zero biomass
                '_updater': 'nonnegative_accumulate',
                '_emit': True
            }

            # Define schema for exchange fluxes with additional information
            schema['exchange_fluxes'][species_name] = {
                reaction_id: {
                        '_default': np.zeros(self.nbins),
                        '_updater': 'set',
                        '_emit': True,
                        '_output': True
                } for reaction_id in self.exchange_fluxes[species_name]
            }

        # add all the fields to molecules
        for molecule in self.parameters['molecules']:
            schema['fields'][molecule] = {
                '_default': np.zeros(self.nbins),  # Initialize to zero concentration
                '_updater': 'nonnegative_accumulate',
                '_emit': True
            }

        return schema

    def get_reaction_id(self, molecule, species_name):
        # Use species_name to fetch the correct flux_id_map and then map molecule to reaction ID
        flux_id_map = self.flux_id_maps.get(species_name, {})
        for mol_name, reaction_id in flux_id_map.items():
            if mol_name.lower() == molecule.lower():  # ensure case-insensitive comparison
                return reaction_id
        return None

    def next_update(self, timestep, states):
        """
        The main process update method. This method is called by the Vivarium engine at each timestep.
        """

        # get the states
        species_states = states['species']
        field_states = states['fields']

        # prepare update dicts
        updated_biomass = {species_id: np.zeros(self.nbins) for species_id in species_states.keys()}
        updated_fields = {field_id: np.zeros(self.nbins) for field_id in field_states.keys()}
        updated_exchange_fluxes = {
            species_id: {
                reaction.id: np.zeros(self.nbins)
                for reaction in self.models[species_id].exchanges
            } for species_id in species_states.keys()
        }

        # iterate over each species and update biomass and fields
        for species_id, species_array in species_states.items():
            if species_id not in self.models:
                print(f"Model for {species_id} not found")
                continue  # Skip this species if the model is not found

            species_model = self.models[species_id] 

            # iterate over each bin in the grid
            # TODO -- this can be parallelized
            for x in range(self.nbins[0]):
                for y in range(self.nbins[1]):
                    # get the local fields at this bin and the species biomass
                    local_fields = {
                        field_id: field_array[x, y]
                        for field_id, field_array in field_states.items()}
                    species_biomass = species_array[x, y]
                    kinetic_params = self.kinetic_params[species_id]

                    # Update uptake rates based on local fields and kinetic parameters
                    for molecule_name, local_concentration in local_fields.items():
                        if molecule_name in kinetic_params:
                            Km, Vmax = kinetic_params[molecule_name]
                            uptake_rate = Vmax * local_concentration / (Km + local_concentration)
                            reaction_id = self.get_reaction_id(molecule_name, species_id)
                            if reaction_id:
                                reaction = species_model.reactions.get_by_id(reaction_id)
                                reaction.lower_bound = -uptake_rate
                                # print(f'SET UPTAKE RATE: {reaction_id} = {uptake_rate}')

                    # run FBA
                    solution = species_model.optimize()
                    # print(f"Species: {species_id}, Bin: ({x}, {y}), Objective: {solution.objective_value}, Status: {solution.status}")

                    # get the solutions
                    if solution.status == 'optimal':

                        # update the species biomass
                        objective_flux = solution.objective_value
                        updated_biomass[species_id][x, y] = objective_flux * species_biomass * timestep

                        # update the fields 
                        for molecule_name in self.molecule_ids:
                            reaction_id = self.get_reaction_id(molecule_name, species_id)
                            if reaction_id and reaction_id in solution.fluxes.index:
                                flux = solution.fluxes[reaction_id]
                                delta_conc = flux * species_biomass * timestep
                                updated_fields[molecule_name][x, y] = delta_conc / self.bin_volume  # TODO -- check normalize for bin volume

                        # update exchange fluxes
                        for reaction_id in self.exchange_fluxes[species_id]:
                            if reaction_id in solution.fluxes.index:
                                updated_exchange_fluxes[species_id][reaction_id][x, y] = solution.fluxes[reaction_id]
        return {
            'species': updated_biomass, 
            'fields': updated_fields,
            'exchange_fluxes': updated_exchange_fluxes
            }
    

def test_spatial_dfba(
        total_time=10,
        nbins=[2, 2],
):
    # Configuration for the spatial environment and simulation
    # timestep = 1/60
    # desired_time_points = [0, 1, int(total_time/4), int(total_time/2), total_time-1]
    # actual_time_points = desired_time_points
    initial_state_config = {
        'random': {
            'glucose': 200.0,  # Max random value for glucose
            'acetate': 200.0,   # Max random value for oxygen
            'species': {
                'ecoli': 0.5,   # Max random value for E. coli biomass
                'Alteromonas': 0.5   # Max random value for Alteromonas biomass
            }
        }
    }

    config = {
        'bounds': [10, 10],  # dimensions of the environment
        'nbins': nbins,   # division into bins
        'molecules': ['glucose', 'acetate'],  # available molecules
        "species_info": [
            {
                "model": os.path.join(data_dir, 'Alteromonas_Model.xml'), 
                "name": "Alteromonas",
                "flux_id_map": {
                    "glucose": "EX_cpd00027_e0",
                    # "oxygen": "EX_cpd00007_e0"
                },
                "kinetic_params": {
                    "glucose": (0.5, 0.0005),  # Km, Vmax for glucose
                    # "acetate": (0.3,  0.0005),   # Km, Vmax for oxygen
                },
                "fixed_bounds": {
                    'EX_cpd00149_e0': {'lower': -10, 'upper': 10}   # Setting fixed bounds for Alteromonas
                }
            },
            {
                "model": os.path.join(data_dir, 'iECW_1372.xml'), 
                "name": "ecoli",
                "flux_id_map": {
                    "glucose": "EX_glc__D_e",
                    'acetate': 'EX_ac_e'
                },
                "kinetic_params": {
                    "glucose": (0.4, 0.6),  # Km, Vmax for glucose
                    "acetate": (0.25, 0.6),  # Km, Vmax for oxygen
                },
                "fixed_bounds": {
                    'EX_o2_e': {'lower': -2},  # Setting fixed bounds for E. coli
                    'ATPM': {'lower': 1, 'upper': 1}  # Setting fixed bounds for E. coli
                    # 'EX_fe3dhbzs_e': {'lower': 0, 'upper': 10}
                }
            }
        ]
    }

    # create the process
    fba_process = SpatialDFBA(config)

    # # get exchange data from the process
    # species_ids = [info['name'] for info in config.get('species_info', [])]
    # exchange_fluxes_info = {
    #     species_id: {
    #         reaction.id: {
    #             'name': reaction.name,
    #             'reaction': reaction.reaction,
    #             'lower_bound': reaction.lower_bound,
    #             'upper_bound': reaction.upper_bound,
    #         } for reaction in fba_process.models[species_id].exchanges
    #     } for species_id in species_ids
    # }

    # initial state
    initial_state = fba_process.initial_state(initial_state_config)

    # make the simulation and run it
    sim = Engine(
        initial_state=initial_state,
        processes={'fba_process': fba_process},
        topology={'fba_process': {
            'fields': ('fields',),
            'species': ('species',),
            'exchange_fluxes': ('exchange_fluxes',),
            'dimensions': ('dimensions',),
        }}
    )
    sim.update(total_time)

    # get the data
    data = sim.emitter.get_timeseries()
    fields = data["fields"]
    fields.update(data["species"])

    # # plots
    # plot_objective_flux(
    #     data,
    #     time_points=desired_time_points,
    #     species_names=[species['name'] for species in config['species_info']],
    #     out_dir='./out',
    #     filename='objective_flux_plot'
    # )

    # plot_fields_temporal(
    #     fields_data=data['fields'], 
    #     desired_time_points=desired_time_points, 
    #     actual_time_points=actual_time_points,
    #     plot_fields=["glucose", "oxygen", "ecoli"],
    #     molecule_colormaps={"glucose": "Blues", "oxygen": "Greens", "ecoli": "Purples"},
    #     out_dir='./out',
    #     filename='spatial_dfba_test',
    # )

if __name__ == '__main__':
    test_spatial_dfba()
