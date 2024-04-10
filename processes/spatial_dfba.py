"""
============
Spatial DFBA
============
"""
import numpy as np
from vivarium.core.process import Process
from vivarium.core.engine import Engine
from diffusion_field import get_bin_volume, plot_fields_temporal
from cobra.io import read_sbml_model

class SpatialDFBA(Process):
    """
    SpatialDFBA

    This process simulates the growth of multiple species in a spatial environment.

    config:
    - bounds: the size of the environment in each dimension
    - nbins: the number of bins in each dimension
    - molecules: the list of molecules in the environment
    - species: a dictionary of species names and paths to their FBA models
    """
    defaults = {
        'bounds': [3, 3],  # cm
        'nbins': [3, 3],
        'molecules': [
            'glucose',
            'oxygen'
        ],
        'species': {
            'Alteromonas': '../data/Alteromonas_Model.xml'
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.species_ids = self.parameters['species']
        # spatial settings
        self.bounds = self.parameters['bounds']
        assert len(self.bounds) == 2, "This process ONLY support 2D"
        self.nbins = self.parameters['nbins']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        self.bin_volume = get_bin_volume(self.bin_size)

        # load FBA models
        self.models = {}
        self.flux_id_maps = {}
        self.kinetic_params={}
        for species in self.parameters.get('species_info', []):
            model_path = species['model']
            species_name = species['name']
            self.models[species_name] = read_sbml_model(model_path)
            self.flux_id_maps[species_name] = species['flux_id_map']
            self.kinetic_params[species_name] = species['kinetic_params']
            

    def initial_state(self, config=None):

        # update fields and species with initial values from config
        fields = {molecule: 0.0 for molecule in self.molecule_ids}
        species = {species: 0.0 for species in self.species_ids}
        fields.update(config.get('fields', {}))
        species.update(config.get('species', {}))

        initial_state = {
            'fields': {
                molecule: np.ones(self.nbins) * concentration
                for molecule, concentration in fields.items()
                if molecule in self.molecule_ids
            },
            'species': {
                species: np.ones(self.nbins) * biomass
                for species, biomass in species.items()
                if species in self.species_ids
            }
        }
        return initial_state

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
                '_emit': True},
            'nbins': {
                '_value': self.nbins,
                '_updater': 'set',
                '_emit': True
            }
        }
        return schema

    def get_reaction_id(self, molecule, species_name):
        # Use species_name to fetch the correct flux_id_map and then map molecule to reaction ID
        flux_id_map = self.flux_id_maps.get(species_name, {})
        for mol_name, reaction_id in flux_id_map.items():
            if mol_name.lower() == molecule.lower():  # ensure case-insensitive comparison
                return reaction_id
        return None
    
    def get_kinetic_params(self, species_name, molecule_name):
        """
        Fetches the kinetic parameters (Km and Vmax) for a given molecule and species.
        """
        species_info = next((item for item in self.parameters['species_info'] if item['name'] == species_name), None)
        if species_info and 'kinetic_params' in species_info:
            return species_info['kinetic_params'].get(molecule_name)
        return None

    #TODO from the objective flux we need to get
    #TODO go through exchange fluxes in the field and remove it from the environment
    #calculate FBA for this species in this location
    def next_update(self, timestep, states):
        species_states = states['species']
        field_states = states['fields']
        updated_biomass = {species_id: np.zeros(self.nbins) for species_id in species_states.keys()}
        updated_fields = {field_id: np.zeros(self.nbins) for field_id in field_states.keys()}

        for species_id, species_array in species_states.items():
            species_model = self.models[species_id]

            for x in range(self.nbins[0]):
                for y in range(self.nbins[1]):
                    local_fields = {field_id: field_array[x, y] for field_id, field_array in field_states.items()}
                    species_biomass = species_array[x, y]

                    # Update uptake rates based on local conditions and kinetic parameters
                    for molecule_name, local_concentration in local_fields.items():
                        kinetic_params = self.get_kinetic_params(species_id, molecule_name)
                        if kinetic_params:
                            Km, Vmax = kinetic_params
                            uptake_rate = Vmax * local_concentration / (Km + local_concentration)
                            reaction_id = self.get_reaction_id(molecule_name, species_id)
                            if reaction_id:
                                reaction = species_model.reactions.get_by_id(reaction_id)
                                initial_lower_bound = reaction.lower_bound  # Store the initial lower bound
                                # Update the reaction lower bound with the more restrictive of the calculated or initial rate
                                reaction.lower_bound = max(initial_lower_bound, -uptake_rate)

                    solution = species_model.optimize()
                    if solution.status == 'optimal':
                        objective_flux = solution.objective_value
                        updated_biomass[species_id][x, y] += objective_flux * species_biomass * timestep

                        for molecule_name in self.molecule_ids:
                            reaction_id = self.get_reaction_id(molecule_name, species_id)
                            if reaction_id and reaction_id in solution.fluxes.index:
                                flux = solution.fluxes[reaction_id]
                                updated_fields[molecule_name.lower()][x, y] += flux * self.bin_volume * timestep

        return {
            'species': updated_biomass, 
            'fields': updated_fields
            }


def test_spatial_dfba():
    # Configuration for the spatial environment and simulation
    total_time = 2
    timestep = 1 
    desired_time_points = [0, timestep, total_time]
    actual_time_points = desired_time_points
    config = {
        'bounds': [3, 3],  # dimensions of the environment
        'nbins': [3, 3],   # division into bins
        'molecules': ['glucose', 'oxygen'],  # available molecules
        "species_info": [
            # {
            #     "model": '../data/Alteromonas_Model.xml', 
            #     "name": "Alteromonas",
            #     "flux_id_map": {
            #         "glucose": "EX_cpd00027_e0",
            #         "oxygen": "EX_cpd00007_e0"
            #     },
            #     "kinetic_params": {
            #         "glucose": (0.5, 2.0),  # Km, Vmax for glucose
            #         "oxygen": (0.3, 5.0),   # Km, Vmax for oxygen
            #     }
            # },
            {
                "model": '../data/e_coli_core.xml', 
                "name": "ecoli",
                "flux_id_map": {
                    "glucose": "EX_glc__D_e",
                    "oxygen": "EX_o2_e"
                },
                "kinetic_params": {
                    "glucose": (0.4, 1.5),  # Km, Vmax for glucose
                    "oxygen": (0.25, 4.5),  # Km, Vmax for oxygen
                }
            }
        ]
    }


    fba_process = SpatialDFBA(config)

    # initial state
    initial_state = fba_process.initial_state({
        'fields': {
            'glucose': 5.0
        },
        'species': {
            'Alteromonas': 1.0
        }
    })

    sim = Engine(
        initial_state=initial_state,
        processes={'fba_process': fba_process},
        topology={'fba_process': {
            'fields': ('fields',),
            'species': ('species',),
            'dimensions': ('dimensions',),
        }}
    )

    sim.update(total_time)
    data = sim.emitter.get_timeseries()
    fields = data["fields"]
    fields.update(data["species"])
    plot_fields_temporal(
        fields_data=fields, 
        desired_time_points=desired_time_points, 
        actual_time_points=actual_time_points,
        plot_fields = ["glucose", "oxygen", "Alteromonas"],
        molecule_colormaps= {"glucose": "Blues" , "oxygen": "Greens", "Alteromonas": "Purples"}
        )



if __name__ == '__main__':
    test_spatial_dfba()
