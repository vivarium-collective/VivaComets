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
            'Alteromonas': '../data/Alteromonas_Model.xml',
            'ecoli': '../data/e_coli_core.xml'
        }
    }

    def __init__(self, parameters=None):
        # assert for 2D 
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
        for species in self.parameters.get('species_info', []):
            model_path = species['model']
            species_name = species['name']
            self.models[species_name] = read_sbml_model(model_path)
            self.flux_id_maps[species_name] = species['flux_id_map']
            

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

    #TODO from the objective flux we need to get
    #TODO go through exchange fluxes in the field and remove it from the environment
    #calculate FBA for this species in this location
    def next_update(self, timestep, states):
        species_states = states['species']
        field_states = states['fields']
        updated_biomass = {species_id: np.zeros(self.nbins) for species_id in species_states.keys()}
        updated_fields = {field_id: np.zeros(self.nbins) for field_id in field_states.keys()}

        # Iterate through each species present
        for species_id, species_array in species_states.items():
            # Retrieve the metabolic model for the current species
            species_model = self.models[species_id]

            # Iterate through each bin in the environment
            for x in range(self.nbins[0]):
                for y in range(self.nbins[1]):
    
                    # Aggregate local environmental conditions for this bin
                    local_fields = {field_id: field_array[x, y] for field_id, field_array in field_states.items()}

                    # Fetch the current biomass for this species at this location
                    species_biomass = species_array[x, y]

                    # Conduct FBA for the current species under local conditions
                    solution = species_model.optimize()
                    objective_flux = solution.objective_value  # Objective flux typically represents growth rate
                    updated_biomass[species_id][x, y] += objective_flux

                    # Update environmental fields based on the metabolic byproducts/consumption
                    for molecule_name in self.molecule_ids:
                        # Convert molecule names to the corresponding reaction IDs in the model
                        reaction_id = self.get_reaction_id(molecule_name, species_id)
                        if reaction_id and reaction_id in solution.fluxes.index:
                            flux = solution.fluxes[reaction_id]
                            if molecule_name.lower() not in updated_fields:
                                updated_fields[molecule_name.lower()] = np.zeros(self.nbins)
                            # Adjust the concentration of the molecule in the environment based on the flux
                            updated_fields[molecule_name.lower()][x, y] += flux * self.bin_volume

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
        'bounds': [3, 3],
        'nbins': [3, 3],
        'molecules': ['glucose', 'oxygen'],
        "species_info": [
            {
                "model": '../data/Alteromonas_Model.xml', 
                "name": "Alteromonas",
                "flux_id_map": {
                    "glucose" : "EX_cpd00027_e0",
                    "Oxygen": "EX_cpd00007_e0"
                }
            },
            {
                "model": '../data/e_coli_core.xml', 
                "name": "ecoli",
                "flux_id_map": {
                    "glucose" : "EX_glc__D_e",
                    "Oxygen": "EX_o2_e"
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
