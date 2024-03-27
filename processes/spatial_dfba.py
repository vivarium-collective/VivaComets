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


flux_mapping = {
    "EX_cpd00027_e0": "Glucose",  # Alteromonas
    "EX_glc__D_e": "Glucose",     # E. coli
    "EX_cpd00001_e0": "Water",    # Alteromonas
    "EX_h2o_e": "Water",          # E. coli
    "EX_cpd00007_e0": "Oxygen",   # Alteromonas
    "EX_o2_e": "Oxygen",          # E. coli
    "EX_cpd00011_e0": "Carbon dioxide",  # Alteromonas
    "EX_co2_e": "Carbon dioxide",        # E. coli
    "EX_cpd00013_e0": "Ammonia",  # Alteromonas
    "EX_nh4_e": "Ammonia",        # E. coli
    "EX_cpd00067_e0": "Hydrogen",  # Alteromonas
    "EX_h_e": "Hydrogen",          # E. coli
    "EX_cpd00009_e0": "Phosphate",  # Alteromonas
    "EX_pi_e": "Phosphate",         # E. coli
}


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
        'bounds': [3, 3, 3],  # cm
        'nbins': [3, 3, 3],
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
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.species_ids = self.parameters['species']

        # spatial settings
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters['nbins']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        self.bin_volume = get_bin_volume(self.bin_size)

        # load FBA models
        self.models = {}
        for species, model_path in self.parameters['species'].items():
            self.models[species] = read_sbml_model(model_path)
            #extract exchange fluxes self.externalmetabolite it should be a list.
            #make a name dictionry for exchange fluxes

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

    def get_reaction_id(self, molecule, species_model):
        # Map molecule name to reaction ID using the species_model id
        for reaction_id, mol_name in flux_mapping.items():
            if mol_name.lower() == molecule and reaction_id in [r.id for r in species_model.reactions]:
                return reaction_id
        return None

    def next_update(self, timestep, states):
        species = states['species']
        fields = states['fields']
        updated_biomass = {
            species_id: np.zeros(self.nbins)
            for species_id in species.keys()}
        updated_fields = {
            field_id: np.zeros(self.nbins)
            for field_id in fields.keys()}

        # go to each small cubic and compute FBA for each species
        for species_id, species_array in species.items():
            # get the model for this species
            species_model = self.models[species_id]

            # go through each position
            for x in range(self.nbins[0]):
                for y in range(self.nbins[1]):
                    for z in range(self.nbins[2]):

                        # get all the fields for this location
                        local_fields = {
                            field_id: field_array[x,y,z]
                            for field_id, field_array in fields.items()}

                        # get the species at this position
                        species_biomass = species_array[x,y,z]

                        
                        # run FBA
                        solution = species_model.optimize()
                        objective_flux = solution.objective_value
                        updated_biomass[species_id][x,y,z] += objective_flux

                        # CHECK Reduced the used flux from the environment.
                        for reaction_id, molecule_name in flux_mapping.items():
                            if molecule_name.lower() in ['glucose', 'oxygen'] and reaction_id in solution.fluxes.index:
                                flux = solution.fluxes[reaction_id]
                                if flux < 0:  # Consume molecule
                                    molecule_key = molecule_name.lower()
                                    if molecule_key not in updated_fields:
                                        # Initialize the field if it doesn't exist (optional, based on your needs)
                                        updated_fields[molecule_key] = np.zeros(self.nbins)
                                    updated_fields[molecule_key][x,y,z] -= flux * self.bin_volume



                        #TODO from the objective flux we need to get
                        #TODO go through exchange fluxes in the field and remove it from the environment
                        #calculate FBA for this species in this location

        return {
            'species': updated_biomass,
            'fields': updated_fields
        }


def test_spatial_dfba():
    # Configuration for the spatial environment and simulation
    total_time = 2
    config = {
        'bounds': [3, 3, 3],
        'nbins': [3, 3, 3],
        'molecules': ['glucose'],
        'species': {
            'Alteromonas': '../data/Alteromonas_Model.xml',
        }
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

    plot_fields_temporal(fields_data=data['fields'])


if __name__ == '__main__':
    test_spatial_dfba()
