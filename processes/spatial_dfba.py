import numpy as np
from vivarium.core.process import Process
from vivarium.core.engine import Engine

from processes.diffusion_field import get_bin_volume

from cobra.io import read_sbml_model


class SpatialDFBA(Process):
    defaults = {
        'bounds': [3, 3, 3],  # cm
        'nbins': [3, 3, 3],
        'molecules': [
            'glucose',
            'oxygen'
        ],
        'species': {
            'Alteromonas': '/Users/amin/Desktop/VivaComet/data/Alteromonas_Model.xml',
            'ecoli': '/Users/amin/Desktop/VivaComet/data/e_coli_core.xml'
        }
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecule_ids = self.parameters['molecules']
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters['nbins']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]
        self.bin_volume = get_bin_volume(self.bin_size)

        # load FBA models
        self.models = {}
        for species, modelpath in self.parameters['species'].items():
            self.models[species] = read_sbml_model(self.parameters['model_file'])
            #extract exchange fluxes self.externalmetabolite it should be a list.
            #make a name dictionry for exchange fluxes

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

    def next_update(self, timestep, states):
        species = states['species']
        fields = states['fields']
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
            'Alteromonas': '/Users/amin/Desktop/VivaComet/data/Alteromonas_Model.xml',
        }
    }

    fba_process = SpatialDFBA(config)

    # initial state
    initial_state = {
        'fields': {
            'glucose': np.full((3, 3, 3), 5.0),

        },
        'species': {
            'Alteromonas': np.full((3, 3, 3), 1.0),
        }
    }

    sim = Engine(
        initial_state=initial_state,
        processes={'fba_process': fba_process},
        topology={'fba_process': {
            'fields': ('fields',),
            'species': ('species',),
        }}
    )

    sim.update(total_time)
    data = sim.emitter.get_timeseries()


if __name__ == '__main__':
    test_spatial_dfba()