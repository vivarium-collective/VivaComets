"""
=============
dFBA Process in Different Temperature
Amin Boroomand, 2024
==============
This code performs Dynamic Flux Balance Analysis (DFBA) by reading
a model from an SBML file and computing the flux at each time step.
"""

from vivarium.core.process import Process, Step
from vivarium.core.engine import Engine, pf
from cobra.io import read_sbml_model
import random
# Add the Time_proportion variable, We consider each time-step a minute
TIME_PROPORTION = (1 / 60)  # we set each time-step as an hour and the Time_proportion as a minute


class ReactionBounds(Process):
    """
    This class initializes and updates the reaction bounds for the model.

    ReactionBounds ingests a COBRA model, creating an initial dictionary of reaction bounds. For each update, it adjusts the reaction bounds considering a predefined aging percentage and resource limitation. The resource limitation is evaluated based on the available resources and their consumption at each timestep. This class now also computes the current_v0 value, which represents the flux calculated using the current glucose concentration by the Michaelis-Menten equation.
    """

    defaults = {
        'model_file': None,
        'enz_concentration': 5,
        'kcat': 5,
        'init_concentration': 11.1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters)
        self.model = read_sbml_model(self.parameters['model_file'])
        self.bounds = self.initialize_bounds()
        self.initial_upper_bound = None # an attribute to hold the initial upper bound
        self.initial_lower_bound = None

    def initialize_bounds(self):
        bounds = {}
        for reaction in self.model.reactions:
            bounds[reaction.id] = (reaction.lower_bound, reaction.upper_bound)
        return bounds

    def ports_schema(self):
        return {
            "reaction_bounds": {
                '_default': self.bounds,
                '_emit': True,
                '_updater': 'set'
            },
            "current_v0": {
                '_default': -10.0,
                '_emit': True,
                '_updater': 'set'
            },
            "concentration": {
                '_default': self.parameters['init_concentration'],
                '_emit': True,
                "_updater": "accumulate",
            },
            "enz_concentration": {
                "_default": self.parameters['enz_concentration'],
                "_updater": "set"
            },
        }

    def next_update(self, timestep, state):
        concentration = state['concentration']
        enz_concentration = state['enz_concentration']  # Use 'enz_concentration' instead of 'enz-conc'
        vmax = self.parameters['kcat'] * enz_concentration
        current_v0 = vmax * concentration / (self.parameters['km'] + concentration)
        current_v0 = - current_v0  # it should be negative because it consumes glucose
        updated_bounds = self.bounds
        if timestep != 0:
            current_bounds = state["reaction_bounds"]
            for reaction_id, old_bounds in current_bounds.items():
                if reaction_id == "EX_glc__D_e":  # it will be always none if the class was Step instead of Process. why?
                    if self.initial_lower_bound is None:  # If the initial lower bound has not been stored yet
                        self.initial_lower_bound = old_bounds[0]  # Store the initial lower bound
                    new_lower_bound = max(self.initial_lower_bound, current_v0)  # Take the maximum of the initial lower bound and current_v0
                    new_bounds = (new_lower_bound, old_bounds[1])  # keep the old upper bound
                    updated_bounds[reaction_id] = new_bounds

        return {
            "reaction_bounds": updated_bounds,
            "current_v0": current_v0,
        }


class DynamicFBA(Process):
    """
    This class conducts the flux balance analysis for the model.

    DynamicFBA accepts an SBML model and the reaction bounds. It calculates flux balance analysis and optimizes it at each update, producing the fluxes and objective value as output.
    """

    defaults = {'reaction_bounds': None}

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters)
        self.model = read_sbml_model(self.parameters["model_file"])
        self.reaction_bounds = self.parameters['reaction_bounds']

    def ports_schema(self):
        return {
            "fluxes": {
                '_default': {},
                '_emit': True,
                "_updater": "set"
            },
            "reactions": {
                '_default': [str(reaction) for reaction in self.model.reactions],
                '_emit': True,
                "_updater": "set"
            },
            "objective_flux": {
                '_default': 0.0,
                '_emit': True,
                "_updater": "set"
            },
            "reaction_bounds": {
                '_default': self.reaction_bounds.bounds if self.reaction_bounds else {},
                '_emit': True,
                "_updater": "set"
            }
        }

    def next_update(self, timestep, state):
        reaction_bounds = state["reaction_bounds"]
        for reaction_id, (lower_bound, upper_bound) in reaction_bounds.items():
            reaction = self.model.reactions.get_by_id(reaction_id)
            reaction.lower_bound, reaction.upper_bound = lower_bound, upper_bound

        solution = self.model.optimize()
        objective_value = solution.objective_value
        fluxes = solution.fluxes.to_dict()

        return {
            "fluxes": fluxes,
            "objective_flux": objective_value
        }



class BiomassCalculator(Process):
    """
This class computes the current biomass based on the objective flux.

BiomassCalculator takes in the initial objective flux and estimates the current biomass at each update. The calculation is based on the objective flux, the time proportion, and the current biomass.
"""

    defaults = {'initial_objective_flux': None}

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters)
        self.current_biomass = self.parameters['initial_objective_flux']


    def compute_biomass(self, objective_flux, time_proportion):
        self.current_biomass += objective_flux * time_proportion * self.current_biomass
        return self.current_biomass

    def ports_schema(self):
        return {
            "objective_flux": {
                '_default': 0.0,
                "_updater": "set"
            },
            "current_biomass": {
                '_default': 0.0,
                '_emit': True,
                "_updater": "set"
            }
        }

    def next_update(self, timestep, state):
        objective_flux = state["objective_flux"]
        current_biomass = self.compute_biomass(objective_flux, TIME_PROPORTION)
        return {
            "current_biomass": current_biomass
        }


class EnvCalculator(Process):
    """
    This class estimates the environmental consumption (Glucose in this case) and concentration.

    EnvCalculator calculates the environmental consumption using the current biomass and flux values. It also updates the concentration.
    """

    defaults = {
        'init_concentration': 11.1,
        'volume': 10,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters=parameters)

    def ports_schema(self):
        return {
            "current_biomass": {
                "_default": 0.0,
            },
            "fluxes_values": {
                "_default": {}
            },
            "current_env_consumption": {
                "_default": 0.0,
                "_emit": True,
                "_updater": "accumulate",
            },
            "concentration": {
                "_default": self.parameters['init_concentration'],
                "_emit": True,
                "_updater": "accumulate",
            },
        }

    def next_update(self, timestep, state):
        current_biomass = state["current_biomass"]
        concentration = state["concentration"]
        fluxes_values = state["fluxes_values"]
        env_consumption = 0
        if "EX_glc__D_e" in fluxes_values:
            env_consumption = (
                    current_biomass * (-fluxes_values["EX_glc__D_e"]) * TIME_PROPORTION
            )
        delta_c = env_consumption / self.parameters['volume']
        concentration += (-delta_c)

        return {
            "current_env_consumption": env_consumption,
            "concentration": - delta_c,
        }



def main(config):
    """
    This function runs the simulation for a specified duration.

    It initializes the ReactionBounds, DynamicFBA, BiomassCalculator, EnvCalculator,
    establishes the topology between these processes, and executes the simulation for the specified timeframe.
    """
    initial_state = config['main']['initial_state']
    simulation_time = config['main']['simulation_time']
    reaction_bounds = ReactionBounds(config['ReactionBounds'])
    dynamic_fba = DynamicFBA(config['DynamicFBA'])
    initial_state = {"reaction_bounds": reaction_bounds.bounds}
    initial_objective_flux_update = dynamic_fba.next_update(1, initial_state)
    initial_objective_flux = initial_objective_flux_update["objective_flux"]
    biomass_calculator = BiomassCalculator({'initial_objective_flux': initial_objective_flux})
    env_calculator = EnvCalculator(config['EnvCalculator'])

    processes = {
        'ReactionBounds': reaction_bounds,
        'DynamicFBA': dynamic_fba,
        'BiomassCalculator': biomass_calculator,
        'EnvCalculator': env_calculator,
    }

    topology = {
        'ReactionBounds': {
            'reaction_bounds': ('reaction_bounds',),
            'current_v0': ('current_v0',),
            'concentration': ('concentration',),
            'enz_concentration': ('enz_concentration',)
        },
        'DynamicFBA': {
            'fluxes': ('fluxes_values',),
            'reactions': ('reactions_list',),
            'objective_flux': ('objective_flux_value',),
            'reaction_bounds': ('reaction_bounds',)
        },
        'BiomassCalculator': {
            'objective_flux': ('objective_flux_value',),
            'current_biomass': ('current_biomass_value',)
        },
        'EnvCalculator': {
            'current_biomass': ('current_biomass_value',),
            'fluxes_values': ('fluxes_values',),
            'current_env_consumption': ('current_env_consumption_value',),
            'concentration': ('concentration',)
        },
    }

    sim = Engine(processes=processes, topology=topology, initial_state=initial_state)
    sim.update(simulation_time)
    data = sim.emitter.get_data()
    output = pf(data)
    return data, output, processes, topology
