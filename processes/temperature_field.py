import numpy as np
from vivarium.core.process import Process
from vivarium.core.engine import Engine
import os
from plots.field import plot_fields_temporal

# Set the script directory for file operations
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

def calculate_temperature_at_depth(surface_temperature, gradient, depth):
    """
    Calculate the temperature of ocean water at a given depth using a linear gradient model.

    Parameters:
    surface_temperature (float): The surface temperature in degrees Celsius.
    gradient (float): The temperature gradient in degrees Celsius per meter.
    depth (float): The depth in meters.

    Returns:
    float: The temperature at the given depth.
    """
    # Calculate the temperature at the given depth
    temperature_at_depth = surface_temperature - (gradient * depth)
    return temperature_at_depth

class TemperatureField(Process):
    """
    TemperatureField Process to simulate the temperature distribution in a 2D grid.
    """
    defaults = {
        'surface_temperature': 25.0,  # Surface temperature in degrees Celsius
        'gradient': 0.03,  # Temperature gradient in degrees Celsius per meter
        'bounds': [10, 4],  # Grid bounds (width, height)
        'nbins': [10, 4],  # Number of bins in the grid (rows, columns)
        'start_hour': 0,  # Starting hour of the simulation
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.surface_temperature = self.parameters['surface_temperature']
        self.gradient = self.parameters['gradient']
        self.bounds = self.parameters['bounds']
        self.nbins = self.parameters['nbins']
        self.start_hour = self.parameters['start_hour']
        self.bin_size = [b / n for b, n in zip(self.bounds, self.nbins)]

    def initial_state(self, config=None):
        """Initialize the temperature grid."""
        if config is None:
            config = {}
        fields = {}

        shape = tuple(self.nbins)
        fields['temperature'] = np.zeros(shape)

        # Set initial temperature for each row based on depth
        for row in range(self.nbins[0]):
            depth = row * 20  # Depth in meters
            temp = calculate_temperature_at_depth(self.surface_temperature, self.gradient, depth)
            fields['temperature'][row, :] = temp

        return {'fields': fields}

    def ports_schema(self):
        schema = {
            'fields': {
                'temperature': {
                    '_default': np.zeros(self.nbins),
                    '_updater': 'set',
                    '_emit': True
                }
            },
            'current_hour': {
                '_value': self.start_hour,
                '_updater': 'set',
                '_emit': True
            }
        }
        return schema

    def next_update(self, timestep, states):
        current_hour = states['current_hour']
        temperature_field = states['fields']['temperature']

        # Calculate temperature variation based on time of day
        if 0 <= current_hour < 6:
            surface_temperature_variation = -0.1  # Decreasing surface temperature
        elif 6 <= current_hour < 18:
            surface_temperature_variation = 0.1  # Increasing surface temperature
        else:
            surface_temperature_variation = -0.1  # Decreasing surface temperature

        # Update the surface temperature
        updated_surface_temperature = self.surface_temperature + surface_temperature_variation

        # Calculate temperature for each row based on the updated surface temperature
        for row in range(self.nbins[0]):
            depth = row * 20  # Depth in meters
            temp = calculate_temperature_at_depth(updated_surface_temperature, self.gradient, depth)
            temperature_field[row, :] = temp

        # Increment the current hour, wrapping around to 0 after 23
        next_hour = (current_hour + 1) % 24

        return {
            'fields': {'temperature': temperature_field},
            'current_hour': next_hour
        }

def run_temperature_simulation(surface_temperature=25.0, gradient=0.03, total_time=120, start_hour=0):
    config = {
        'surface_temperature': surface_temperature,
        'gradient': gradient,
        'bounds': [10, 4],
        'nbins': [10, 4],
        'start_hour': start_hour,
    }

    # Create the temperature field process and simulation engine
    temperature_field = TemperatureField(config)
    initial_state = temperature_field.initial_state()
    sim = Engine(
        initial_state=initial_state,
        processes={'temperature_process': temperature_field},
        topology={'temperature_process': {
            'fields': ('fields',),
            'current_hour': ('current_hour',),
        }}
    )

    # Run the simulation
    sim.update(total_time)

    # Get the results
    data = sim.emitter.get_timeseries()
    time_list = list(range(total_time + 1))

    # Plot the temperature results
    plot_fields_temporal(
        fields_data=data['fields'],
        desired_time_points=time_list,
        actual_time_points=data['current_hour'],
        out_dir='./out',
        filename='Temperature_test',
        molecule_colormaps={'temperature': 'coolwarm'},
        plot_fields=['temperature']
    )

if __name__ == '__main__':
    run_temperature_simulation(surface_temperature=25.0, gradient=0.03, total_time=120, start_hour=0)
