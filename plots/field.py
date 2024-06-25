import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io
import base64
from IPython.display import display, HTML


def plot_fields_temporal_to_gif(
        fields_data,
        species_data,
        desired_time_points,
        actual_time_points,
        filename='fields_over_time.gif',
        molecule_colormaps={'glucose': 'YlOrBr', 'acetate': 'Blues', 'oxygen': 'Blues'},  # Specify color maps for each molecule
        species_colormaps={'Thermotoga': 'Greens', 'ecoli': 'Reds'},  # Specify color maps for each species
        plot_fields=None,
        plot_species=None,
        skip_frames=1,  # TODO -- maybe this should be used to skip frames
):
    if not os.path.exists('temp'):
        os.makedirs('temp', exist_ok=True)

    if not plot_fields:
        plot_fields = list(fields_data.keys())
    if not plot_species:
        plot_species = list(species_data.keys())

    # Convert desired and actual time points to float for accurate indexing
    desired_time_points = [float(time) for time in desired_time_points]
    actual_time_points = [float(time) for time in actual_time_points]
    num_molecules = len(plot_fields) if plot_fields else len(fields_data)
    num_species = len(plot_species) if plot_species else len(species_data)
    num_times = len(desired_time_points)
    images = []

    # Calculate global min/max for each molecule and species across all timepoints
    global_min_max = {}
    for data_dict, colormaps, plot_items in zip(
            [fields_data, species_data],
            [molecule_colormaps, species_colormaps],
            [plot_fields, plot_species]):
        for item in data_dict.keys():
            if item not in plot_items:
                continue
            all_data = np.concatenate([np.array(times_data) for times_data in data_dict[item]], axis=0)
            global_min_max[item] = (np.min(all_data), np.max(all_data))

    for time_idx, desired_time in enumerate(desired_time_points):
        if desired_time in actual_time_points:
            actual_idx = actual_time_points.index(desired_time)
            
            # Calculate the number of rows needed
            num_columns = 3
            total_plots = num_molecules + num_species
            num_rows = (total_plots + num_columns - 1) // num_columns  # Ceiling division
            
            fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5), squeeze=False)

            for mol_idx, molecule in enumerate(plot_fields):
                row = mol_idx // num_columns
                col = mol_idx % num_columns
                data_array = np.array(fields_data[molecule][actual_idx])  # Accessing the time-specific data
                ax = axs[row, col]
                cmap = molecule_colormaps.get(molecule, 'viridis')  # Default to 'viridis' if molecule not in dict
                vmin, vmax = global_min_max[molecule]  # Use global min/max
                cax = ax.imshow(data_array, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
                ax.set_title(f'{molecule} at Time {desired_time}', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                # Add a colorbar for each subplot
                cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=10)

            for spec_idx, species in enumerate(plot_species):
                row = (num_molecules + spec_idx) // num_columns
                col = (num_molecules + spec_idx) % num_columns
                data_array = np.array(species_data[species][actual_idx])  # Accessing the time-specific data
                ax = axs[row, col]
                cmap = species_colormaps.get(species, 'viridis')  # Default to 'viridis' if species not in dict
                vmin, vmax = global_min_max[species]  # Use global min/max
                cax = ax.imshow(data_array, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
                ax.set_title(f'{species} at Time {desired_time}', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                # Add a colorbar for each subplot
                cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=10)

            # Hide any unused subplots
            for i in range(total_plots, num_rows * num_columns):
                fig.delaxes(axs[i // num_columns, i % num_columns])
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            images.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)

    #imageio.mimsave(filename, images, fps=5)

    # Create and save the GIF with loop=0 for infinite loop
    imageio.mimsave(filename, images, duration=0.5, loop=0)

    # Optionally display the GIF in a Jupyter notebook
    with open(filename, 'rb') as file:
        data = file.read()
        data_url = 'data:image/gif;base64,' + base64.b64encode(data).decode()
    display(HTML(f'<img src="{data_url}" alt="Fields Over Time" style="max-width:100%;"/>'))

def save_fig_to_dir(
        fig,
        filename,
        out_dir='out/',
):
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, filename)
    print(f"Writing {fig_path}")
    fig.savefig(fig_path, bbox_inches='tight')


def plot_field(matrix, out_dir=None, filename='field'):
    matrix = np.array(matrix)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax)

    if out_dir:
        save_fig_to_dir(fig, filename, out_dir)
    else:
        plt.show()
    return fig


def plot_fields(fields_dict, out_dir=None, filename='fields'):
    num_fields = len(fields_dict)
    fig, axes = plt.subplots(1, num_fields, figsize=(5 * num_fields, 5))  # Adjust subplot size as needed

    if num_fields == 1:
        axes = [axes]  # Make axes iterable for a single subplot

    for ax, (field_name, matrix) in zip(axes, fields_dict.items()):
        cax = ax.imshow(np.array(matrix), cmap='viridis', interpolation='nearest')
        ax.set_title(field_name)
        fig.colorbar(cax, ax=ax)

    plt.tight_layout()

    if out_dir:
        save_fig_to_dir(fig, filename, out_dir)
    else:
        plt.show()
    return fig

def plot_objective_flux(
        data,
        time_points,
        species_names,
        out_dir='out',
        filename='objective_flux'
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_species = len(species_names)
    num_times = len(time_points)
    fig, axs = plt.subplots(num_times, num_species + 1, figsize=(num_species * 5, num_times * 5), squeeze=False)

    # Calculate global min and max for each species and total biomass
    global_min = [np.inf] * (num_species + 1)  # +1 for total biomass
    global_max = [-np.inf] * (num_species + 1)

    # Precompute global min/max for species and total biomass
    for time in time_points:
        time_index = data["time"].index(time)
        total_biomass = np.zeros_like(data["species"][species_names[0]][time_index])

        for j, species_id in enumerate(species_names):
            current_species = data["species"][species_id][time_index]
            total_biomass += current_species
            global_min[j] = min(global_min[j], np.min(current_species))
            global_max[j] = max(global_max[j], np.max(current_species))

        # Update total biomass global min and max
        global_min[-1] = min(global_min[-1], np.min(total_biomass))
        global_max[-1] = max(global_max[-1], np.max(total_biomass))

    # Plotting each species and total biomass for each time
    for i, time in enumerate(time_points):
        time_index = data["time"].index(time)
        total_biomass = np.zeros_like(data["species"][species_names[0]][time_index])

        for j, species_id in enumerate(species_names):
            current_species = data["species"][species_id][time_index]
            total_biomass += current_species
            im = axs[i, j].imshow(current_species, cmap='viridis', vmin=global_min[j], vmax=global_max[j])
            if i == 0:  # Set title only for the first row
                axs[i, j].set_title(species_id)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i == 0:  # Add colorbar only in the first row
                plt.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)

            if j == 0:  # Add time label to the leftmost column
                axs[i, j].set_ylabel(f'Time {time}', fontsize=12)

        # Plot total biomass in the last column
        im = axs[i, -1].imshow(total_biomass, cmap='viridis', vmin=global_min[-1], vmax=global_max[-1])
        if i == 0:  # Set title only for the first row
            axs[i, -1].set_title("Total Biomass")
        axs[i, -1].set_xticks([])
        axs[i, -1].set_yticks([])
        if i == 0:  # Add colorbar only in the first row
            plt.colorbar(im, ax=axs[i, -1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_fields_temporal(
        fields_data,
        desired_time_points,
        actual_time_points,
        out_dir='out',
        filename='fields_at_z',
        molecule_colormaps={},
        plot_fields=["glucose" , "acetate"]
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
    return fig
