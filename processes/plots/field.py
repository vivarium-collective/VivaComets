import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


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


def plot_fields_temporal(fields_dict, nth_timestep=1, out_dir=None, filename='fields_temporal'):
    # Determine the number of rows and columns for subplots
    num_fields = len(fields_dict)
    max_time_points = max(len(matrices) for matrices in fields_dict.values())
    num_columns = (max_time_points - 1) // nth_timestep + 1  # Calculate number of columns after applying nth_timestep

    # Create a GridSpec layout with an extra column for colorbars
    fig = plt.figure(figsize=(5 * num_columns, 5 * num_fields))
    gs = gridspec.GridSpec(num_fields, num_columns + 1, width_ratios=[*([1] * num_columns), 0.05])

    for i, (field_name, matrices) in enumerate(fields_dict.items()):
        # Determine the min and max values across all matrices for this field
        field_min = min(np.min(matrix) for matrix in matrices)
        field_max = max(np.max(matrix) for matrix in matrices)

        for j in range(0, len(matrices), nth_timestep):
            ax = plt.subplot(gs[i, j // nth_timestep])
            cax = ax.imshow(np.array(matrices[j]), cmap='viridis', interpolation='nearest', vmin=field_min, vmax=field_max)
            if j == 0:
                ax.set_ylabel(field_name)
            if i == 0:
                ax.set_title(f"Time {j}")

        # Add colorbar to the last column for each row
        cbar_ax = plt.subplot(gs[i, -1])
        fig.colorbar(cax, cax=cbar_ax)

    plt.tight_layout()

    if out_dir:
        save_fig_to_dir(fig, filename, out_dir)
    else:
        plt.show()
    return fig
