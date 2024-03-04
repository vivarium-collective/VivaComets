import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ast  # for safely evaluating the string representation of lists


def plot_simulation_data(
        simulation_data,
        num_rows,
        num_cols,
        skip_paths=None,
        out_dir='out',
        filename=None
):
    skip_paths = skip_paths or []

    # Function to check if the current path should be skipped

    def should_skip(current_path):
        return any(all(k == current_path[i] for i, k in enumerate(path)) for path in skip_paths if len(path) <= len(current_path))

    # Function to recursively process the nested dictionary
    def process_node(node, path, ax):
        if should_skip(path):
            return  # Skip this path

        if isinstance(node, dict):
            for key, value in node.items():
                process_node(value, path + [key], ax)
        else:
            # Assuming 'node' is now a list or similar iterable of data points
            ax.plot(node, label=' - '.join(path))

    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(num_cols * 4.5, num_rows * 3))
    gs = GridSpec(num_rows, num_cols, figure=fig)

    # Dictionary to store all the plot handles for legend
    plot_handles = {}

    # Index for current cell
    cell_index = 0
    for cell_id, data in simulation_data.items():
        # Parse the cell_id to get x, y
        x, y = ast.literal_eval(cell_id)
        ax = fig.add_subplot(gs[x, y])
        ax.set_title(f'Cell ID: {cell_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_yscale('log')  # log scale

        process_node(data, [], ax)

        # Update handles for legend
        handles, labels = ax.get_legend_handles_labels()
        plot_handles.update(dict(zip(labels, handles)))

        cell_index += 1

    # Add a shared legend to the figure
    # Adjust the location and font size of the legend
    fig.legend(
        plot_handles.values(),
        plot_handles.keys(),
        loc='center right',
        # bbox_to_anchor=(1.00, 0.5),
        fontsize='large'
    )

    # Adjust layout to accommodate the external legend
    # rect : tuple (left, bottom, right, top)
    fig.tight_layout(rect=[0, 0, 0.8, 1])

    if filename:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        fig.savefig(fig_path, bbox_inches='tight')

    return fig
