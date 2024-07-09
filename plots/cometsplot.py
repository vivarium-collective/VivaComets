import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io
import base64
from IPython.display import display, HTML
from PIL import Image
import shutil

def precompute_global_min_max(data, names, data_key):
    num_elements = len(names)
    global_min = [np.inf] * num_elements
    global_max = [-np.inf] * num_elements

    for time in data["time"]:
        time_index = int(data["time"].index(time))
        for j, element_id in enumerate(names):
            current_data = data[data_key][element_id][time_index]
            global_min[j] = min(global_min[j], np.min(current_data))
            global_max[j] = max(global_max[j], np.max(current_data))
    
    return global_min, global_max

def plot_elements_to_gif(data, total_time, element_names, data_key, temp_dir, file_prefix, include_total_biomass=False):
    valid_time_points = list(range(total_time + 1))
    num_elements = len(element_names)
    images = []

    global_min, global_max = precompute_global_min_max(data, element_names, data_key)
    
    if include_total_biomass:
        global_min.append(np.inf)
        global_max.append(-np.inf)
        for time in valid_time_points:
            time_index = int(data["time"].index(time))
            total_biomass = np.zeros_like(data[data_key][element_names[0]][time_index])
            for element_id in element_names:
                total_biomass += data[data_key][element_id][time_index]
            global_min[-1] = min(global_min[-1], np.min(total_biomass))
            global_max[-1] = max(global_max[-1], np.max(total_biomass))

    for time in valid_time_points:
        time_index = int(data["time"].index(time))
        if include_total_biomass:
            total_biomass = np.zeros_like(data[data_key][element_names[0]][time_index])
        
        num_columns = 3
        num_rows = (num_elements + (1 if include_total_biomass else 0) + num_columns - 1) // num_columns
        
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 5, num_rows * 5), squeeze=False)

        for j, element_id in enumerate(element_names):
            row = j // num_columns
            col = j % num_columns
            current_data = data[data_key][element_id][time_index]
            if include_total_biomass:
                total_biomass += current_data
            im = axs[row, col].imshow(current_data, cmap='viridis', vmin=global_min[j], vmax=global_max[j])
            axs[row, col].set_title(element_id)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)

        if include_total_biomass:
            row = num_elements // num_columns
            col = num_elements % num_columns
            im = axs[row, col].imshow(total_biomass, cmap='viridis', vmin=global_min[-1], vmax=global_max[-1])
            axs[row, col].set_title("Total Biomass")
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)

        for i in range(num_elements + (1 if include_total_biomass else 0), num_rows * num_columns):
            fig.delaxes(axs[i // num_columns, i % num_columns])

        plt.suptitle(f'Time: {time}', fontsize=16)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    save_gif(images, temp_dir, file_prefix)

def save_gif(images, temp_dir, file_prefix):
    for i, img in enumerate(images):
        imageio.imwrite(os.path.join(temp_dir, f'{file_prefix}_{i}.png'), img)

def combine_gifs(output_filename, temp_dir, num_images):
    combined_images = []
    for i in range(num_images):
        obj_flux_img = imageio.imread(os.path.join(temp_dir, f'obj_flux_{i}.png'))
        molecule_img = imageio.imread(os.path.join(temp_dir, f'molecule_{i}.png'))

        obj_flux_img = np.array(Image.fromarray(obj_flux_img).resize(molecule_img.shape[1::-1]))

        combined_img = np.vstack((obj_flux_img, molecule_img))
        combined_images.append(combined_img)

    imageio.mimsave(output_filename, combined_images, duration=0.5, loop=0)
    display_gif(output_filename)

def display_gif(output_filename):
    with open(output_filename, 'rb') as file:
        data = file.read()
        data_url = 'data:image/gif;base64,' + base64.b64encode(data).decode()
    display(HTML(f'<img src="{data_url}" alt="Combined GIF" style="max-width:100%;"/>'))
