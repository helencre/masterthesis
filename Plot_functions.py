import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm, CenteredNorm
import os
import fiona
import rasterio.mask
import Raster_Manipulator as Rm
import json
import BD_finder
import matplotlib.colors as mcolors


def crop_none_outside(source, shp_file):
    with fiona.open(shp_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    with rasterio.open(source) as src:
        masked_data, _ = rasterio.mask.mask(src, shapes, crop=True, filled=False)
    output = masked_data[0]
    return output


def get_json_data(source, json_folder):
    base = os.path.basename(source)
    region = os.path.splitext(base)[0]

    for file in os.listdir(json_folder):
        filename = os.fsdecode(file)
        if filename.startswith(region):
            with open(json_folder, 'r') as j_dict:
                data = json.load(j_dict)
    return data


def plot_rasters_from_folder(source_folder, save_folder, area_scale_factor=1, shp_file=None, color_profile=None,
                             cmap=['viridis'], res=400, clip_val=-np.inf, extension='.pdf', json_folder=None):
    for file in os.listdir(source_folder):
        save = save_folder + '/' + os.path.splitext(file)[0]
        plot_raster(source_folder + '/' + file, save, area_scale_factor, shp_file, color_profile, cmap, res, clip_val,
                    extension)


def plot_raster(source, save_dir, area_scale_factor=1, shp_file=None, color_profile=None, cmap=['viridis'], res=400,
                clip_val=-np.inf, extension='.pdf'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if shp_file is not None:
        raster_data = crop_none_outside(source, shp_file)

    else:
        with rasterio.open(source) as src:
            # Read the raster data
            raster_data = src.read(1)

    raster_data = raster_data.clip(clip_val) / area_scale_factor
    # raster_data = np.nan_to_num(raster_data, nan=0)

    for c in cmap:
        plt.imshow(raster_data, cmap=c, norm=color_profile)
        plt.axis('off')  # Turn off the axes
        # Add color legend
        plt.colorbar(shrink=0.5)  # Adjust fraction and pad as needed
        plt.savefig(save_dir + '/' + c + extension, dpi=res)
        plt.clf()
        print('Plot saved to: ', save_dir + '/' + c + extension)


def plot_raster_and_cods_from_folder(source_folder, save_dir, json_folder, area_scale_factor=1, shp_file=None,
        color_profile=None, cmap=['viridis'], res=400, clip_val=-np.inf, extension='.pdf', dr_scaler=0.1, pr_scaler=0.84, cutoff_radius=np.inf):
    tot_steps = len(os.listdir(source_folder) * len(os.listdir(json_folder)))
    pd = BD_finder.ProgressBar(tot_steps)
    for demand_file in os.listdir(source_folder):
        region = os.path.splitext(demand_file)[0]
        for json_file in os.listdir(json_folder):
            pd.update()
            if json_file.startswith(region):
                save = save_dir + '/' + region
                plot_raster_and_cods(source_folder + '/' + demand_file, save, json_folder + '/' + json_file,
                    area_scale_factor, shp_file, color_profile, cmap, res, clip_val, extension, dr_scaler, pr_scaler, cutoff_radius)


def plot_raster_and_cods(source, save_dir, json_file, area_scale_factor=1, shp_file=None, color_profile=None,
                         cmap=['viridis'], res=400, clip_val=-np.inf, extension='.pdf', distnace_scaler=0.1, pr_scaler=0.84, cutoff_radius=np.inf):

    if shp_file is not None:
        raster_data = crop_none_outside(source, shp_file)

    else:
        with rasterio.open(source) as src:
            # Read the raster data
            raster_data = src.read(1)

    total_energy = int(np.sum(raster_data.clip(clip_val)))

    raster_data = raster_data.clip(clip_val) / area_scale_factor
    # raster_data = np.nan_to_num(raster_data, nan=0)

    with open(json_file, 'r') as file:
        data = json.load(file)
    cods = np.array(data['centroids'])
    for i in range(len(data['demand_radius'])):
        for c in cmap:
            plt.imshow(raster_data, cmap=c, norm=color_profile)
            plt.axis('off')  # Turn off the axes
            # Add color legend
            plt.colorbar(shrink=0.5)  # Adjust fraction and pad as needed
            # plt.scatter(cods[:, 0], cods[:, 1], color='red', label='COD')
            total_energy_coverage = 0
            for (x, y), production_radius, energy_coverage in zip(cods, data['production_radius'][i], data['energy_coverage'][i]):
                if production_radius * pr_scaler < cutoff_radius:
                    circle = plt.Circle((x, y), production_radius * pr_scaler, color='green', fill=False)
                    plt.gca().add_artist(circle)
                    circle = plt.Circle((x, y), data['demand_radius'][i], color='red', fill=False)
                    plt.gca().add_artist(circle)
                    total_energy_coverage += energy_coverage
            save = save_dir + '/nr_clusters' + str(len(data['production_radius'][i])) + '_demandR' + str(data['demand_radius'][i])
            if not os.path.exists(save):
                os.makedirs(save)
            plt.tight_layout()
            plt.savefig(save + '/' + c + extension, dpi=res)
            plt.clf()
            if cutoff_radius < np.inf:
                data_text = 'Energy demand covered by biodigesters for a demand radius: ' + str(data['demand_radius'][i] * distnace_scaler)\
                            +' km is '+ str(int(total_energy_coverage)) + ' GJ per year \nThe total enerygy demand of the region is ' + str(total_energy) + ' GJ per year'
                with open(save + '/data.txt', 'w') as file:
                    file.write(data_text)


def line_graph_from_folder(source_folder, save_folder, d_scalar=0.1, p_scalar=0.084, extension='.pdf', res=400, colors=[]):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for file in os.listdir(source_folder):
        save = save_folder + '/' + os.path.splitext(file)[0]
        line_graph(source_folder + '/' + file, save, d_scalar, p_scalar, extension, res, colors)


def line_graph(json_file, save, d_scalar=0.1, p_scalar=0.084, extension='.pdf', res=400, colors=[]):
    colors += list(mcolors.CSS4_COLORS.keys())

    with open(json_file, 'r') as file:
        data = json.load(file)

    sorted_p_r = []
    labels = []

    for p_line, d_r in zip(data['production_radius'], data['demand_radius']):
        sorted_p_r.append(sorted(p_line + [0]))
        labels.append('Demand radius ' + str(d_r * d_scalar) + 'km')

    scaled_sorted_p_r = [[element * p_scalar for element in sublist] for sublist in sorted_p_r]

    for plot_list, color, label in zip(scaled_sorted_p_r, colors, labels):
        plt.plot(np.array(plot_list).T, label=label, color=color)
    plt.xlabel('Number of CODs')
    plt.ylabel('Radius of area for required crop waste (km)')
    plt.legend()
    plt.savefig(save + extension, dpi=res)
    plt.clf()


def e_demand_and_product_rad_from_folder(source_folder, save_folder, d_scalar=0.1, p_scalar=0.084, extension='.pdf', res=400,colors=[]):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for file in os.listdir(source_folder):
        save = save_folder + '/' + os.path.splitext(file)[0]
        e_demand_and_product_rad(source_folder + '/' + file, save, d_scalar, p_scalar, extension, res, colors)


def e_demand_and_product_rad(json_file, save, d_scalar=0.1, p_scalar=0.084, extension='.pdf', res=400, colors=[]):
    # if not os.path.exists(save):
    #     os.makedirs(save)

    colors += list(mcolors.CSS4_COLORS.keys())

    with open(json_file, 'r') as file:
        data = json.load(file)
    sorted_p_r = []
    sorted_e_d = []
    labels = []

    for p_line, e_demand, d_r in zip(data['production_radius'], data['energy_coverage'], data['demand_radius']):
        paired_lists = list(zip(p_line, e_demand))

        # Sort the list of tuples based on the first element of each tuple (elements of list1)
        paired_lists_sorted = sorted(paired_lists, key=lambda x: x[0])

        # Unzip the sorted list of tuples back into two lists
        list1_sorted, list2_sorted = zip(*paired_lists_sorted)

        # Convert the results back to lists
        p_line_sorted = list(list1_sorted)
        e_demand_sorted = list(list2_sorted)

        cumulative_sum = 0
        e_demand_acumul = []
        for value in e_demand_sorted:
            cumulative_sum += value
            e_demand_acumul.append(cumulative_sum)

        sorted_p_r.append([0] + p_line_sorted)
        sorted_e_d.append([0] + e_demand_acumul)

        labels.append('Demand radius ' + str(d_r * d_scalar) + 'km')

    scaled_sorted_p_r = [[element * p_scalar for element in sublist] for sublist in sorted_p_r]

    fig, ax1 = plt.subplots()
    flattened_list = [item for sublist in sorted_e_d for item in sublist]

    ax2 = ax1.twinx()
    for i in range(len(data['demand_radius'])):
        label = 'Demand radius ' + str(data['demand_radius'][i] * d_scalar) + 'km'
        ax1.plot(np.array(sorted_p_r[i]).T * p_scalar, label=label, color=colors[i])
        ax2.plot(np.array(sorted_e_d[i]).T, linestyle='--', color=colors[i])

    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.set_xlabel('Number of CODs')
    ax1.tick_params(axis='y')
    ax1.set_ylabel('Radius of area for required crop waste  (km) (Solid line)')
    ax2.tick_params(axis='y')
    ax2.set_ylabel('Energy demand coverage (GJ/year) (Dashed line)')
    ax1.legend()

    plt.savefig(save + extension, dpi=res)


def progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = f'{(100 * (iteration / float(total))):.1f}'
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    # Print a new line on completion
    if iteration == total:
        print()
