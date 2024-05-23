import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import json
import os
import Raster_Manipulator as Rm
from tqdm import tqdm
from functools import reduce


class ProgressBar:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = tqdm(total=total_steps, desc='Progress', ncols=100,
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

    def update(self):
        self.progress_bar.update(1)

    def close(self):
        self.progress_bar.close()


def find_cods_from_folder(demand_source, save_folder, l_of_n_clusters):
    for demand_file in os.listdir(demand_source):
        for n_clusters in l_of_n_clusters:
            ds = demand_source + '/' + demand_file
            find_cods(ds, save_folder, n_clusters)


def find_crop_radius_from_folder(demand_source, production_source, json_source, max_demand_r_list):
    tot_steps = len(os.listdir(production_source)) * len(os.listdir(demand_source)) * len(os.listdir(json_source))
    pd = ProgressBar(tot_steps)
    for production_file in os.listdir(production_source):
        p_region = os.path.splitext(production_file)[0]
        for demand_file in os.listdir(demand_source):
            d_region = os.path.splitext(demand_file)[0]
            for json_file in os.listdir(json_source):
                j_region = os.path.splitext(json_file)[0]
                if p_region == d_region and j_region.startswith(d_region):
                    ps = production_source + '/' + production_file
                    ds = demand_source + '/' + demand_file
                    js = json_source + '/' + json_file
                    find_crop_radius(ds, ps, js, max_demand_r_list)
                    pd.update()


def calculate_multiple_centroids(energy_demand, n_clusters):
    # Flatten the raster data and create coordinate grids
    rows, cols = np.indices(energy_demand.shape)
    energy_demand_flat = energy_demand.compressed()  # Flatten the array and remove masked elements
    rows_flat = rows[~energy_demand.mask].flatten()
    cols_flat = cols[~energy_demand.mask].flatten()

    # Stack the coordinates and the energy demand values for clustering
    data = np.stack((rows_flat, cols_flat), axis=1)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data, sample_weight=energy_demand_flat)
    labels = kmeans.labels_

    # Calculate centroids of the clusters
    centroids = []
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        cluster_weights = energy_demand_flat[cluster_mask]
        cluster_rows = rows_flat[cluster_mask]
        cluster_cols = cols_flat[cluster_mask]

        total_weight = np.sum(cluster_weights)
        x_centroid = np.sum(cluster_cols * cluster_weights) / total_weight
        y_centroid = np.sum(cluster_rows * cluster_weights) / total_weight

        # Convert pixel coordinates to geographic coordinates
        # x_centroid_geo, y_centroid_geo = rasterio.transform.xy(src.transform, y_centroid, x_centroid,
        #                                                       offset='center')
        # centroids.append((x_centroid_geo, y_centroid_geo))  # original
        centroids.append((x_centroid, y_centroid))

    # plot_cods(energy_demand, np.array(centroids))
    return centroids


def plot_cods(array, cods, radii, const_r):
    # Plot the array
    plt.imshow(array, cmap='viridis', origin='lower', interpolation='nearest', norm=SymLogNorm(linthresh=10))

    # Plot the points on top of the array
    plt.scatter(cods[:, 0], cods[:, 1], color='red', label='Points')

    # Plot circles around each point
    for (x, y), radius in zip(cods, radii):
        circle = plt.Circle((x, y), radius, color='green', fill=False)
        plt.gca().add_artist(circle)
        circle = plt.Circle((x, y), const_r, color='red', fill=False)
        plt.gca().add_artist(circle)


    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Array with Points and Circles')

    # Add legend
    plt.legend()

    # Show plot
    plt.colorbar()
    plt.show()


def sum_pixels_in_circle(matrix, center, radius):
    cy, cx = center
    rows, cols = matrix.shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    pixel_sum = np.sum(matrix[mask])
    modified_matrix = matrix.copy()
    modified_matrix[mask] = 0
    return pixel_sum.item(), modified_matrix


def find_radius(matrix, center, edemand, initial_guess, tolerance=0.1):
    lower_radius = 0
    upper_radius = max(matrix.shape) // 2  # Maximum possible radius
    current_radius = initial_guess
    reduced_matrix = np.copy(matrix)
    while upper_radius - lower_radius > tolerance:
        sum_of_pixels, reduced_matrix = sum_pixels_in_circle(matrix, center, current_radius)

        if sum_of_pixels < edemand:
            lower_radius = current_radius
        else:
            upper_radius = current_radius

        current_radius = (lower_radius + upper_radius) / 2
    return current_radius, reduced_matrix


def find_cods(demand_source, save_folder, n_clusters):
    with rasterio.open(demand_source) as src:
        # Read the raster data as a numpy array
        energy_demand = src.read(1).clip(0)
        # Handle NoData values by masking them
        energy_demand = np.ma.masked_equal(energy_demand, src.nodata)

    centroids = calculate_multiple_centroids(energy_demand, n_clusters)  # a list of cods

    data = {'centroids': centroids}

    base = os.path.basename(demand_source)
    region = os.path.splitext(base)[0]

    if not os.path.exists(save_folder):
        os.makedirs(os.path.dirname(save_folder))

    save_path = save_folder + '/' + region + '_n_clusters' + str(n_clusters) + '.json'

    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def find_crop_radius(demand_source, production_source, json_file, max_demand_r_list):
    with rasterio.open(demand_source) as src:
        # Read the raster data as a numpy array
        energy_demand = src.read(1).clip(0)
        # Handle NoData values by masking them
        energy_demand = np.ma.masked_equal(energy_demand, src.nodata)

    with rasterio.open(production_source) as src:
        # Read the raster data as a numpy array
        energy_production = src.read(1).clip(0)
        # Handle NoData values by masking them
        energy_production = np.ma.masked_equal(energy_production, src.nodata)

    x_coordinate_scaler = energy_production.shape[1] / energy_demand.shape[1]
    y_coordinate_scaler = energy_production.shape[0] / energy_demand.shape[0]

    with open(json_file, 'r') as j_dict:
        data = json.load(j_dict)

    centroids = data['centroids']
    req_radius = []
    energy_coverage = []

    for max_demand_r in max_demand_r_list:
        rr = []
        reduced_ed = np.copy(energy_demand)

        e_sum_list = []
        for centroid in centroids:
            e_sum, reduced_ed = sum_pixels_in_circle(reduced_ed, (centroid[1], centroid[0]), max_demand_r)
            e_sum_list.append(e_sum)
            scaled_coordinate = (x_coordinate_scaler * centroid[1], y_coordinate_scaler * centroid[0])
            res, _ = find_radius(energy_production, scaled_coordinate, e_sum, max_demand_r)
            rr.append(res)

        req_radius.append(rr)
        energy_coverage.append(e_sum_list)

    new_data = {'production_radius': req_radius,
                'demand_radius': max_demand_r_list,
                'energy_coverage': energy_coverage}
    data.update(new_data)
    plt.imshow(reduced_ed, norm=SymLogNorm(linthresh=10))
    plt.show()
    with open(json_file, 'w') as jf:
        json.dump(data, jf, indent=4)


def refine_crop_radius_from_folder(source_folder, json_folder):
    tot_steps = len(os.listdir(source_folder) * len(os.listdir(json_folder)))
    pd = ProgressBar(tot_steps)
    for production_file in os.listdir(source_folder):
        region = os.path.splitext(production_file)[0]
        for json_file in os.listdir(json_folder):
            pd.update()
            if json_file.startswith(region):
                refine_crop_radius(source_folder + '/' + production_file, json_folder + '/' + json_file)


def refine_crop_radius(production_source, json_file, coord_scaler=0.84):
    with rasterio.open(production_source) as src:
        # Read the raster data as a numpy array
        energy_production = src.read(1).clip(0)
        # Handle NoData values by masking them
        energy_production = np.ma.masked_equal(energy_production, src.nodata)

    with open(json_file, 'r') as j_dict:
        data = json.load(j_dict)

    sorted_production_radius = []
    sorted_centroids = []
    sorted_energy_coverage = []

    for demand_index in range(len(data['demand_radius'])):

        nr_clusters = len(data['centroids'])

        paired_lists = list(zip(data['production_radius'][demand_index], data['centroids'],
                                data['energy_coverage'][demand_index], list(range(nr_clusters))))
        paired_lists_sorted = sorted(paired_lists, key=lambda x: x[0])
        production_radius, centroids, energy_coverage, indexes = zip(*paired_lists_sorted)
        production_radius = list(production_radius)
        centroids = list(centroids)
        energy_coverage = list(energy_coverage)
        indexes = list(indexes)

        reduced_ep = np.copy(energy_production)

        for pri in range(len(production_radius)):
            scaled_coordinate = (coord_scaler * centroids[pri][1], coord_scaler * centroids[pri][0])
            r, reduced_ep = find_radius(reduced_ep, scaled_coordinate, energy_coverage[pri],
                                        data['production_radius'][demand_index][indexes[pri]])
            data['production_radius'][demand_index][indexes[pri]] = r

    with open(json_file, 'w') as jf:
        json.dump(data, jf, indent=4)


def calc_demand_coverage_from_folder(source_folder, json_folder):
    tot_steps = len(os.listdir(source_folder) * len(os.listdir(json_folder)))
    pd = ProgressBar(tot_steps)
    for demand_file in os.listdir(source_folder):
        region = os.path.splitext(demand_file)[0]
        for json_file in os.listdir(json_folder):
            pd.update()
            if json_file.startswith(region):
                calc_demand_coverage(source_folder + '/' + demand_file, json_folder + '/' + json_file)


def calc_demand_coverage(demand_source, json_file):
    with rasterio.open(demand_source) as src:
        # Read the raster data as a numpy array
        energy_demand = src.read(1).clip(0)
        # Handle NoData values by masking them
        energy_demand = np.ma.masked_equal(energy_demand, src.nodata)
    with open(json_file, 'r') as j_dict:
        data = json.load(j_dict)

    energy_coverage = {'energy_coverage': []}

    for max_demand_r in data['demand_radius']:
        reduced_ed = np.copy(energy_demand)
        e_sum_list = []
        for centroid in data['centroids']:
            e_sum, reduced_ed = sum_pixels_in_circle(reduced_ed, (centroid[1], centroid[0]), max_demand_r)
            e_sum_list.append(e_sum)
        energy_coverage['energy_coverage'].append(e_sum_list)

    data.update(energy_coverage)

    with open(json_file, 'w') as jf:
        json.dump(data, jf, indent=4)
