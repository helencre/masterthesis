import fiona
import rasterio
import rasterio.mask
import os
import numpy as np
import matplotlib.pyplot as plt
import shapefile
from shapely.geometry import box
from rasterio.warp import reproject, Resampling


# os.chdir("path_to_your_desired_working_directory")

energy_crop_multiplier = {
        'Maize': 0.43*0.31*0.95*0.66*17.5*1000,
        'Wheat': 0.43*0.85*0.95*0.55*17.4*1000,
        'Sorghum': 0.43*0.31*0.95*0.66*17.5*1000,
        'Barley': 0.43*0.85*0.95*0.55*17.4*1000,
        'Millet': 0.43*0.85*0.95*0.55*17.4*1000
        # Giga Joule / (ton * year)
    }


def project_raster(source, save, projection_profile):
    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))
    """Opens one raster tif file and projects it onto the default or provided profile. A Save folder path
    can be """
    with rasterio.open(source) as src:
        # source.crs = "EPSG:4326"  # This is the crs of the rpcs
        out_meta = src.meta

        out_image = np.zeros(src.shape)

        _, dst_transform = reproject(
            rasterio.band(src, 1),
            out_image,
            rpcs=src.rpcs,
            src_crs=src.crs,
            dst_crs=projection_profile,
            resampling=Resampling.nearest,
            #**kwargs
        )

    with rasterio.open(save, "w", **out_meta) as dest:
        dest.write(out_image, 1)


def project_rasters_from_folder(source, save, projection_profile):
    if not os.path.exists(save):
        os.makedirs(save)

    data_directory = os.fsencode(source)
    for file in os.listdir(data_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tif"):
            project_raster(source + '/' + filename, save + '/' + filename, projection_profile)
            continue
        else:
            continue


def calculate_total_crop_energy(source, save, clip_val=None):

    # creates the sve folder if it doesn't already exist
    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    data_directory = os.fsencode(source)

    energy_array_sum = None
    out_meta = None

    for file in os.listdir(data_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tif"):
            raster = rasterio.open(source + '/' + filename)
            out_meta = raster.meta
            energy_array = raster.read(1)

            if clip_val is not None:
                energy_array = energy_array.clip(clip_val)

            for key in energy_crop_multiplier.keys():
                if key in filename:
                    energy_array = energy_array * energy_crop_multiplier[key]

            if energy_array_sum is None:
                energy_array_sum = energy_array
            else:
                energy_array_sum += energy_array
            continue
        else:
            continue

    with rasterio.open(save, "w", **out_meta) as dest:
        dest.write(energy_array_sum, 1)


def crop_rasters_form_folder(source_path, save, shp_file, clip_val=None, nan_to_0=False):

    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    if os.path.isdir(source_path):
        data_directory = os.fsencode(source_path)
    else:
        data_directory = os.fsencode(shp_file)

    for file in os.listdir(data_directory):
        filename = os.fsdecode(file)
        if filename.endswith('.tif'):
            crop(source_path + '/' + filename, save + '/' + filename, shp_file, clip_val, nan_to_0=nan_to_0)
        elif filename.endswith('.shp'):
            crop(source_path, save + '/' + filename[:-4] + '.tif', shp_file + '/' + filename, clip_val, nan_to_0=nan_to_0)


def crop(source, save, shp_file, clip_val=None, nan_to_0=False, data_type=None):

    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    with fiona.open(shp_file, "r") as sf:
        shapes = [feature["geometry"] for feature in sf]

    with rasterio.open(source) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    if clip_val is not None:
        out_image = out_image.clip(clip_val)

    if nan_to_0:
        out_image = np.nan_to_num(out_image, nan=0)

    if data_type is not None:
        out_meta['dtype'] = data_type

    with rasterio.open(save, "w", **out_meta) as dest:
        dest.write(out_image)


def scale_pixels(source, save, pixel_scale_factor):

    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    with rasterio.open(source) as dataset:
        out_image = dataset.read(1)
        out_image = out_image * pixel_scale_factor
        out_meta = dataset.meta

    with rasterio.open(save, "w", **out_meta) as dest:
        dest.write(out_image, 1)


def calculate_energy_surplus(source, demand_energy, save):
    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    with rasterio.open(source) as dataset1:
        crops_energy_array = dataset1.read(1)
        out_meta = dataset1.meta

    with rasterio.open(demand_energy) as dataset2:
        demand_energy_array = dataset2.read(1)

    energy_surplus_array = crops_energy_array - demand_energy_array

    with rasterio.open(save, "w", **out_meta) as dest:
        dest.write(energy_surplus_array, 1)


def crop_outside_val(source, shp_file):
    with fiona.open(shp_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    with rasterio.open(source) as src:
        masked_data, _ = rasterio.mask.mask(src, shapes, crop=True, filled=False)
    output = masked_data[0]
    return output


def plot(source, log=False, colors='viridis'):
    raster_data = crop_outside_val(source, 'data/shape_files/ethiopia.shp')
    #with rasterio.open(source) as src:
    #    # Read metadata of source raster
    #    src_profile = src.profile
    #    out_meta = src.meta
    #    raster_data = src.read(1)  # Reading a single band, adjust if needed

    plt.imshow(raster_data, cmap=colors)
    # plt.axis('off')  # Turn off the axes
    # Add color legend
    plt.colorbar(shrink=0.5)  # Adjust fraction and pad as needed
    plt.show()

    plt.clf()


def create_rectangular_shp(save_name, min_x, min_y, max_x, max_y):
    # Create a box geometry representing the rectangular region
    rectangular_region = box(min_x, min_y, max_x, max_y)

    # Create a shapefile
    shp = shapefile.Writer(save_name, shapeType=shapefile.POLYGON)

    # Add fields if needed
    # shp.field("Field1", "C", 50)

    # Add the geometry to the shapefile
    shp.poly([rectangular_region.exterior.coords])

    # Save the shapefile
    shp.close()


def get_info(source, clip_val=-np.inf):
    with rasterio.open(source) as src:
        # Read the raster data
        raster_data = src.read()
        print('Shape of raster: ' + str(raster_data.shape))
        print('Sum of all pixels: ' + str(np.sum(raster_data.clip(clip_val))))
        # Get the transform (affine matrix)
        transform = src.transform

        # Calculate pixel size
        pixel_width = transform[0]
        pixel_height = -transform[4]  # In case the raster is flipped

        print('Pixel height: ', pixel_height)
        print('Pixel width: ', pixel_width)

        print('Max, Min: ', raster_data.max(), raster_data.min())

        print(src.meta)


def progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = f'{(100 * (iteration / float(total))):.1f}'
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    # Print a new line on completion
    if iteration == total:
        print()


def copernicus_over_harvard(copernicus, harvard, save):
    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    with rasterio.open(copernicus) as cop:
        cop_data = cop.read(1).clip(0)
        cop_profile = cop.profile

    with rasterio.open(harvard) as harv:
        harv_data = harv.read(1).clip(0)

    # Calculate scaling factors
    y_slope = cop_data.shape[0] / harv_data.shape[0]
    x_slope = cop_data.shape[1] / harv_data.shape[1]

    output = np.zeros_like(cop_data, dtype=np.float32)
    out_profile = cop_profile
    out_profile['nodata'] = -np.inf
    out_profile['dtype'] = 'float32'

    # Resample source raster to match target raster
    for x in range(harv_data.shape[1]):
        progress_bar(x, harv_data.shape[1]-1)
        for y in range(harv_data.shape[0]):
            x_lower = int(x * x_slope)
            x_upper = int(x * x_slope + x_slope)
            y_lower = int(y * y_slope)
            y_upper = int(y * y_slope + y_slope)

            cop_section_sum = np.sum(cop_data[y_lower:y_upper, x_lower:x_upper])
            if cop_section_sum == 0:
                scale_cop_section = 0
            else:
                scale_cop_section = 1 / cop_section_sum

            output[y_lower:y_upper, x_lower:x_upper] = cop_data[y_lower:y_upper, x_lower:x_upper] * harv_data[y, x] * scale_cop_section

    # Plot the resampled image
    plt.matshow(output)
    plt.show()

    # Save the resampled raster
    with rasterio.open(save, 'w', **out_profile) as resampled_src:
        resampled_src.write(output, 1)


def downscale(source, save, target):
    if not os.path.exists(os.path.dirname(save)):
        os.makedirs(os.path.dirname(save))

    with rasterio.open(source) as src:
        eth_ppp = src.read(1).clip(0)
        out_profile = src.profile

    with rasterio.open(target) as trg:
        target_data = trg.read(1).clip(0)


    # Calculate scaling factors
    y_slope = eth_ppp.shape[0] / target_data.shape[0]
    x_slope = eth_ppp.shape[1] / target_data.shape[1]

    output = np.zeros_like(target_data, dtype=np.float32)

    # out_profile['dtype'] = 'float32'
    out_profile['height'] = output.shape[0]
    out_profile['width'] = output.shape[1]

    # Resample source raster to match target raster
    for x in range(target_data.shape[1]):
        progress_bar(x, target_data.shape[1]-1)
        for y in range(target_data.shape[0]):
            x_lower = int(x * x_slope)
            x_upper = int(x * x_slope + x_slope)
            y_lower = int(y * y_slope)
            y_upper = int(y * y_slope + y_slope)

            # print(x_lower, x_upper)

            output[y, x] = np.sum(eth_ppp[y_lower:y_upper, x_lower:x_upper])

    # Plot the resampled image
    # plt.imshow(output)
    # plt.show()

    # Save the resampled raster
    with rasterio.open(save, 'w', **out_profile) as resampled_src:
        resampled_src.write(output, 1)
