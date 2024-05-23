import Raster_Manipulator as Rm
import Plot_functions as Pf
from matplotlib.colors import LogNorm, SymLogNorm
import BD_finder


def produce_all_crop_results():
    Rm.crop_rasters_form_folder('data/Harvard_Data/raw_raster_data', 'data/Harvard_Data/ethiopia',
                                'data/shape_files/ethiopia.shp')
    Rm.project_rasters_from_folder('data/Harvard_Data/ethiopia', 'data/Harvard_Data/ethiopia_EPSG4326',
                                   'EPSG:4326')
    Rm.calculate_total_crop_energy('data/Harvard_Data/ethiopia_EPSG4326', 'data/Harvard_Data/total_crop_energyGJ.tif',
                                   clip_val=0)

    Rm.crop('data/raw_raster_data/eth_ppp_2020.tif', 'data/process_results/EthCrop_eth_ppp_2020.tif',
            'data/shape_files/ethiopia.shx', clip_val=0)
    Rm.downscale('data/process_results/EthCrop_eth_ppp_2020.tif',
                 'data/process_results/LowRes_EthCrop_eth_ppp_2020.tif',
                 'data/Harvard_Data/total_crop_energyGJ.tif')
    Rm.scale_pixels('data/process_results/LowRes_EthCrop_eth_ppp_2020.tif',
                    'data/process_results/ScaleToEdemandGJ_LowRes_EthCrop_eth_ppp_2020.tif',
                    2.2*10**(-3) * 365 * 2.5)  # GJ
    Rm.scale_pixels('data/raw_raster_data/eth_ppp_2020.tif', 'data/process_results/ScaleToEdemandGJ_EthCrop_eth_ppp_2020.tif',
                    2.2*10**(-3) * 365 * 2.5)  # GJ

    Rm.calculate_energy_surplus('data/Harvard_Data/total_crop_energyGJ.tif',
                                'data/process_results/ScaleToEdemandGJ_LowRes_EthCrop_eth_ppp_2020.tif',
                                'data/process_results/Energy_surplusGJ.tif')
    print('yey')


def process_copernicus_data():
    Rm.crop('data/raw_raster_data/cope_cropcover_band1.tif',
            'data/process_results/ethiopiaCrop_cope_cropcover_band1.tif', 'data/shape_files/ethiopia.shp',
            nan_to_0=True, data_type='float32')
    Rm.downscale('data/process_results/ScaleToEdemandGJ_eth_ppp_2020.tif',
                 'data/process_results/resamToCopernicus_ScaleToEdemandGJ_eth_ppp_2020.tif',
                 'data/process_results/ethiopiaCrop_cope_cropcover_band1.tif')
    Rm.copernicus_over_harvard('data/process_results/ethiopiaCrop_cope_cropcover_band1.tif',
                               'data/Harvard_Data/total_crop_energyGJ.tif',
                               'data/process_results/cop_over_harv.tif')
    Rm.crop_rasters_form_folder('data/process_results/ScaleToEdemandGJ_EthCrop_eth_ppp_2020.tif',
                                'data/regions_energy_demand', 'data/shape_files', nan_to_0=True)
    Rm.crop_rasters_form_folder('data/process_results/cop_over_harv.tif',
                                'data/regions_energy_production', 'data/shape_files', nan_to_0=True)


def produce_all_plots():
    Pf.plot_raster('data/process_results/ScaleToEdemandGJ_eth_ppp_2020.tif',
                   'data/plots/ScaleToEdemandGJ_eth_ppp_2020',
                   area_scale_factor=0.01,
                   shp_file='data/shape_files/ethiopia.shp',
                   color_profile=LogNorm(),
                   cmap=['gist_rainbow', 'viridis', 'terrain', 'YlGn', 'plasma', 'summer', 'Greens'],
                   clip_val=0)
    Pf.plot_raster('data/Harvard_Data/total_crop_energyGJ.tif',
                   'data/plots/total_crop_energyGJ',
                   area_scale_factor=86.055,
                   shp_file='data/shape_files/ethiopia.shp',
                   cmap=['gist_rainbow', 'viridis', 'terrain', 'YlGn', 'plasma', 'summer', 'Greens'],
                   clip_val=0)
    Pf.plot_raster('data/process_results/Energy_surplusGJ.tif',
                   'data/plots/energy_surplusGJ',
                   area_scale_factor=86.055,
                   shp_file='data/shape_files/ethiopia.shp',
                   color_profile=SymLogNorm(linthresh=10),
                   cmap=['gist_rainbow', 'viridis', 'terrain', 'YlGn', 'plasma', 'summer', 'Greens'])
    Pf.plot_raster('data/process_results/cop_over_harv.tif',
                   'data/plots/cop_over_harv',
                   area_scale_factor=0.01,
                   color_profile=SymLogNorm(linthresh=10),
                   shp_file='data/shape_files/ethiopia.shp',
                   cmap=['gist_rainbow', 'viridis', 'terrain', 'YlGn', 'plasma', 'summer', 'Greens'])
    Pf.plot_rasters_from_folder('data/regions_energy_production',
                                'data/plots/regions/energy_production',
                                area_scale_factor=0.01,
                                color_profile=SymLogNorm(linthresh=10),
                                shp_file='data/shape_files/ethiopia.shp',
                                cmap=['gist_rainbow', 'viridis', 'terrain', 'YlGn', 'plasma', 'summer', 'Greens'],
                                json_folder='data/cod_data')
    Pf.plot_rasters_from_folder('data/regions_energy_demand',
                                'data/plots/regions/energy_demand',
                                area_scale_factor=0.01,
                                color_profile=SymLogNorm(linthresh=10),
                                shp_file='data/shape_files/ethiopia.shp',
                                cmap=['gist_rainbow', 'viridis', 'terrain', 'YlGn', 'plasma', 'summer', 'Greens'],
                                json_folder='data/cod_data', clip_val=0)
    Pf.line_graph_from_folder('data/cod_data', 'data/plots/production_radius_line_graph', colors=['blue', 'lime', 'green'])
    Pf.e_demand_and_product_rad_from_folder('data/cod_data',
        'data/plots/production_radius_and_energy_coverage_line_graph', colors=['blue', 'lime', 'green'])
    Pf.plot_raster_and_cods_from_folder('data/regions_energy_demand', 'data/plots/cods', 'data/cod_data',
                                        area_scale_factor=0.01, color_profile=SymLogNorm(linthresh=10),
                                        shp_file='data/shape_files/ethiopia.shp',
                                        cmap=['gist_rainbow', 'viridis', 'terrain', 'plasma', 'summer'],
                                        clip_val=0)
    Pf.plot_raster_and_cods_from_folder('data/regions_energy_demand', 'data/plots/cods_reduced', 'data/cod_data',
                                        area_scale_factor=0.01, color_profile=SymLogNorm(linthresh=10),
                                        shp_file='data/shape_files/ethiopia.shp',
                                        cmap=['gist_rainbow', 'viridis', 'terrain', 'plasma', 'summer'],
                                        clip_val=0, cutoff_radius=50)


def find_centers_of_demand():
    BD_finder.find_cods_from_folder('data/regions_energy_demand', 'data/cod_data', [50, 100, 200])
    BD_finder.find_crop_radius_from_folder('data/regions_energy_demand', 'data/regions_energy_production',
                                           'data/cod_data', [1])
    BD_finder.calc_demand_coverage_from_folder('data/regions_energy_demand', 'data/cod_data')
    BD_finder.refine_crop_radius_from_folder('data/regions_energy_production', 'data/cod_data')


if __name__ == '__main__':
    produce_all_crop_results()
    process_copernicus_data()
    find_centers_of_demand()
    produce_all_plots()
