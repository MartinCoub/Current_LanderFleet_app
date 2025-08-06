#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:55:47 2025

@author: pascalecoubard
"""
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, butter, filtfilt


#%% Classical functions


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on the Earth
    using the Haversine formula.

    INPUT:
    - lat1, lon1: latitude and longitude of the first point (in degrees)
    - lat2, lon2: latitude and longitude of the second point (in degrees)

    OUTPUT:
    - distance in kilometers (float)
    """
    R = 6371.0  # Average radius of Earth in kilometers
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    
    return distance


def direction_speed_from_u_v(u,v): 
    direction = (np.arctan2(u, v) * 180 / np.pi) %360
    speed = np.sqrt(u**2 + v**2)
    return direction, speed


def u_v_from_direction_speed(direction, speed):
    # Direction convention: North = 0°, East = 90°
    theta_rad = np.deg2rad(direction)
    u = speed * np.sin(theta_rad)  #  East componant
    v = speed * np.cos(theta_rad)  #  North componant
    return u,v

#%% Functions for Uc and Ua & projection


def projection_baseref_base2(vector, u_new, v_new): 
    """
    Project a speed vector express in the (U,V)=(West-East, South_North) base into a new base (u_new, v_new) -> (Uc,Ua) for ex. 
   
    INPUT: 
    - vector: two float array, correspond to [u, v] of a speed vector
    - u_new: first vector of the new base [u_unew, v_unew], Uc for ex
    - v_new: first vector of the new base [u_vnew, v_vnew], Ua for ex

    OUTPUT:
    - vector_new_base: [x,y] in the base (u_new, v_new)
    """
    # New base vector expressed in the standard one 
    e1 = np.array(u_new)
    e2 = np.array(v_new)
    # Change of basis matrix
    P = np.column_stack((e1, e2))  # [[e11, e21],
                                   #  [e12, e22]]
    # Inverse of the previous matrix
    P_inv = np.linalg.inv(P)
    # Coordinates in the new base
    vector_new_base = P_inv @ vector 
    
    return vector_new_base

def projection_baseref_base2_batch(vectors, u_new, v_new):
    """
    Project a list of speed vectors expressed in the (U,V) base into a new base (u_new, v_new).
    
    INPUT:
    - vectors: array of shape (n, 2), each row is [u, v]
    - u_new: first vector of the new base [u_unew, v_unew], e.g., Uc
    - v_new: second vector of the new base [u_vnew, v_vnew], e.g., Ua

    OUTPUT:
    - vectors_new_base: array of shape (n, 2), each row is the projection in the new base
    """
    # New base vectors as matrix
    P = np.column_stack((u_new, v_new))  # Shape: (2, 2)
    P_inv = np.linalg.inv(P)  # Inverse matrix: shape (2, 2)

    # Apply transformation: (n, 2) @ (2, 2).T -> (n, 2)
    vectors = np.atleast_2d(vectors)  # Ensure input is at least 2D
    return vectors @ P_inv.T


def compute_Uc_Ua(bathy_dataset, lat_ref, lon_ref, delta_lat, delta_lon):
    """
    For a given location and a given bathymetry, provide the along slope and across slope velocities.
    
    INPUT: 
    - bathy_dataset: xarray of the bathymetry
    - lat_ref, lon_ref: : reference latitude, longitude
    - delta_lat, delta_lon: height and width of the window taken for across slope vector

    OUTPUT:  
    - slope_U_vectors:  associated base, {'Uc_ref': [compo. u, compo. v], 'Ua_ref': [compo. u, compo. v]}   
    - dz_dx: 2D tables associated to plot
    - dz_dy: 2D tables associated to plot
    
    """

    bathy_zoom = bathy_dataset.sel(
        lat=slice(lat_ref - delta_lat / 2, lat_ref + delta_lat / 2),
        lon=slice(lon_ref - delta_lon / 2, lon_ref + delta_lon / 2))

    lat = bathy_zoom['lat'].values
    lon = bathy_zoom['lon'].values
    z = bathy_zoom['elevation'].values
    
    # Compute raw gradient (∂z/∂lat, ∂z/∂lon)
    dz_dlat, dz_dlon = np.gradient(z, lat, lon)
    # Convert lat/lon degrees to meters
    lat_mean = np.mean(lat)
    deg_to_m_lat = 111320  # meters per degree latitude
    deg_to_m_lon = 111320 * np.cos(np.deg2rad(lat_mean))  # meters per degree longitude
    # Compute physical gradient (∂z/∂y and ∂z/∂x in meters)
    dz_dy = dz_dlat / deg_to_m_lat  # north-south gradient
    dz_dx = dz_dlon / deg_to_m_lon  # east-west gradient
    
    # take the mean of the gradient for all points of the window
    uacross = np.nanmean(dz_dx)
    vacross = np.nanmean(dz_dy)

    # Normalisation
    uacross_n = uacross / np.hypot(uacross, vacross)
    vacross_n = vacross / np.hypot(uacross, vacross)
    ualong_n=-vacross_n
    valong_n=uacross_n

    return {'Uc_ref': [-uacross_n, -vacross_n], 'Ua_ref': [ualong_n, valong_n]}, dz_dx, dz_dy

#%% Filter function 

def butter_lowpass_filter(dt, time, signal, cutoff_hours=30, order=4):
    """
    Apply a Butterworth low-pass filter to a time series.

    INPUT:
    dt : float, Sampling interval in seconds.
    time : array-like, Timestamps, can be datetime64 or numeric (e.g., hours).
    signal : array-like, The signal to filter (e.g., current meter data).
    cutoff_hours : float, optional, Cutoff period in hours for the low-pass filter (default is 30 hours).
    order : int, optional, Order of the Butterworth filter (default is 4).
    OUTPUT:
    filtered_signal : ndarray, The filtered signal.
    """
    # Convert time to seconds if it's datetime64
    if np.issubdtype(time.dtype, np.datetime64):
        time_seconds = (time - time[0]).astype('timedelta64[s]').astype(float)
    else:
        time_seconds = np.array(time) * 3600  # if time is in hours, convert to seconds

    # Sampling frequency in Hz
    fs = 1.0 / dt

    # Cutoff frequency in Hz (from hours), then normalize by Nyquist frequency
    cutoff_hz = 1.0 / (cutoff_hours * 3600.0)
    nyquist = 0.5 * fs
    Wn = cutoff_hz / nyquist  # normalized cutoff frequency

    # Design Butterworth filter
    b, a = butter(order, Wn, btype='low')

    # Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal




#%% Function to smooth and recover datasets for a given lander 


# # Open other datasets according to a Lander position
# def closest_buoy(lander_lat, lander_lon, buoys_data): 
#     """
#     Function that find the nearest buoy from a given Lander
#     INPUT: 
#     - lander_lat: latitude and longitude of the Lander (target coordinates)
#     - buoys_data: data with name of buoys, datasets and informations
#     OUPTUT: 
#     - closest_name: name of the closest buoy
#     - min_distance: euclidian distance of the closest buoy
#     - buoys_data[closest_name]['dataset']: dataset corresponding to the closest buoy
#     - buoys_data[closest_name]['info']: info corresponding to the closest buoy
#     """
#     closest_name = None
#     min_distance = float('inf')
#     for name, content in buoys_data.items():
#         lat, lon = content['info']['lat'], content['info']['lon']
#         # Euclidian distance
#         distance = np.sqrt((lat - lander_lat)**2 + (lon - lander_lon)**2)
#         if distance < min_distance:
#             min_distance = distance
#             closest_name = name
#     info=buoys_data[closest_name]['info']
#     ds=buoys_data[closest_name]['dataset']
#     ds = ds.assign_coords(lat=info['lat'], lon=info['lon'])
    
#     return closest_name, min_distance, ds,info

# def find_nearest_reanalysis(reanalysis_ds, lon, lat, depth, var, radius=1):
#     """
#     Find the nearest valid value in a dataset for a given variable and coordinates. Find nearest coordinates with similar depth.
#     INPUT: 
#     - reanalysis_ds: original xarray dataset 
#     - lon, lat,depth: target coordinates
#     - var: variable on which we want to avoid NaN
#     - radius=1
#     OUTPUT: 
#     - xarray dataset with only time as coordinates
#     """
#     # Select a layer in closer depth and first time step
#     depth_sel = reanalysis_ds.sel(depth=depth, method='nearest').depth.values
#     ds_depth = reanalysis_ds.sel(depth=depth_sel)
#     ds=ds_depth.isel(time=30) # first time at the given depth
    
#     # Select a subset of points within a certain radius
#     subset = ds.sel(lon=slice(lon-radius, lon+radius), lat=slice(lat-radius, lat+radius))
#     # Calculate the Euclidean distance to the points in the subset
#     dist = np.sqrt((subset.lon - lon)**2 + (subset.lat - lat)**2)
#     # Create a new dataset that only includes valid values
#     valid_subset = subset.where(~np.isnan(subset[var]), drop=True)
#     # Find the coordinates of the valid point with the smallest distance
#     min_dist_coords = dist.where(dist == dist.where(~np.isnan(valid_subset[var])).min(), drop=True)
#     nearest_lon = min_dist_coords.lon.values.item()
#     nearest_lat = min_dist_coords.lat.values.item()
#     return reanalysis_ds.sel(lat=nearest_lat, lon= nearest_lon, depth= depth_sel)

# def compute_u_v_speed_direction(ds):
#     """
#     Compute component that are missing in the dataset, we want a dataset with variables 'u', 'v', 'speed' and 'direction'
#     INPUT: 
#     - ds: xarray dataset
#     OUTPUT: 
#     - xds: xarray with all variables
#     """
#     if 'u' in ds.data_vars: 
#         # Compute missing variables
#         direction, speed = direction_speed_from_u_v(ds['u'].values, ds['v'].values)
#         # Add variables
#         ds['speed'] = xr.DataArray(
#             speed,
#             dims=ds['u'].dims,
#             coords=ds['u'].coords,
#             attrs={'units': 'm/s', 'long_name': 'Current speed'})
#         ds['direction'] = xr.DataArray(
#             direction,
#             dims=ds['u'].dims,
#             coords=ds['u'].coords,
#             attrs={'units': '°', 'long_name': 'current direction'})
#     else: 
#         # Compute missing variables
#         u, v = u_v_from_direction_speed(ds['direction'].values, ds['speed'].values)
#         # Add variables
#         ds['u'] = xr.DataArray(
#             u,
#             dims=ds['speed'].dims,
#             coords=ds['speed'].coords,
#             attrs={'units': 'm/s', 'long_name': 'u component current'})
#         ds['v'] = xr.DataArray(
#             v,
#             dims=ds['speed'].dims,
#             coords=ds['speed'].coords,
#             attrs={'units': 'm/s', 'long_name': 'v component current'})
#     """
#     print(list(ds.data_vars))
#     print('U:', ds['u'].values[:5])
#     print('V:', ds['v'].values[:5])
#     print('Speed:', ds['speed'].values[:5])
#     print('Direction:', ds['direction'].values[:5])
#     """
#     return ds

# # MFW function

# def moving_average(T, w):
#     """
#     Calculate the moving average of a time series T with window size w. 
    
#     INPUT:
#     - T (array-like): Time series data
#     - w (int): Size of the moving window
#     OUTPUT:
#     - moving_avg (numpy array): The moving average of the time series
#     """
#     # Cumulative sum of T
#     moving_avg = np.nancumsum(T)
#     # Subtract the previous cumulative sum to get the windowed moving average
#     moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
#     # Divide by the window size to get the average
#     moving_avg = moving_avg[w - 1:] / w
    
#     return moving_avg    


# def display_MFW_results(dico_window):
#     """
#     Display results from MFW results in a plot
#     INPUT: 
#         - Dico that output from function 'multi_window_filter'
#     OUTPUT: 
#         - PLot
#     """
#     plt.figure(figsize=(8,6))
#     plt.plot(dico_window['windows'], dico_window['distances'], label='Moving distance')
#     index_minima=dico_window['selected_windows']
#     dist_minima=[dico_window['distances'][i-1] for i in index_minima]
#     index_opti=np.min(dico_window['selected_windows'])
#     plt.scatter(index_minima, dist_minima,color='black', marker='o', label='3 smallest minima')
#     plt.scatter(index_opti, dico_window['distances'][index_opti-1],color='red', marker='o', label='Optimal window')
    
#     plt.xlabel('Window size')
#     plt.ylabel('Moving distance')
#     plt.legend()
#     plt.title('Moving average vs window sizes')

# def smooth_dataset(ds_og,s,e): 
#     """ 
#     Apply a rolling filter to 'u', 'v' ad 'speed' of the dataset. The size is chosen from the MWF algo. 
#     INPUT: 
#     - ds: xarray dataset
#     - s: minimium winwow size
#     - e: maximum window size
#     OUTPUT: 
#     - ds: dataset with new variables 'u_smooth', 'v_smooth', 'speed_smooth'
#     - size_window:
#     """
#     ds=ds_og.copy() # not to overwrite in the original dataset
#     dico_window = multi_window_filter(ds['u'].values, s, e)
#     #print(dico_window)
#     #display_MFW_results(dico_window)
#     """
#     plt.figure()
#     plt.plot(dico_window['windows'], dico_window['distances'], 'k*-')
#     plt.title(f'Selection optimal window: {dico_window['selected_windows']} ')
#     plt.show()
#     """
#     if dico_window['selected_windows']: 
#         size_window=np.min(dico_window['selected_windows'])
#         for variable in ['u','v','speed']:
#             ds[variable+'_smooth'] = xr.DataArray(
#                 ds[variable].rolling(time=size_window, center=True).mean(skipna=True),
#                 dims=ds['speed'].dims,
#                 coords=ds['speed'].coords,
#                 attrs={'units': f'same as {variable}', 'long_name': f'{variable} smoothed'})
#     else: 
#         size_window = 0
#         for variable in ['u','v','speed']:
#             ds[variable+'_smooth'] = xr.DataArray(
#                 ds[variable].copy(),
#                 dims=ds['speed'].dims,
#                 coords=ds['speed'].coords,
#                 attrs={'units': f'same as {variable}', 'long_name': f'{variable} smoothed (!!!! not here because no optimal window size)'})
        
#     return ds, size_window

# def show_smoothing(ds_smooth, size_window, type_ds=None):
#     '''
#     Present smoothing result
#     INPUT:
#     - ds_smooth: xarray dataset that have been smoothed
#     - size_window: size of the window used for smoothing
#     - type_ds=None: type of dataset for the title
#     OUPTPUT: 
#     - plot
#     '''
#     fig, ax = plt.subplots(2,2,figsize=(12, 6))
#     fig.suptitle(f"Results after smoothing {type_ds} dataset, window size={size_window}")
#     ax[0,0].plot(ds_smooth.time, ds_smooth['u'].values,color='cornflowerblue', marker='*', linestyle='-',label='u',alpha=0.5)
#     ax[0,0].plot(ds_smooth.time, ds_smooth['u_smooth'].values,color='blue', marker='', linestyle='-',label='u smoothed',alpha=1)
#     ax[0,0].legend()
#     ax[0,0].set_ylim([-0.2,0.3])
    
#     ax[0,1].plot(ds_smooth.time, ds_smooth['v'].values,color='springgreen', marker='*', linestyle='-',label='v',alpha=0.5)
#     ax[0,1].plot(ds_smooth.time, ds_smooth['v_smooth'].values,color='darkgreen', marker='', linestyle='-',label='v smoothed',alpha=1)
#     ax[0,1].legend()
#     ax[0,1].set_ylim([-0.2,0.3])
    
#     ax[1,0].plot(ds_smooth.time, ds_smooth['speed'].values,color='magenta', marker='*', linestyle='-',label='speed',alpha=0.5)
#     ax[1,0].plot(ds_smooth.time, ds_smooth['speed_smooth'].values,color='purple', marker='', linestyle='-',label='speed smoothed',alpha=1)
#     ax[1,0].legend()
#     ax[1,0].set_ylim([-0.1,0.4])
    
#     ax[1,1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Mult Window Filter (MWF)

# def multi_window_filter(T, s, e):
#     """
#     Compute the multi window filter on a signal T. Give the 3 minimal optimal window size
#     INPUT:
#     - T (array-like): Time series data
#     - s: minimum window size
#     - e: maximum window size
#     OUTPUT:
#     - dictionnary with diffrent windows computed, corresponding distances and the three optimal selected windows. For th eoptimal 
#         one we can choose the smallest one between the three
#     """
#     windows= []
#     moving_dist= []
#     for w in range(s,e,1):
#         avg= moving_average(T, w)
#         dist= np.sum(np.log(np.abs(avg-np.mean(avg))))
#         moving_dist.append(dist)
#         windows.append(w)
#     ### Index of local minima
#     # avoid nan
#     dist_array = np.array(moving_dist)
#     valid_idx = ~np.isnan(dist_array)
#     cleaned_dist = dist_array[valid_idx]    
#     # Find local minima in cleaned distances
#     minima_clean_idx = argrelextrema(cleaned_dist, np.less)[0]
    
#     # Re-map indices to original window list
#     minima_idx = np.where(valid_idx)[0][minima_clean_idx]
#     #minima_idx = argrelextrema(np.array(moving_dist), np.less)[0]
#     # get 3 smallest local minima
#     best_windows = [windows[i] for i in minima_idx]
#     best_dists = [moving_dist[i] for i in minima_idx]
#     sorted_best = sorted(zip(best_windows,best_dists ))[:3]
#     selected_windows = [w for w,_ in sorted_best]
#     return {
#         "windows": windows,
#         "distances": moving_dist,
#         "selected_windows": selected_windows,
#     }

# def recover_all_datasets(lander_name, landers_data, HF_radar_ds, HF_radar_ds_daily, HF_radar_ds_reconstructed, alti_ds, buoys_data, reanalysis_ds):
#     """
#     Recover all nearest datasets for one given Lander. You recover raw datasets and smoothed ones 

#     INPUT:
#     - lander_name: name of the lander (ex: 'LRL2')
#     OUTPUT: 
#     - datasets: all type of datasets for the correct location (according to each criteria. ex:: Reanalysis criteria different from HFR crit.)
#     - datasets_smooth: all type of smoothed datasets
#     """
#     # Select Lander dataset
#     lander_ds=landers_data[lander_name]['dataset']
#     lander_lat, lander_lon, lander_depth =landers_data[lander_name]['info']
#     lander_ds = lander_ds.assign_coords(lat=lander_lat, lon=lander_lon, depth=lander_depth)

#     # Select closest point 
#     HF_loc = HF_radar_ds.sel(lat=lander_lat, lon=lander_lon, method="nearest")
#     HF_daily_loc = HF_radar_ds_daily.sel(lat=lander_lat, lon=lander_lon, method="nearest")
#     HF_reconstruction_loc = HF_radar_ds_reconstructed.sel(lat=lander_lat, lon=lander_lon, method="nearest")
#     alti_loc = alti_ds.sel(lat=lander_lat ,lon=lander_lon, method='nearest')
#     name_buoy_nearest, buoy_distance, buoy_loc, buoy_info = closest_buoy(lander_lat, lander_lon, buoys_data)
#     reanalysis_loc=find_nearest_reanalysis(reanalysis_ds, lander_lon, lander_lat, lander_depth, 'u' )

#     # Compute u, v, speed & direction for all datasets
#     HF_loc=compute_u_v_speed_direction(HF_loc)
#     HF_daily_loc=compute_u_v_speed_direction(HF_daily_loc)
#     alti_loc=compute_u_v_speed_direction(alti_loc)
#     buoy_loc=compute_u_v_speed_direction(buoy_loc)
#     reanalysis_loc=compute_u_v_speed_direction(reanalysis_loc)

#     # Daily mean for buoy and lander datasets
#     buoy_daily_loc = buoy_loc.resample(time='1D').mean()
#     lander_daily_ds = lander_ds.resample(time='1D').mean()

#     # Dico with our datasets
#     datasets_lander={ 'Buoy': buoy_daily_loc,'Reanalysis': reanalysis_loc, 'Altimetry': alti_loc, 'HF reconstructed': HF_reconstruction_loc, 'HF': HF_daily_loc, 'Lander': lander_daily_ds}

#     # Apply Smoother to our datasets
#     datasets_smooth_lander={}
    
#     for type_ds, ds in datasets_lander.items():
#         #print(type_ds)
#         #print(ds.time)
#         ds_smooth, size_window = smooth_dataset(ds, 1,20)
#         #print(f'Number of nan in the dataset {type_ds}: {np.count_nonzero(np.isnan(np.array(ds_smooth.u)))}')
#         #show_smoothing(ds_smooth, size_window, type_ds=type_ds)
#         datasets_smooth_lander[type_ds]=ds_smooth
    
#     return datasets_lander, datasets_smooth_lander