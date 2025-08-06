
import os 
import pandas as pd
import argparse
import xarray as xr
import numpy as np
import pickle
import geoviews as gv
import geoviews.feature as gf
import panel as pn
import pandas as pd
import holoviews as hv
import hvplot.xarray
from holoviews import opts  # Needed for .opts on Layouts
import cartopy.crs as ccrs  


import matplotlib.pyplot as plt
import Global_functions


#%%

path_data='/Users/pascalecoubard/Desktop/IEO/1_Data/'
path_event = os.path.join(path_data, '8_Events_Lander_allDatasets')
path_bathy=os.path.join(path_data, '6_Emodnet_bathymetry')
bathy_dataset = xr.open_dataset(os.path.join(path_bathy,'bathy_merged.nc')) 


    
path_pkl = os.path.join(os.path.dirname(__file__), 'Cluster_Datasets_Landers_V3.pkl')

with open(path_pkl, 'rb') as f:
    final_dico = pickle.load(f)

datasets_color={'HF': 'blue','HF reconstructed': 'lightseagreen', 'Altimetry': 'green','Reanalysis': 'black', 'Buoy': 'magenta', 'Lander': 'red'}

# Parameters 
delta_lat = 0.08
delta_lon = 0.08

#%% Fill data and data_associated to display on the map

# Fill data dico with our values --> to display points on the top panel
data = {
    'name': [],
    'lat': [],
    'lon': [],
    'Uc': [],
    'Ua': []}

data_associated = {}

for name_lander, related_dico in final_dico.items():
    # fill data
    ds = related_dico['Lander']['dataset']
    lat = float(ds.lat)
    lon = float(ds.lon)
    
    # data dico
    data['name'].append(name_lander)
    data['lat'].append(lat)
    data['lon'].append(lon)
    dico_Uc_Ua, _, _ = Global_functions.compute_Uc_Ua(bathy_dataset, lat, lon, delta_lat, delta_lon)
    data['Uc'].append(dico_Uc_Ua['Uc_ref'])
    data['Ua'].append(dico_Uc_Ua['Ua_ref'])

    # fill data_associated
    surrounding_points_df = pd.DataFrame(columns=['lon', 'lat', 'name'])
    for name_dataset, content in related_dico.items():
        ds = content.get('dataset', None)
        if ds is None:
            continue  
        if 'lon' in ds and 'lat' in ds:
            df_temp = pd.DataFrame({
                'lon': [ds['lon'].values],
                'lat': [ds['lat'].values],
                'name': [name_dataset ],
                'color': [datasets_color[name_dataset]]
            })
    
            surrounding_points_df = pd.concat([surrounding_points_df, df_temp], ignore_index=True)
    data_associated[name_lander]=surrounding_points_df

data= pd.DataFrame(data)


#%%  PARMETERS 
 
hv.extension('bokeh')
pn.extension()
width_panel=1200
height_top_panel= 180
height_second_panel=380
height_third_panel=50

#%% -------- First Line
# Lander points for the map
points = gv.Points(data, kdims=['lon', 'lat'], vdims=['name'], crs=ccrs.PlateCarree()).opts(
    size=10,
    color='crimson',
    tools=['tap'],
    active_tools=['tap'],
    height=height_top_panel,
    width=int(width_panel),
    marker='circle',
    
    title="Lander location (select point to have time series)")

# Add coast and ocean layers
coastline = gf.coastline.opts(line_color='black')
ocean = gf.ocean.opts(fill_color='lightblue')

def reactive_map(index):
    base_map = (ocean * coastline * points)

    if index is None or len(index) == 0:
        return base_map.opts(framewise=True)
    
    lander_name = data.iloc[index[0]]['name']
    surrounding_points_df = data_associated[lander_name]
    
    # Create gv.Points for surrounding points with different style
    surrounding_points = gv.Points(
        surrounding_points_df,
        kdims=['lon', 'lat'],
        vdims=['color'], 
        crs=ccrs.PlateCarree()
    ).opts(
        size=8,
        color='color',
        marker='circle',
        alpha=0.5,
        height=height_top_panel,
        width=int(width_panel),
        tools=[],
    )

    # Overlay base map + surrounding points
    return (base_map * surrounding_points).opts(framewise=True)


#%% ------------ Second Line

# Function to create a time series plot of variable 'uc' and 'ua'
def plot_uc_ua(index):
    u_max=0.5

    if index is None or len(index) == 0:
        return hv.Text(0.5, 0.5, 'No Lander selected').opts(height=height_second_panel, width=width_panel, fontsize=14,)
    
    lander_name = data.iloc[index[0]]['name']
    related_dico = final_dico[lander_name]
    depth_lander = related_dico['Lander']['dataset']['depth'].values
    curves_ua = []  # Ua = along-slope
    curves_uc = []  # Uc = cross-slope
    
    # Loop through datasets
    for name_dataset, raw_smooth_dico in related_dico.items():
        if name_dataset == 'HF':  # Skip HF
            continue 
        
        ds = raw_smooth_dico['dataset']
        if 'ua' in ds and 'uc' in ds:
            color = datasets_color.get(name_dataset, 'gray')

            curve_ua = ds['ua'].hvplot.line(
                x='time', y='ua', label=name_dataset,
                color=color, alpha=0.7, ylim=(-u_max, u_max),ylabel='Ua in m/s (along-slope)', width=width_panel, height=int(height_second_panel/2))
            curves_ua.append(curve_ua)

            curve_uc = ds['uc'].hvplot.line(
                x='time', y='uc', label=name_dataset,
                color=color, alpha=0.7, ylim=(-u_max, u_max),ylabel=' Uc in m/s (cross-slope)', width=width_panel, height=int(height_second_panel/2))
            curves_uc.append(curve_uc)

    subplot_ua = hv.Overlay(curves_ua).opts(show_legend=False, title="")
    subplot_uc = hv.Overlay(curves_uc).opts(show_legend=False, title="")

    # Combine the two plots vertically
    full_plot = (subplot_ua + subplot_uc).cols(1).opts(
        opts.Layout(title=f'Lander: {lander_name}, {depth_lander}m', merge_tools=True))

    return full_plot


#%% Third panel

# --- LEGEND (2/3 of the width) ---
"""
legend_html = '''
<div style="font-size:14px; padding:10px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
'''

for name, color in datasets_color.items():
    legend_html += f'''
    <div style="display: flex; align-items: center;">
        <div style="width: 10px; height: 10px; background-color: {color}; border-radius: 50%; margin-right: 6px;"></div>
        <span>{name}</span>
    </div>
    '''

legend_html += '</div>'


legend_panel = pn.pane.HTML(legend_html, width=int(width_panel / 2 ), height=height_third_panel)
"""

def interactive_legend(index):
    legend_html = '''
    <div style="font-size:14px; padding:10px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
    '''

    lander_name = data.iloc[index[0]]['name'] if index else None
    depth_reanalysis = None
    depth_real = None

    if lander_name and 'Reanalysis' in final_dico[lander_name]:
        ds = final_dico[lander_name]['Reanalysis']['dataset']
        if 'depth' in ds:
            try:
                depth_reanalysis = ds['depth'].values.item()
            except:
                depth_reanalysis = 'N/A'
        if 'depth_real' in ds:
            try:
                depth_real = ds['depth_real'].values.item()
            except:
                depth_real = 'N/A'

    for name, color in datasets_color.items():
        legend_html += f'''
        <div style="display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; background-color: {color}; border-radius: 50%; margin-right: 6px;"></div>
                <span>{name}{" ({:.1f} m)".format(depth_reanalysis) if name == "Reanalysis" and isinstance(depth_reanalysis, (int, float)) else ""}</span>
            </div>
        '''

        if name == 'Reanalysis' and depth_real is not None:
            legend_html += f'''
            <div style="margin-left: 16px; font-size:12px; color: gray;">
                Bathymetry at this loc: {depth_real:.1f}m
            </div>
            '''

        legend_html += '</div>'

    legend_html += '</div>'
    return pn.pane.HTML(legend_html, width=int(width_panel / 2), height=height_third_panel)



# --- TEXT PANEL ---
text_panel = pn.pane.Markdown(
    '**<span style="color:orange;">Uc</span> and <span style="color:red;">Ua</span> at Lander loc.:**',
    width=int(width_panel / 8),
    height=height_third_panel)

# --- MATPLOTLIB PLOT ---

def plot_arrows(index):
    if index is None or len(index) == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('off')
        ax.set_facecolor('white')
        fig.tight_layout(pad=0)
        return  pn.pane.Matplotlib(fig, dpi=100, height=height_third_panel, width=int(width_panel *3/ 8))
    
    Uc= data.iloc[index[0]]['Uc']
    Ua= data.iloc[index[0]]['Ua']
    
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.axis('off')
    ax.set_facecolor('white')
    # Base point
    ax.quiver(1, 1, Uc[0], Uc[1], angles='xy', scale_units='xy', scale=1 ,color='orange', width=0.05)
    ax.quiver(1, 1, Ua[0], Ua[1], angles='xy', scale_units='xy', scale=1 ,color='red', width=0.05)
    ax.set_xlim([0,2])
    ax.set_ylim([0,2])
    fig.tight_layout(pad=0)
    
    arrow_panel = pn.pane.Matplotlib(fig, dpi=100, height=height_third_panel, width=int(width_panel *3/ 8))
    return arrow_panel


# %%------- Interactive part

# Define tap stream 
tap_stream = hv.streams.Selection1D(source=points)
# Bind reactive map to tap stream index
interactive_map = pn.bind(reactive_map, tap_stream.param.index)
# Bind the plotting function to the click stream
time_series_panel = pn.bind(plot_uc_ua, tap_stream.param.index)
# Bind the legend function to the click stream
legend_panel = pn.bind(interactive_legend, tap_stream.param.index)
# Bind the arrow function to the click stream
arrow_panel =  pn.bind(plot_arrows, tap_stream.param.index)

#%% 
# --- THIRD PANEL ROW ---
third_panel = pn.Row(
    legend_panel,
    text_panel,
    arrow_panel
)
#%% ------- FINAL LAYOUT -------
layout = pn.Column(interactive_map, time_series_panel, third_panel)
layout.servable()


