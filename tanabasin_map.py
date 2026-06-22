import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import Point, mapping
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
workspace      = r'C:\Users\Rono\Desktop\Tana River Floods'
basins_path    = os.path.join(workspace, 'ke_water_basins', 'ke_water_basins.shp')
hillshade_path = os.path.join(workspace, 'hillshd_90m', 'hillshd_90m', 'hillshd_90m', 'w001001.adf')
kenya_path     = os.path.join(workspace, 'gadm41_KEN_shp', 'gadm41_KEN_0.shp')
rivers_path    = r'C:\Users\Rono\Desktop\hotosm_ken_waterways_lines_shp\hotosm_ken_waterways_lines_shp.shp'

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
C = dict(
    canvas      = '#FFFFFF',       
    map_bg      = '#F4F7F9',       
    main_river  = '#115599',       
    trib        = '#4488CC',       
    stream      = '#A0C0E0',       
    dam_marker  = '#E6A817',       
    dam_edge    = '#604000',
    station     = '#D32F2F',       
    station_edge= '#FFFFFF',
    basin_edge  = '#222222',       
    grid        = '#777777',       
    text_label  = '#0A2240',       
    text_title  = '#0A2240',
    text_sub    = '#555555',
    ocean_ins   = '#B0D0E8',       
    kenya_ins   = '#EAE3D2',       
)

sat_colors = [
    (0.00, '#D4EDDA'), (0.15, '#C3E6CB'), (0.35, '#E6DFBC'), 
    (0.55, '#D1A168'), (0.75, '#A67343'), (0.90, '#805028'), (1.00, '#F5F5F5')
]
sat_cmap = mcolors.LinearSegmentedColormap.from_list('light_elev', [(v, c) for v, c in sat_colors])


# LOAD DATA

gdf_basins = gpd.read_file(basins_path)
tana_basin = gdf_basins[gdf_basins['KE_BASIN'] == 4]
gdf_kenya  = gpd.read_file(kenya_path)
gdf_rivers = gpd.read_file(rivers_path)

if gdf_rivers.crs != tana_basin.crs:
    gdf_rivers = gdf_rivers.to_crs(tana_basin.crs)

gdf_rivers_tana = gpd.clip(gdf_rivers, tana_basin)

if 'waterway' in gdf_rivers_tana.columns:
    main_rivers = gdf_rivers_tana[gdf_rivers_tana['waterway'] == 'river']
    streams     = gdf_rivers_tana[gdf_rivers_tana['waterway'].isin(['stream', 'canal'])]
    tribs       = gdf_rivers_tana[~gdf_rivers_tana['waterway'].isin(['river', 'stream', 'canal'])]
else:
    main_rivers = gdf_rivers_tana
    streams     = gpd.GeoDataFrame()
    tribs       = gpd.GeoDataFrame()


# 2. READ & MASK HILLSHADE

os.environ['GDAL_CACHEMAX'] = '256'
basin_geom = [mapping(geom) for geom in tana_basin.geometry]

with rasterio.open(hillshade_path) as src:
    if tana_basin.crs != src.crs:
        tana_reproj = tana_basin.to_crs(src.crs)
        basin_geom_reproj = [mapping(geom) for geom in tana_reproj.geometry]
    else:
        basin_geom_reproj = basin_geom

    fill_val = 0
    scale_factor = 2
    hs_data_full = src.read(1, out_shape=(src.height // scale_factor, src.width // scale_factor))
    hs_transform_scaled = src.transform * rasterio.transform.Affine.scale(scale_factor, scale_factor)
    
    from rasterio.io import MemoryFile
    with MemoryFile() as memfile:
        with memfile.open(driver='GTiff', height=hs_data_full.shape[0], width=hs_data_full.shape[1],
                          count=1, dtype=hs_data_full.dtype, crs=src.crs, transform=hs_transform_scaled, nodata=fill_val) as mem:
            mem.write(hs_data_full, 1)
            hs_data, hs_transform = rasterio_mask(mem, basin_geom_reproj, crop=True, nodata=fill_val)
    
    hs_arr = hs_data[0].astype(np.float32)
    nodata_val = src.nodata if src.nodata is not None else fill_val
    hs_arr[hs_arr == nodata_val] = np.nan

    hs_min, hs_max = np.nanmin(hs_arr), np.nanmax(hs_arr)
    hs_norm = (hs_arr - hs_min) / (hs_max - hs_min)

    h_h, h_w = hs_arr.shape
    xs, ys = [], []
    for r in [0, h_h]:
        for c in [0, h_w]:
            x, y = hs_transform * (c, r)
            xs.append(x); ys.append(y)
    extent = [min(xs), max(xs), min(ys), max(ys)]


# 3. INFRASTRUCTURE POINTS

seven_forks_dams = {
    'Masinga':   [37.59, -0.89], 'Kamburu':   [37.67, -0.83],
    'Gitaru':    [37.75, -0.80], 'Kindaruma': [37.81, -0.81],
    'Kiambere':  [37.91, -0.64],
}
gdf_dams = gpd.GeoDataFrame([{'name': k, 'geometry': Point(v[0], v[1])} for k, v in seven_forks_dams.items()], crs="EPSG:4326")

stations = {
    'Garissa (4GQ01)':    [39.64, -0.45],
    'Bura (4G03)':         [39.89, -1.19],
    'Galole/Hola (4G04)': [40.03, -1.50],
    'Garsen (4G02)':      [40.12, -2.27],
}
gdf_stations = gpd.GeoDataFrame([{'name': k, 'geometry': Point(v[0], v[1])} for k, v in stations.items()], crs="EPSG:4326")

# 4. FIGURE SETUP

fig = plt.figure(figsize=(14, 14), facecolor=C['canvas'])
ax  = fig.add_axes([0.06, 0.08, 0.90, 0.82]) 
ax.set_facecolor(C['map_bg'])

bounds = tana_basin.total_bounds
PAD_X = 0.25
PAD_Y = 0.40  
ax.set_xlim(bounds[0] - PAD_X, bounds[2] + PAD_X)
ax.set_ylim(bounds[1] - PAD_Y, bounds[3] + PAD_X)


# 5. GRATICULES

ax.set_xlabel('Longitude (°E)', fontsize=8.5, color='#333333', labelpad=5)
ax.set_ylabel('Latitude (°)', fontsize=8.5, color='#333333', labelpad=5)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.grid(True, linestyle=':', linewidth=0.5, color=C['grid'], alpha=0.35, zorder=0)
ax.tick_params(labelsize=8, colors='#333333')
for spine in ax.spines.values():
    spine.set_edgecolor('#666666')
    spine.set_linewidth(1.0)

# RENDER LAYER STACK

ax.imshow(sat_cmap(hs_norm), extent=extent, origin='upper', aspect='auto', zorder=1, alpha=0.85)
ax.imshow(hs_norm, extent=extent, origin='upper', cmap='gray', aspect='auto', zorder=2, alpha=0.25, vmin=0, vmax=1)

tana_basin.plot(ax=ax, facecolor='none', edgecolor=C['basin_edge'], linewidth=1.4, linestyle='--', zorder=3, alpha=0.8)

if not streams.empty:
    streams.plot(ax=ax, color=C['stream'], linewidth=0.4, alpha=0.6, zorder=4)
if not tribs.empty:
    tribs.plot(ax=ax, color=C['trib'], linewidth=0.8, alpha=0.7, zorder=5)
main_rivers.plot(ax=ax, color=C['main_river'], linewidth=2.2, alpha=1.0, zorder=6)

gdf_dams.plot(ax=ax, color=C['dam_marker'], marker='D', markersize=80, edgecolor=C['dam_edge'], linewidth=1.0, zorder=8)
gdf_stations.plot(ax=ax, color=C['station'], marker='o', markersize=55, edgecolor=C['station_edge'], linewidth=1.0, zorder=9)

# TYPOGRAPHY LABELS

halo = [pe.withStroke(linewidth=3.0, foreground='#FFFFFF')]
label_offsets = {
    'Garissa (4GQ01)':    (0.16, 0.06),
    'Bura (4G03)':        (-0.16, 0.06),   
    'Galole/Hola (4G04)': (0.16, 0.05),
    'Garsen (4G02)':      (-0.16, -0.04),  
}

for _, row in gdf_stations.iterrows():
    dx, dy = label_offsets.get(row['name'], (0.15, 0.05))
    align_side = 'left' if dx > 0 else 'right'
    ax.annotate(
        row['name'],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(row.geometry.x + dx, row.geometry.y + dy),
        fontsize=8, fontweight='bold', color=C['text_label'],
        path_effects=halo,
        arrowprops=dict(arrowstyle='-', color='#555555', lw=0.7),
        va='center', ha=align_side, zorder=20
    )
 #TITLE & OVERLAYS
ax.text(0.50, 1.070, 'TANA RIVER BASIN', transform=ax.transAxes,
        fontsize=19, fontweight='bold', color=C['text_title'], va='top', ha='center', fontfamily='serif')
ax.text(0.50, 1.035, '', transform=ax.transAxes,
        fontsize=10, color=C['text_sub'], va='top', ha='center', style='italic')

# Reliable Vector Character North Arrow
ax.text(0.96, 0.945, '↑\nN', transform=ax.transAxes, ha='center', va='center', 
        fontsize=14, fontweight='bold', color='#222222', zorder=25,
        path_effects=[pe.withStroke(linewidth=2.0, foreground='#FFFFFF')])

# Linear Scale Bar
seg = 0.5 
sb_x = bounds[2] - 1.4
sb_y = bounds[1] - 0.22
for i, fc in enumerate(['#222222', '#FFFFFF', '#222222']):
    ax.barh(sb_y, seg, left=sb_x + i * seg, height=0.035, color=fc, edgecolor='#222222', linewidth=0.8, zorder=10)
for val, lab in [(sb_x, '0'), (sb_x + seg, '55'), (sb_x + 2*seg, '110 km')]:
    ax.text(val, sb_y - 0.07, lab, ha='center', fontsize=7.5, color='#222222', zorder=10, path_effects=halo)


# ADJUSTED SIDEBAR: SHIFTED LEGEND BOX UP

# legend_y bumped from 0.17 up to 0.35 to use the open space completely
legend_x, legend_y, legend_w, legend_h = 0.01, 0.30, 0.25, 0.18
legend_bg = FancyBboxPatch((legend_x, legend_y), legend_w, legend_h, boxstyle='square,pad=0.005',
                            facecolor='#FFFFFF', edgecolor='#BBBBBB', linewidth=0.8, alpha=0.95, transform=ax.transAxes, zorder=15)
ax.add_patch(legend_bg)

def lp(x0, x1, y, color, lw, ls='-'):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, transform=ax.transAxes, zorder=16)
def lm(x, y, marker, color, edge, ms=6):
    ax.plot(x, y, marker=marker, color=color, markersize=ms, markeredgecolor=edge, markeredgewidth=0.8, transform=ax.transAxes, zorder=16)
def lt(x, y, text, color='#222222', fs=7.2, fw='normal'):
    ax.text(x, y, text, color=color, fontsize=fs, fontweight=fw, va='center', transform=ax.transAxes, zorder=16)

lt(legend_x + legend_w/2, legend_y + legend_h - 0.02, 'LEGEND', color=C['text_title'], fw='bold', fs=8)
lp(legend_x + 0.01, legend_x + legend_w - 0.01, legend_y + legend_h - 0.038, '#CCCCCC', 0.6)

col_sym, col_sym2, col_txt = legend_x + 0.015, legend_x + 0.065, legend_x + 0.075
rows = [
    (0.11, 'line', (C['main_river'], 2.0, '-',  'Main Tana River')),
    (0.07, 'mark', (C['dam_marker'], 'D', C['dam_edge'], 6.0, 'Seven Forks Dams')),
    (0.04, 'mark', (C['station'],    'o', C['station_edge'], 5.0, 'River Gauging Station')),
    (0.01, 'line', (C['basin_edge'], 1.2, '--', 'Basin Catchment Boundary'))
]

for ry, rtype, rargs in rows:
    ay = legend_y + ry
    if rtype == 'line':
        color, lw, ls, label = rargs
        lp(col_sym, col_sym2, ay, color, lw, ls)
        lt(col_txt, ay, label)
    elif rtype == 'mark':
        color, marker, edge, ms, label = rargs
        lm((col_sym + col_sym2)/2, ay, marker, color, edge, ms)
        lt(col_txt, ay, label)

#
# bbox_to_anchor coordinates 
ax_ins = inset_axes(ax, width='18%', height='18%', loc='lower left', 
                    bbox_to_anchor=(0.01, 0.10, 1, 1), bbox_transform=ax.transAxes)
ax_ins.set_facecolor(C['ocean_ins'])
gdf_kenya.plot(ax=ax_ins, color=C['kenya_ins'], edgecolor='#999999', lw=0.5)
tana_basin.plot(ax=ax_ins, color='#72B174', edgecolor='#222222', lw=0.8)
ax_ins.set_xticks([]); ax_ins.set_yticks([])


# Metadata footer
ax.text(0.22, 0.002, 'Coord System: WGS 84 (EPSG:4326) · Sources: HOTOSM, GADM, SRTM 90m', transform=ax.transAxes, fontsize=6.5, color='#777777', va='bottom')

# Output Save execution
out_path = os.path.join(workspace, 'Tana_Basin.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=C['canvas'])
plt.show()
print(f"Layout engine updated successfully → {out_path}")