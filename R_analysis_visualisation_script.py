import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURATION 
EXCEL_FILE = 'TanaRiver Flow.xlsx'
SHEET_NAME = 'Rawdata'
WINDOW_SIZE = 14

THRESHOLDS = {
    'Bura': 721,
    'Galole': 723,
    'Garsen': 1723
}

STATIONS = ['Bura', 'Galole', 'Garsen']
OUTPUT_DIR = './flood_visualizations/'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

#  UTILITY FUNCTIONS 
def year_aware_rolling_max(series, dates, year_col, window=14):
    result = []
    series_vals = series.values
    year_vals = year_col.values
    for i in range(len(series_vals)):
        current_year = year_vals[i]
        year_indices = np.where(year_vals == current_year)[0]
        year_start_idx = year_indices[0] if len(year_indices) > 0 else 0
        lookback_idx = max(0, i - window + 1)
        start_idx = max(year_start_idx, lookback_idx)
        window_data = series_vals[start_idx:i+1]
        if len(window_data) > 0:
            max_val = np.max(window_data)
            if not np.isnan(max_val):
                result.append(int(max_val))
            else:
                result.append(np.nan)
        else:
            result.append(np.nan)
    return result

def identify_flood_events(rolling_max_series, threshold):
    event_numbers = []
    current_event = 0
    in_event = False
    for value in rolling_max_series:
        if pd.isna(value):
            event_numbers.append(0)
            continue
        if value > threshold:
            if not in_event:
                current_event += 1
                in_event = True
            event_numbers.append(current_event)
        else:
            in_event = False
            event_numbers.append(0)
    return event_numbers

def mark_event_dates(event_numbers, dates):
    start_dates = [None] * len(event_numbers)
    end_dates = [None] * len(event_numbers)
    durations = [None] * len(event_numbers)
    current_event = None
    event_start_idx = None
    event_start_date = None
    for idx, event_num in enumerate(event_numbers):
        if event_num > 0:
            if event_num != current_event:
                current_event = event_num
                event_start_idx = idx
                event_start_date = dates.iloc[idx]
                start_dates[idx] = event_start_date
        else:
            if current_event is not None:
                end_date = dates.iloc[idx - 1]
                duration = (end_date - event_start_date).days + 1
                end_dates[idx - 1] = end_date
                durations[idx - 1] = duration
                current_event = None
    if current_event is not None:
        end_date = dates.iloc[len(dates) - 1]
        duration = (end_date - event_start_date).days + 1
        end_dates[len(dates) - 1] = end_date
        durations[len(dates) - 1] = duration
    return start_dates, end_dates, durations

def calculate_event_block_max(event_numbers, rolling_values):
    event_block_max = [None] * len(event_numbers)
    event_to_max = {}
    for idx, event_num in enumerate(event_numbers):
        if event_num > 0 and not pd.isna(rolling_values[idx]):
            if event_num not in event_to_max:
                event_to_max[event_num] = rolling_values[idx]
            else:
                event_to_max[event_num] = max(event_to_max[event_num], rolling_values[idx])
    for idx, event_num in enumerate(event_numbers):
        if event_num > 0:
            is_event_end = (idx == len(event_numbers) - 1) or (event_numbers[idx + 1] != event_num)
            if is_event_end:
                event_block_max[idx] = event_to_max.get(event_num, None)
    return event_block_max

#  LOAD & PROCESS DATA 
print("=" * 80)
print("FLOOD EVENT VISUALIZATIONS")
print("=" * 80)
print("\\n[STEP 1] Loading data...")

raw_data = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=1, usecols=[0, 1, 2, 3])
raw_data.columns = ['date', 'Bura', 'Galole', 'Garsen']
raw_data['date'] = pd.to_datetime(raw_data['date'], errors='coerce')

for col in STATIONS:
    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

raw_data['year'] = raw_data['date'].dt.year
raw_data['month'] = raw_data['date'].dt.month
raw_data = raw_data.dropna(subset=['date']).reset_index(drop=True)

print(f"✓ Data loaded: {raw_data['date'].min()} to {raw_data['date'].max()}")

#  PROCESS STATIONS 
results = {}

print("\\n[STEP 2] Processing stations...")
for station in STATIONS:
    threshold = THRESHOLDS[station]
    rolling_max = year_aware_rolling_max(raw_data[station], raw_data['date'], raw_data['year'], WINDOW_SIZE)
    events = identify_flood_events(rolling_max, threshold)
    start_dates, end_dates, durations = mark_event_dates(events, raw_data['date'])
    block_max = calculate_event_block_max(events, rolling_max)
    
    event_data_list = []
    for idx, event_num in enumerate(events):
        if event_num > 0 and durations[idx] is not None:
            event_data_list.append({
                'event_num': event_num,
                'year': raw_data['year'].iloc[idx],
                'duration': durations[idx],
                'peak': block_max[idx]
            })
    
    event_df = pd.DataFrame(event_data_list).drop_duplicates(subset=['event_num'])
    
    results[station] = {
        'rolling_max': rolling_max,
        'events': events,
        'start_dates': start_dates,
        'end_dates': end_dates,
        'durations': durations,
        'block_max': block_max,
        'threshold': threshold,
        'event_df': event_df
    }
    print(f"✓ {station}: {len(event_df)} events")

#  VISUALIZATION 1: EVENT COUNTS BY STATION 
print("\\n[STEP 3] Creating visualizations...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('1. Event Frequency Comparison', fontsize=14, fontweight='bold', y=1.02)

for idx, station in enumerate(STATIONS):
    event_df = results[station]['event_df']
    events_per_year = event_df.groupby('year').size()
    
    ax = axes[idx]
    ax.bar(events_per_year.index, events_per_year.values, color='steelblue', alpha=0.7, edgecolor='navy')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Number of Events', fontweight='bold')
    ax.set_title(f'{station.upper()} Station\\nTotal Events: {len(event_df)}', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    z = np.polyfit(events_per_year.index, events_per_year.values, 1)
    p = np.poly1d(z)
    ax.plot(events_per_year.index, p(events_per_year.index), "r--", linewidth=2, label='Trend')
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}01_Event_Frequency_By_Year.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_Event_Frequency_By_Year.png")
plt.close()

#  VISUALIZATION 2: DURATION DISTRIBUTION 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('2. Event Duration Distribution', fontsize=14, fontweight='bold', y=1.02)

for idx, station in enumerate(STATIONS):
    event_df = results[station]['event_df']
    ax = axes[idx]
    
    ax.hist(event_df['duration'], bins=15, color='coral', alpha=0.7, edgecolor='darkred')
    ax.axvline(event_df['duration'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {event_df["duration"].mean():.1f}')
    ax.axvline(event_df['duration'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {event_df["duration"].median():.1f}')
    
    ax.set_xlabel('Duration (days)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{station.upper()} Station', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}02_Duration_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_Duration_Distribution.png")
plt.close()

#  VISUALIZATION 3: PEAK LEVEL DISTRIBUTION 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('3. Peak Level Distribution (with Threshold)', fontsize=14, fontweight='bold', y=1.02)

for idx, station in enumerate(STATIONS):
    event_df = results[station]['event_df']
    threshold = results[station]['threshold']
    
    ax = axes[idx]
    ax.hist(event_df['peak'], bins=15, color='skyblue', alpha=0.7, edgecolor='navy')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2.5, label=f'Threshold: {threshold}')
    ax.axvline(event_df['peak'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {event_df["peak"].mean():.0f}')
    
    ax.set_xlabel('Peak Level', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{station.upper()} Station', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_Peak_Level_Distribution.png', dpi=300, bbox_inches='tight')
print("Saved: 03_Peak_Level_Distribution.png")
plt.close()

#  VISUALIZATION 4: DECADE TRENDS 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('4. Decade-based Trends', fontsize=14, fontweight='bold', y=0.995)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

ax = axes[0, 0]
decade_data = {}
for station in STATIONS:
    event_df = results[station]['event_df']
    event_df['decade'] = (event_df['year'] // 10) * 10
    decade_data[station] = event_df.groupby('decade').size()

for idx, station in enumerate(STATIONS):
    if station in decade_data:
        ax.plot(decade_data[station].index, decade_data[station].values, marker='o', label=station, linewidth=2, color=colors[idx])

ax.set_xlabel('Decade', fontweight='bold')
ax.set_ylabel('Event Count', fontweight='bold')
ax.set_title('Events by Decade', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
for idx, station in enumerate(STATIONS):
    event_df = results[station]['event_df']
    event_df['decade'] = (event_df['year'] // 10) * 10
    decade_durations = event_df.groupby('decade')['duration'].mean()
    ax.plot(decade_durations.index, decade_durations.values, marker='s', label=station, linewidth=2, color=colors[idx])

ax.set_xlabel('Decade', fontweight='bold')
ax.set_ylabel('Avg Duration (days)', fontweight='bold')
ax.set_title('Average Event Duration by Decade', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
for idx, station in enumerate(STATIONS):
    event_df = results[station]['event_df']
    event_df['decade'] = (event_df['year'] // 10) * 10
    decade_peaks = event_df.groupby('decade')['peak'].mean()
    ax.plot(decade_peaks.index, decade_peaks.values, marker='^', label=station, linewidth=2, color=colors[idx])

ax.set_xlabel('Decade', fontweight='bold')
ax.set_ylabel('Avg Peak Level', fontweight='bold')
ax.set_title('Average Peak Level by Decade', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

summary_data = []
for station in STATIONS:
    event_df = results[station]['event_df']
    years_coverage = raw_data['year'].max() - raw_data['year'].min() + 1
    summary_data.append([
        station,
        len(event_df),
        f"{len(event_df) / years_coverage:.2f}",
        f"{event_df['duration'].mean():.1f}",
        f"{event_df['peak'].mean():.0f}"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Station', 'Total', 'Events/Yr', 'Avg Dur', 'Avg Peak'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.12, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(5):
    table[(0, i)].set_facecolor('#1F4E78')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 4):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax.set_title('Summary Statistics', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Decade_Trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_Decade_Trends.png")
plt.close()

# VISUALIZATION 5: SEASONALITY HEATMAP 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('5. Seasonality Heatmap (Events by Month & Year)', fontsize=14, fontweight='bold', y=1.02)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx, station in enumerate(STATIONS):
    events = results[station]['events']
    start_dates = results[station]['start_dates']
    
    years = sorted(raw_data['year'].unique())
    months = range(1, 13)
    heatmap_data = np.zeros((len(months), len(years)))
    
    for i, date in enumerate(start_dates):
        if date is not None:
            month_idx = date.month - 1
            year_idx = years.index(date.year)
            heatmap_data[month_idx, year_idx] += 1
    
    ax = axes[idx]
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45)
    ax.set_yticks(range(len(months)))
    ax.set_yticklabels(month_names)
    
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Month', fontweight='bold')
    ax.set_title(f'{station.upper()} Station', fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Event Count')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}05_Seasonality_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_Seasonality_Heatmap.png")
plt.close()

#  VISUALIZATION 6: MONTHLY EVENT COUNTS 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('6. Monthly Event Distribution', fontsize=14, fontweight='bold', y=1.02)

month_names_long = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                    7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

for idx, station in enumerate(STATIONS):
    events = results[station]['events']
    start_dates = results[station]['start_dates']
    
    months_with_events = []
    for i, date in enumerate(start_dates):
        if date is not None:
            months_with_events.append(date.month)
    
    month_counts = Counter(months_with_events)
    
    months_list = list(range(1, 13))
    counts_list = [month_counts.get(m, 0) for m in months_list]
    
    ax = axes[idx]
    bars = ax.bar(range(12), counts_list, color='mediumseagreen', alpha=0.7, edgecolor='darkgreen')
    
    max_count = max(counts_list) if max(counts_list) > 0 else 1
    for bar, count in zip(bars, counts_list):
        if count > 0:
            bar.set_color('darkgreen' if count >= max_count * 0.5 else 'mediumseagreen')
    
    ax.set_xticks(range(12))
    ax.set_xticklabels([month_names_long[i][:3] for i in range(1, 13)], rotation=45)
    ax.set_ylabel('Event Count', fontweight='bold')
    ax.set_title(f'{station.upper()} Station', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_Monthly_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_Monthly_Distribution.png")
plt.close()

#  VISUALIZATION 7: SCATTER - DURATION VS PEAK 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('7. Relationship: Event Duration vs Peak Level', fontsize=14, fontweight='bold', y=1.02)

for idx, station in enumerate(STATIONS):
    event_df = results[station]['event_df']
    threshold = results[station]['threshold']
    
    ax = axes[idx]
    scatter = ax.scatter(event_df['duration'], event_df['peak'], 
                        s=100, alpha=0.6, c=event_df['peak'], cmap='viridis', edgecolor='black', linewidth=0.5)
    
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    
    if len(event_df) > 1:
        corr = event_df['duration'].corr(event_df['peak'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Duration (days)', fontweight='bold')
    ax.set_ylabel('Peak Level', fontweight='bold')
    ax.set_title(f'{station.upper()} Station (n={len(event_df)})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Peak Level')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}07_Duration_vs_Peak.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_Duration_vs_Peak.png")
plt.close()

#  VISUALIZATION 8: SPATIAL COMPARISON 
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle('8. Spatial Comparison: All Stations', fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(gs[0, 0])
stations = [s for s in STATIONS]
event_counts = [len(results[s]['event_df']) for s in stations]
bars = ax1.bar(stations, event_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('Total Events', fontweight='bold')
ax1.set_title('Total Flood Events', fontweight='bold')
for bar, count in zip(bars, event_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
avg_peaks = [results[s]['event_df']['peak'].mean() for s in stations]
bars = ax2.bar(stations, avg_peaks, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Average Peak Level', fontweight='bold')
ax2.set_title('Average Peak Level', fontweight='bold')
for bar, peak in zip(bars, avg_peaks):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
avg_durs = [results[s]['event_df']['duration'].mean() for s in stations]
bars = ax3.bar(stations, avg_durs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
ax3.set_ylabel('Average Duration (days)', fontweight='bold')
ax3.set_title('Average Event Duration', fontweight='bold')
for bar, dur in zip(bars, avg_durs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
data_to_plot = [results[s]['event_df']['peak'].values for s in stations]
bp = ax4.boxplot(data_to_plot, labels=stations, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Peak Level', fontweight='bold')
ax4.set_title('Peak Level Distribution', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.savefig(f'{OUTPUT_DIR}08_Spatial_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_Spatial_Comparison.png")
plt.close()

#  VISUALIZATION 9: TEMPORAL TIME SERIES 
fig, axes = plt.subplots(3, 1, figsize=(16, 10))
fig.suptitle('9. Time Series: Rolling Maximum & Events Over Time', fontsize=14, fontweight='bold')

for idx, station in enumerate(STATIONS):
    ax = axes[idx]
    
    rolling_max = results[station]['rolling_max']
    dates = raw_data['date']
    threshold = results[station]['threshold']
    
    ax.plot(dates, rolling_max, linewidth=1, alpha=0.7, label='14-day Rolling Max', color='steelblue')
    
    events = results[station]['events']
    event_dates = []
    event_peaks = []
    for i, event_num in enumerate(events):
        if event_num > 0:
            event_dates.append(dates.iloc[i])
            event_peaks.append(rolling_max[i])
    
    ax.scatter(event_dates, event_peaks, color='red', s=20, alpha=0.5, label='Flood Events')
    ax.axhline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    
    ax.set_ylabel('Water Level', fontweight='bold')
    ax.set_title(f'{station.upper()} Station', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('Date', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}09_Time_Series.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_Time_Series.png")
plt.close()

#  SUMMARY 
print("\\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\\nAll visualizations saved to: {OUTPUT_DIR}")
print("\\nFiles created:")
print("  1. 01_Event_Frequency_By_Year.png")
print("  2. 02_Duration_Distribution.png")
print("  3. 03_Peak_Level_Distribution.png")
print("  4. 04_Decade_Trends.png")
print("  5. 05_Seasonality_Heatmap.png")
print("  6. 06_Monthly_Distribution.png")
print("  7. 07_Duration_vs_Peak.png")
print("  8. 08_Spatial_Comparison.png")
print("  9. 09_Time_Series.png")
