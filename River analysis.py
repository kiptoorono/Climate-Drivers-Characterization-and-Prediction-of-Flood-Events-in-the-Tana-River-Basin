import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURATION =====
EXCEL_FILE = 'TanaRiver Flow.xlsx'
SHEET_NAME = 'Rawdata'
WINDOW_SIZE = 14

THRESHOLDS = {
    'Bura': 721,
    'Galole': 723,
    'Garsen': 1723
}

STATIONS = ['Bura', 'Galole', 'Garsen']

#  YEAR-AWARE ROLLING MAX 
def year_aware_rolling_max(series, dates, year_col, window=14):
    """Calculate 14-day rolling max that respects year boundaries."""
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

#  FUNCTION 2: IDENTIFY FLOOD EVENTS 
def identify_flood_events(rolling_max_series, threshold):
    """Identify flood events: start when rolling_max > threshold, end when <= threshold."""
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

#  FUNCTION 3: MARK EVENT DATES =====
def mark_event_dates(event_numbers, dates):
    """Mark event start/end dates and duration."""
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

#  FUNCTION 4: CALCULATE EVENT BLOCK MAX 
def calculate_event_block_max(event_numbers, rolling_values):
    """Record highest rolling window value at event end date."""
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

print("=" * 80)
print("TIER 1: FLOOD EVENT CHARACTERIZATION & FREQUENCY ANALYSIS")
print("=" * 80)

#  LOAD & PREPARE DATA 
print("\n[STEP 1] Loading data...")
raw_data = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=1, usecols=[0, 1, 2, 3])
raw_data.columns = ['date', 'Bura', 'Galole', 'Garsen']
raw_data['date'] = pd.to_datetime(raw_data['date'], errors='coerce')

for col in STATIONS:
    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

raw_data['year'] = raw_data['date'].dt.year
raw_data['month'] = raw_data['date'].dt.month
raw_data = raw_data.dropna(subset=['date']).reset_index(drop=True)

print(f"✓ Data loaded: {raw_data['date'].min()} to {raw_data['date'].max()}")
print(f"✓ Total records: {len(raw_data)}")
print(f"✓ Years covered: {raw_data['year'].min()} to {raw_data['year'].max()} ({raw_data['year'].max() - raw_data['year'].min() + 1} years)")

#  PROCESS EACH STATION 
results = {}

for station in STATIONS:
    print(f"\n[STEP 2a] Processing {station} station...")
    
    threshold = THRESHOLDS[station]
    
    # Calculate rolling max
    rolling_max = year_aware_rolling_max(raw_data[station], raw_data['date'], raw_data['year'], WINDOW_SIZE)
    
    # Identify events
    events = identify_flood_events(rolling_max, threshold)
    
    # Mark dates
    start_dates, end_dates, durations = mark_event_dates(events, raw_data['date'])
    
    # Get block max
    block_max = calculate_event_block_max(events, rolling_max)
    
    # Store results
    results[station] = {
        'rolling_max': rolling_max,
        'events': events,
        'start_dates': start_dates,
        'end_dates': end_dates,
        'durations': durations,
        'block_max': block_max,
        'threshold': threshold
    }
    
    print(f"✓ {station}: threshold={threshold}, total events identified")

print("\n" + "=" * 80)
print("TIER 1 ANALYSIS 1: EVENT FREQUENCY & CHARACTERIZATION")
print("=" * 80)

#  ANALYSIS 1: EVENT FREQUENCY 
for station in STATIONS:
    print(f"\n{station.upper()} STATION")
    print("-" * 60)
    
    events = results[station]['events']
    durations = results[station]['durations']
    block_max = results[station]['block_max']
    threshold = results[station]['threshold']
    
    # Count events per year
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
    
    # Overall stats
    total_events = len(event_df)
    years_coverage = raw_data['year'].max() - raw_data['year'].min() + 1
    events_per_year = total_events / years_coverage
    
    print(f"Total flood events: {total_events}")
    print(f"Years covered: {years_coverage}")
    print(f"Average events/year: {events_per_year:.2f}")
    print(f"\nDuration (days):")
    print(f"  Mean: {event_df['duration'].mean():.2f}")
    print(f"  Median: {event_df['duration'].median():.2f}")
    print(f"  Min: {event_df['duration'].min()}")
    print(f"  Max: {event_df['duration'].max()}")
    print(f"\nPeak level (m or unit):")
    print(f"  Mean: {event_df['peak'].mean():.2f}")
    print(f"  Median: {event_df['peak'].median():.2f}")
    print(f"  Min: {event_df['peak'].min():.2f}")
    print(f"  Max: {event_df['peak'].max():.2f}")
    print(f"  Std Dev: {event_df['peak'].std():.2f}")
    print(f"\nExceedance above threshold:")
    event_df['exceedance'] = event_df['peak'] - threshold
    print(f"  Mean exceedance: {event_df['exceedance'].mean():.2f}")
    print(f"  Max exceedance: {event_df['exceedance'].max():.2f}")
    
    # Store for later analysis
    results[station]['event_df'] = event_df

print("\n" + "=" * 80)
print("TIER 1 ANALYSIS 2: TREND ANALYSIS (Events/Decade)")
print("=" * 80)

for station in STATIONS:
    print(f"\n{station.upper()}")
    print("-" * 60)
    
    event_df = results[station]['event_df']
    
    # Bin into decades
    event_df['decade'] = (event_df['year'] // 10) * 10
    decade_counts = event_df.groupby('decade').size()
    decade_durations = event_df.groupby('decade')['duration'].mean()
    decade_peaks = event_df.groupby('decade')['peak'].mean()
    
    print("\nEvents by decade:")
    for decade, count in decade_counts.items():
        print(f"  {decade}s: {count} events")
    
    print("\nAverage duration by decade (days):")
    for decade, duration in decade_durations.items():
        print(f"  {decade}s: {duration:.2f}")
    
    print("\nAverage peak by decade:")
    for decade, peak in decade_peaks.items():
        print(f"  {decade}s: {peak:.2f}")

print("\n" + "=" * 80)
print("TIER 1 ANALYSIS 3: SEASONALITY (Month of Event Start)")
print("=" * 80)

for station in STATIONS:
    print(f"\n{station.upper()}")
    print("-" * 60)
    
    events = results[station]['events']
    start_dates = results[station]['start_dates']
    
    months_with_events = []
    for idx, date in enumerate(start_dates):
        if date is not None:
            months_with_events.append(date.month)
    
    month_counts = Counter(months_with_events)
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    print("Flood event starts by month:")
    for month in range(1, 13):
        count = month_counts.get(month, 0)
        bar = '█' * count
        print(f"  {month_names[month]:3s}: {count:2d} {bar}")

print("\n" + "=" * 80)
print("TIER 1 ANALYSIS 4: SPATIAL COMPARISON (3 Stations)")
print("=" * 80)

summary_table = []
for station in STATIONS:
    event_df = results[station]['event_df']
    summary_table.append({
        'Station': station,
        'Total Events': len(event_df),
        'Events/Year': len(event_df) / (raw_data['year'].max() - raw_data['year'].min() + 1),
        'Avg Duration (days)': event_df['duration'].mean(),
        'Avg Peak': event_df['peak'].mean(),
        'Max Peak': event_df['peak'].max(),
        'Threshold': results[station]['threshold']
    })

summary_df = pd.DataFrame(summary_table)
print("\n", summary_df.to_string(index=False))

# Correlation between stations
print("\n\nPeak values correlation between stations:")
for i, st1 in enumerate(STATIONS):
    for st2 in STATIONS[i+1:]:
        peaks1 = [x for x in results[st1]['block_max'] if x is not None]
        peaks2 = [x for x in results[st2]['block_max'] if x is not None]
        
        # Match by index
        valid_indices = [(idx, results[st1]['block_max'][idx], results[st2]['block_max'][idx]) 
                         for idx in range(len(results[st1]['block_max'])) 
                         if results[st1]['block_max'][idx] is not None and results[st2]['block_max'][idx] is not None]
        
        if len(valid_indices) > 1:
            p1 = [x[1] for x in valid_indices]
            p2 = [x[2] for x in valid_indices]
            corr = np.corrcoef(p1, p2)[0, 1]
            print(f"  {st1} vs {st2}: r = {corr:.3f}")

print("\n" + "=" * 80)
print("TIER 1 ANALYSIS 5: ANNUAL MAXIMUM & POT PEAKS")
print("=" * 80)

for station in STATIONS:
    print(f"\n{station.upper()}")
    print("-" * 60)
    
    event_df = results[station]['event_df']
    
    # Annual maximum
    annual_max = event_df.groupby('year')['peak'].max()
    print(f"\nAnnual Maximum floods:")
    print(f"  Mean: {annual_max.mean():.2f}")
    print(f"  Median: {annual_max.median():.2f}")
    print(f"  Std Dev: {annual_max.std():.2f}")
    
    # POT (all peaks)
    all_peaks = event_df['peak'].values
    print(f"\nAll flood peaks (POT):")
    print(f"  Count: {len(all_peaks)}")
    print(f"  Mean: {all_peaks.mean():.2f}")
    print(f"  Median: {np.median(all_peaks):.2f}")
    print(f"  Std Dev: {all_peaks.std():.2f}")

