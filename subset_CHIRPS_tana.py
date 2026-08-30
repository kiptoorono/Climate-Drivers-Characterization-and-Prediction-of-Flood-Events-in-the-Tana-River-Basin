"""
Subset CHIRPS v3.0 Daily Data to Tana River Basin


This reduces ~330 MiB files to ~1-5 MiB per month.

Usage:
    python subset_chirps_tana.py                      # Subset all downloaded files
    python subset_chirps_tana.py --start-year 1990    # Only from 1990 onwards
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray is required. Install with:")
    print("  pip install xarray netCDF4")
    sys.exit(1)

CHIRPS_DIR = Path("F:/CHIRPS_v3_rnl")        # Where you saved CHIRPS files
OUTPUT_DIR = Path("F:/CHIRPS_v3_rnl_tana")    # Where to save subset
FILE_PATTERN = "chirps-v3.0.{year}.{month:02d}.days_p05.nc"

# Tana River Basin bounding box 
# Based on the basin extent upstream of Garissa
LAT_MIN = -2.5    # South boundary
LAT_MAX = 0.5     # North boundary
LON_MIN = 37.0    # West boundary
LON_MAX = 42.0    # East boundary

#create a merged annual or full-period file
CREATE_MERGED = True


def subset_single_file(input_path, output_path, lat_min, lat_max, lon_min, lon_max):
    """Subset a single CHIRPS NetCDF to a bounding box."""
    try:
        ds = xr.open_dataset(input_path)

        # Subset to bounding box
        ds_sub = ds.sel(
            latitude=slice(lat_max, lat_min),  
            longitude=slice(lon_min, lon_max)
        )

        # Check if we got data
        if ds_sub['precip'].size == 0:
            print(f"    [WARN] No data in bounding box for {input_path.name}")
            ds.close()
            return False

        # Save compressed
        ds_sub.to_netcdf(output_path, encoding={
            'precip': {'zlib': True, 'complevel': 4}
        })

        ds.close()
        ds_sub.close()
        return True

    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Subset CHIRPS to Tana Basin")
    parser.add_argument("--input-dir", type=str, default=str(CHIRPS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated months, e.g. '3,4,5'")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.months:
        months = [int(m.strip()) for m in args.months.split(',')]
    else:
        months = list(range(1, 13))

    # Find all matching files
    files = []
    for year in range(args.start_year, args.end_year + 1):
        for month in months:
            filename = FILE_PATTERN.format(year=year, month=month)
            filepath = input_dir / filename
            if filepath.exists():
                files.append(filepath)


    print("CHIRPS v3.0 → Tana Basin Subset")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Bounds:     lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    print(f"Files found: {len(files)}")
    print()

    if not files:
        print("No CHIRPS files found! Check your input directory.")
        print(f"Looking for: {FILE_PATTERN.format(year='YYYY', month='MM')}")
        return

    # Subset each file
    success = 0
    for i, filepath in enumerate(files, 1):
        out_name = filepath.name
        out_path = output_dir / out_name

        print(f"[{i}/{len(files)}] {out_name}", end='')

        if out_path.exists():
            print(" [SKIP]")
            success += 1
            continue

        print(" ... ", end='', flush=True)
        ok = subset_single_file(filepath, out_path, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        if ok:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[OK] ({size_mb:.1f} MiB)")
            success += 1
        else:
            print("[FAILED]")

    print(f"\nSubset complete: {success}/{len(files)} files")

    # Create merged file
    if CREATE_MERGED and success > 0:
        print("Creating merged Tana Basin file...")

        nc_files = sorted(glob.glob(str(output_dir / "chirps-v3.0.*.days_p05.nc")))
        if nc_files:
            ds_merged = xr.open_mfdataset(nc_files, combine='by_coords')
            merged_path = output_dir / "chirps_v3_tana_basin_merged.nc"
            ds_merged.to_netcdf(merged_path, encoding={
                'precip': {'zlib': True, 'complevel': 4}
            })
            size_mb = merged_path.stat().st_size / (1024 * 1024)
            print(f"Merged file: {merged_path} ({size_mb:.1f} MiB)")
            print(f"Period: {ds_merged.time.min().values} to {ds_merged.time.max().values}")
            print(f"Shape: {ds_merged.dims}")

    # Quick validation

    print("VALIDATION")
    nc_files = sorted(glob.glob(str(output_dir / "chirps-v3.0.*.days_p05.nc")))
    if nc_files:
        ds = xr.open_mfdataset(nc_files, combine='by_coords')
        precip = ds['precip']

        print(f"Total days:   {len(precip.time)}")
        print(f"Date range:   {precip.time.min().values} to {precip.time.max().values}")
        print(f"Lat range:    {float(precip.latitude.min()):.2f} to {float(precip.latitude.max()):.2f}")
        print(f"Lon range:    {float(precip.longitude.min()):.2f} to {float(precip.longitude.max()):.2f}")
        print(f"Grid points:  {len(precip.latitude)} x {len(precip.longitude)}")
        print(f"Mean daily precip: {float(precip.mean()):.2f} mm/day")
        print(f"Max daily precip:  {float(precip.max()):.1f} mm/day")

        # Annual totals
        annual = precip.resample(time='1Y').sum()
        print(f"\nAnnual precip range: {float(annual.min()):.0f} to {float(annual.max()):.0f} mm/year")
        print(f"Mean annual precip:  {float(annual.mean()):.0f} mm/year")

        ds.close()
if __name__ == "__main__":
    main()
