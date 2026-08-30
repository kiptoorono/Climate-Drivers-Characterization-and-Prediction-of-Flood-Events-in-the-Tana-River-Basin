"""
Download CHIRPS v3.0 Daily RNL (ERA5-downscaled) Data
=====================================================
URL: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byMonth/

Usage:
    python download_chirps.py                    # Download all years/months
    python download_chirps.py --years 1990-2020  # Specific year range
    python download_chirps.py --months 3,4,5     # Only Mar-May (Long Rains)
    python download_chirps.py --months 3,4,5,10,11,12  # Both rain seasons
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl/netcdf/byMonth/"
SAVE_DIR = Path("F:/CHIRPS_v3_rnl")  
FILE_PATTERN = "chirps-v3.0.{year}.{month:02d}.days_p05.nc"

# Default settings
START_YEAR = 1981
END_YEAR = 2026
ALL_MONTHS = list(range(1, 13))

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
CHUNK_SIZE = 1024 * 1024  # 1 MiB read chunks


def get_file_size(url):
    """Get file size from headers without downloading."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as resp:
            size = resp.headers.get('Content-Length')
            return int(size) if size else None
    except Exception:
        return None


def format_size(bytes_val):
    """Format bytes to human readable."""
    if bytes_val is None:
        return "unknown"
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TiB"


def download_file(url, dest_path, resume=True):
    """Download a file with resume support and retries."""
    for attempt in range(MAX_RETRIES):
        try:
            # Check if file already exists (resume support)
            existing_size = 0
            if resume and dest_path.exists():
                existing_size = dest_path.stat().st_size
                if existing_size > 0:
                    # Check if complete
                    remote_size = get_file_size(url)
                    if remote_size and existing_size >= remote_size:
                        print(f"    [SKIP] Already complete ({format_size(existing_size)})")
                        return True
                    print(f"    [RESUME] Starting from {format_size(existing_size)}")

            # Set up request with range header for resume
            req = urllib.request.Request(url)
            if existing_size > 0:
                req.add_header('Range', f'bytes={existing_size}-')

            with urllib.request.urlopen(req, timeout=60) as response:
                # If resuming, check if server supports range requests
                if existing_size > 0 and response.status == 200:
                    # Server doesn't support range, restart
                    existing_size = 0
                elif response.status == 206:
                    pass  # Partial content, good for resume

                total_size = response.headers.get('Content-Length')
                total_size = int(total_size) + existing_size if total_size else None

                mode = 'ab' if existing_size > 0 else 'wb'
                downloaded = existing_size

                with open(dest_path, mode) as f:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress indicator
                        if total_size:
                            pct = downloaded / total_size * 100
                            dl_fmt = format_size(downloaded)
                            total_fmt = format_size(total_size)
                            print(f"\r    [{pct:5.1f}%] {dl_fmt} / {total_fmt}", end='', flush=True)
                        else:
                            print(f"\r    {format_size(downloaded)} downloaded", end='', flush=True)

                print()  # New line after progress
                return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"\n    [RETRY {attempt+1}/{MAX_RETRIES}] Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    print(f"    [FAILED] Could not download after {MAX_RETRIES} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download CHIRPS v3.0 daily RNL data")
    parser.add_argument("--years", type=str, default=f"{START_YEAR}-{END_YEAR}",
                        help="Year range, e.g. '1990-2020' or '2019'")
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated months, e.g. '3,4,5' for Mar-May. "
                             "Default: all months (1-12)")
    parser.add_argument("--save-dir", type=str, default=str(SAVE_DIR),
                        help=f"Directory to save files (default: {SAVE_DIR})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume partial downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list files that would be downloaded")
    args = parser.parse_args()

    # Parse year range
    if '-' in args.years:
        start_y, end_y = map(int, args.years.split('-'))
    else:
        start_y = end_y = int(args.years)

    # Parse months
    if args.months:
        months = [int(m.strip()) for m in args.months.split(',')]
    else:
        months = ALL_MONTHS

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build file list
    files = []
    for year in range(start_y, end_y + 1):
        for month in months:
            filename = FILE_PATTERN.format(year=year, month=month)
            url = BASE_URL + filename
            dest = save_dir / filename
            files.append((url, dest, filename))

    # Estimate total size
    
    print("CHIRPS v3.0 Daily RNL Download")
    print(f"URL base:    {BASE_URL}")
    print(f"Save to:     {save_dir}")
    print(f"Years:       {start_y}-{end_y}")
    print(f"Months:      {months}")
    print(f"Total files: {len(files)}")
    print(f"Est. size:   ~{len(files) * 330} MiB (~{len(files) * 330 / 1024:.1f} GiB)")
    print()

    if args.dry_run:
        print("DRY RUN - files that would be downloaded:")
        for url, dest, filename in files:
            exists = "[EXISTS]" if dest.exists() else ""
            print(f"  {filename} {exists}")
        return

    # Download
    success = 0
    failed = 0
    skipped = 0

    for i, (url, dest, filename) in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {filename}")

        if dest.exists() and dest.stat().st_size > 250_000_000:  # ~250 MiB minimum
            print(f"    [SKIP] Already exists ({format_size(dest.stat().st_size)})")
            skipped += 1
            continue

        ok = download_file(url, dest, resume=not args.no_resume)
        if ok:
            success += 1
        else:
            failed += 1

    # Summary
    print(f"  Downloaded: {success}")
    print(f"  Skipped:    {skipped} (already existed)")
    print(f"  Failed:     {failed}")
    print(f"  Location:   {save_dir}")
    print()

if __name__ == "__main__":
    main()
