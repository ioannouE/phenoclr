import os
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time
import subprocess
import shutil

# Create a session with connection pooling for faster downloads
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def validate_bil_file(file_path, skip_validation=False):
    """
    Validate if the BIL file is not corrupted by trying to read it.
    Returns True if valid, False otherwise.
    """
    if skip_validation:
        return True
        
    try:
        # Try to open and read the file
        with open(file_path, 'rb') as f:
            # Check if file has content
            data = f.read(1024)
            if not data:
                print(f"Warning: {file_path} is empty.")
                return False
            
            # For proper validation, we would need to check if the file
            # can be properly processed as a spectral image, which may require
            # additional libraries like GDAL. This is a basic check.
            return True
    except Exception as e:
        print(f"Error validating {file_path}: {str(e)}")
        return False

def validate_hdr_file(file_path, skip_validation=False):
    """
    Validate if the HDR file is not corrupted by trying to read it.
    Returns True if valid, False otherwise.
    """
    if skip_validation:
        return True
        
    try:
        # Try to open and read the file
        with open(file_path, 'rb') as f:
            # Check if file has content
            data = f.read(1024)
            if not data:
                print(f"Warning: {file_path} is empty.")
                return False
            
            # Basic check for common HDR file content
            text_data = data.decode('utf-8', errors='ignore')
            if any(keyword in text_data for keyword in ['ENVI', 'samples', 'lines', 'bands']):
                return True
            else:
                print(f"Warning: {file_path} does not appear to be a valid HDR file.")
                return False
    except Exception as e:
        print(f"Error validating {file_path}: {str(e)}")
        return False

def download_file_requests(session, file_url, file_path, max_retries=3, skip_validation=False, chunk_size=None):
    """
    Download a file using requests with retry mechanism.
    Returns True if download and validation successful, False otherwise.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Download the file
            if chunk_size:
                # Chunked download for larger files
                with session.get(file_url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
            else:
                # Standard download
                file_response = session.get(file_url, timeout=30)
                file_response.raise_for_status()
                
                # Save the file
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
            
            # Validate the downloaded file
            if validate_file(file_path, skip_validation):
                return True
            else:
                # Remove invalid file and retry
                os.remove(file_path)
                retry_count += 1
                time.sleep(1)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)  # Wait before retrying
            else:
                return False
        except Exception as e:
            # Any other exception, don't retry
            return False
    
    return False

def download_file_wget(file_url, file_path, max_retries=3, skip_validation=False):
    """
    Download a file using wget with retry mechanism.
    Returns True if download and validation successful, False otherwise.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run wget with appropriate options
            result = subprocess.run([
                "wget", 
                "--quiet",           # Reduce output
                "-t", "2",           # Retry attempts
                "-T", "30",          # Timeout
                "--no-check-certificate",  # Skip SSL verification
                "-O", str(file_path),  # Output file
                file_url
            ], check=True)
            
            # Validate the downloaded file
            if validate_file(file_path, skip_validation):
                return True
            else:
                # Remove invalid file and retry
                os.remove(file_path)
                retry_count += 1
                time.sleep(1)
        except subprocess.SubprocessError as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)
            else:
                return False
        except Exception as e:
            # Any other exception, don't retry
            return False
    
    return False

def download_file_aria2(file_url, file_path, max_retries=3, skip_validation=False):
    """
    Download a file using aria2 with retry mechanism.
    Returns True if download and validation successful, False otherwise.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run aria2c with optimized settings
            result = subprocess.run([
                "aria2c",
                "--file-allocation=none",  # Faster allocation
                "-x", "16",                # Connections per server
                "-s", "16",                # Split file into 16 parts
                "-j", "16",                # Max concurrent downloads
                "--retry-wait=2",          # Wait between retries
                "--max-tries=2",           # Number of retries
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--quiet",                 # Reduce output
                "-d", str(file_path.parent),  # Directory
                "-o", file_path.name,         # Filename
                file_url
            ], check=True)
            
            # Validate the downloaded file
            if validate_file(file_path, skip_validation):
                return True
            else:
                # Remove invalid file and retry
                os.remove(file_path)
                retry_count += 1
                time.sleep(1)
        except subprocess.SubprocessError as e:
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)
            else:
                return False
        except Exception as e:
            # Any other exception, don't retry
            return False
    
    return False

def download_file(file_url, file_path, download_method='requests', session=None, max_retries=3, 
                 skip_validation=False, chunk_size=None):
    """
    Download a file using the specified method.
    Returns True if download and validation successful, False otherwise.
    """
    if download_method == 'wget' and shutil.which('wget'):
        return download_file_wget(file_url, file_path, max_retries, skip_validation)
    elif download_method == 'aria2' and shutil.which('aria2c'):
        return download_file_aria2(file_url, file_path, max_retries, skip_validation)
    else:
        # Fall back to requests if the selected method isn't available
        if download_method != 'requests' and download_method in ['wget', 'aria2']:
            print(f"Warning: {download_method} not found, falling back to requests")
        return download_file_requests(session, file_url, file_path, max_retries, skip_validation, chunk_size)

def validate_file(file_path, skip_validation=False):
    """
    Validate a file based on its extension.
    Returns True if valid, False otherwise.
    """
    if file_path.name.endswith('.bil'):
        return validate_bil_file(file_path, skip_validation)
    elif file_path.name.endswith('.hdr'):
        return validate_hdr_file(file_path, skip_validation)
    else:
        # For unknown file types, just check if it has content
        try:
            if skip_validation:
                return True
            return file_path.stat().st_size > 0
        except Exception:
            return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download spectral data files.')
    parser.add_argument('--output-dir', type=str, default="lepidoptera_scans",
                        help='Directory to save downloaded files (default: lepidoptera_scans)')
    parser.add_argument('--max-workers', type=int, default=20,
                        help='Maximum number of concurrent downloads (default: 20)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force download even if files already exist')
    parser.add_argument('--download-method', type=str, choices=['requests', 'wget', 'aria2'], 
                        default='requests', help='Download method to use (default: requests)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation to speed up downloads (use with caution)')
    parser.add_argument('--chunk-size', type=int, default=0,
                        help='Use chunked downloads with specified chunk size in bytes (default: 0, disabled)')
    parser.add_argument('--download-hdr', action='store_true', default=True,
                        help='Download corresponding .hdr files for .bil files (default: True)')
    args = parser.parse_args()
    
    # Create session for requests downloads
    session = create_session()
    
    # URL of the browse page
    base_url = "https://arts.st-andrews.ac.uk/lepidoptera/browse.html"
    
    # Create a directory to save the downloaded files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Files will be downloaded to: {output_dir.absolute()}")
    print(f"Download method: {args.download_method}")
    print(f"HDR files will {'also be downloaded' if args.download_hdr else 'not be downloaded'}")
    
    # Check if selected download method is available
    if args.download_method == 'wget' and not shutil.which('wget'):
        print("Warning: wget not found, falling back to requests")
        args.download_method = 'requests'
    elif args.download_method == 'aria2' and not shutil.which('aria2c'):
        print("Warning: aria2c not found, falling back to requests")
        args.download_method = 'requests'
    
    # Fetch the browse page
    try:
        response = session.get(base_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching the browse page: {str(e)}")
        return
    
    # Find all links to .bil files
    download_tasks = []
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.bil'):
            # Extract just the filename regardless of the href format
            filename = os.path.basename(href)
            
            # Construct the correct URL format based on the example
            file_url = f"https://arts.st-andrews.ac.uk/lepidoptera/uploads/{filename}"
            file_path = output_dir / filename
            
            # Skip if file already exists and is valid (unless force download is enabled)
            if not args.force_download and file_path.exists() and validate_file(file_path, args.skip_validation):
                print(f"Skipping {filename} - already downloaded and validated")
            else:
                download_tasks.append((file_url, file_path))
            
            # Add corresponding .hdr file if requested
            if args.download_hdr:
                hdr_filename = f"{filename}.hdr"
                hdr_url = f"https://arts.st-andrews.ac.uk/lepidoptera/uploads/{hdr_filename}"
                hdr_path = output_dir / hdr_filename
                
                # Skip if HDR file already exists and is valid
                if not args.force_download and hdr_path.exists() and validate_file(hdr_path, args.skip_validation):
                    print(f"Skipping {hdr_filename} - already downloaded and validated")
                else:
                    download_tasks.append((hdr_url, hdr_path))
    
    # Show total number of files to download
    total_files = len(download_tasks)
    print(f"Found {total_files} files to download")
    
    # Keep track of downloads
    successful = {'bil': 0, 'hdr': 0, 'other': 0}
    failed = {'bil': 0, 'hdr': 0, 'other': 0}
    
    # Configure chunk size
    chunk_size = args.chunk_size if args.chunk_size > 0 else None
    
    # Use ThreadPoolExecutor to download files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all download tasks
        future_to_file = {
            executor.submit(
                download_file, 
                url, 
                path, 
                args.download_method,
                session,
                3,  # max_retries
                args.skip_validation,
                chunk_size
            ): (url, path) 
            for url, path in download_tasks
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(future_to_file), unit='file') as progress_bar:
            for future in concurrent.futures.as_completed(future_to_file):
                url, path = future_to_file[future]
                filename = path.name
                
                # Determine file type for statistics
                file_type = 'other'
                if filename.endswith('.bil'):
                    file_type = 'bil'
                elif filename.endswith('.hdr'):
                    file_type = 'hdr'
                
                try:
                    if future.result():
                        successful[file_type] += 1
                        progress_bar.set_description(f"Downloaded {filename}")
                    else:
                        failed[file_type] += 1
                        progress_bar.set_description(f"Failed {filename}")
                except Exception as e:
                    failed[file_type] += 1
                    progress_bar.set_description(f"Error {filename}: {str(e)}")
                
                progress_bar.update(1)
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Successfully downloaded and validated:")
    print(f"    BIL files: {successful['bil']}")
    if args.download_hdr:
        print(f"    HDR files: {successful['hdr']}")
    print(f"    Other files: {successful['other']}")
    print(f"  Failed downloads:")
    print(f"    BIL files: {failed['bil']}")
    if args.download_hdr:
        print(f"    HDR files: {failed['hdr']}")
    print(f"    Other files: {failed['other']}")
    print(f"  All files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
