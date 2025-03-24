import gdown
import os
import sys
import zipfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.variables import *

def download_from_gdrive(gdrive_url, output_dir="MFSD_dataset", output_filename="MFSD_dataset.zip"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full path for the downloaded file
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Check if zip file already exists
        if os.path.exists(output_path):
            print(f"Zip file {output_filename} already exists. Skipping download.")
        else:
            # Extract the file ID from the URL
            file_id = gdrive_url.split('/d/')[1].split('/')[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download the file using gdown
            print("Downloading dataset from Google Drive...")
            gdown.download(download_url, output_path, quiet=False)
            print(f"Dataset downloaded successfully to {output_path}")
        
        # Extract the zip file
        print(f"Extracting {output_filename}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extraction complete. Files extracted to {output_dir}")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")

# Run the download
if __name__ == "__main__":
    download_from_gdrive(DATASET_URL_2)