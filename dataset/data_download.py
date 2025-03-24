import gdown
import os

def download_from_gdrive(gdrive_url, output_dir="MFSD_dataset", output_filename="MFSD_dataset.zip"):
    # Extract the file ID from the URL
    file_id = gdrive_url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Full path for the downloaded file
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Download the file using gdown
        print("Downloading dataset from Google Drive...")
        gdown.download(download_url, output_path, quiet=False)
        print(f"Dataset downloaded successfully to {output_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

# Google Drive URL from the README
gdrive_url = "https://drive.google.com/file/d/1KycQj4dik91RuBGvbhDJou7YDQEKAH2Z/view"

# Run the download
download_from_gdrive(gdrive_url)