import os
import sys
import tarfile
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    archive_name = "LJSpeech-1.1.tar.bz2"
    target_dir = "LJSpeech-1.1"
    
    if os.path.exists(target_dir):
        print(f"Directory '{target_dir}' already exists. Skipping download.")
        return

    # Download
    if not os.path.exists(archive_name):
        print(f"Downloading LJSpeech dataset from {url}...")
        try:
            download_url(url, archive_name)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            if os.path.exists(archive_name):
                os.remove(archive_name)
            sys.exit(1)
    else:
        print(f"Archive '{archive_name}' already exists. Skipping download.")

    # Extract
    print(f"Extracting '{archive_name}'...")
    try:
        with tarfile.open(archive_name, "r:bz2") as tar:
            tar.extractall()
        print("Extraction completed successfully.")
    except Exception as e:
        print(f"Error extracting archive: {e}")
        sys.exit(1)

    # Clean up archive
    if os.path.exists(archive_name):
        print(f"Removing archive file '{archive_name}' to save space...")
        os.remove(archive_name)
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
