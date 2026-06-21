import os
import sys
import urllib.request

def progress_callback(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100.0, (downloaded / total_size) * 100.0) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading NISQA Corpus: {percent:.2f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
    sys.stdout.flush()

def main():
    url = "https://zenodo.org/record/4728081/files/NISQA_Corpus.zip"
    dest = "NISQA_Corpus.zip"
    
    print(f"Starting download from: {url}")
    print(f"Destination: {os.path.abspath(dest)}")
    
    try:
        urllib.request.urlretrieve(url, dest, progress_callback)
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError downloading file from Zenodo: {e}")
        # Try mirror
        mirror_url = "https://depositonce.tu-berlin.de/bitstream/11303/13012.5/9/NISQA_Corpus.zip"
        print(f"Attempting to download from mirror: {mirror_url}")
        try:
            urllib.request.urlretrieve(mirror_url, dest, progress_callback)
            print("\nDownload completed successfully from mirror!")
        except Exception as err:
            print(f"\nError downloading from mirror: {err}")
            sys.exit(1)

if __name__ == "__main__":
    main()
