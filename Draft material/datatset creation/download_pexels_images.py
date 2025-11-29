import os
import requests
import time

# --- CONFIGURABLE PARAMETERS ---

# Add one or more Pexels API keys to this list.
# The script will automatically switch to the next key if a rate limit is hit.
PEXELS_API_KEYS = [
    'ronQ7GMGMcDmAR3azva65SOGlKQsaGvEv9OYv4xVXB6pNuFlVMKnPnTt',
    'by86Tnb11OIZjhFzMOjTsQmbuVJMPqMAqfRANTPkLfJSvMOtVtjH9Z1t', # Optional: add a second key
    'w105g0tTMpvxYcrcvbJDXwqyPEMrMJtmnKioEKlIBH2kLS9NtGr9thzs',
    "V7atb38sCW8NA1MoztlvmGRYkt8bHj8eBGT7XGB30eH5JZlNpZYH0vGI"# Optional: add more keys
]

# --- CATEGORY DEFINITIONS ---
# Define all the categories you want to download here.
# The script will process them one by one in this order.
# I've used your descriptions to create more effective search queries.
CATEGORIES_TO_DOWNLOAD = [
    {
        "query": "urban architecture  4k",
        "output_dir": "dataset/raw_hr/urban_architecture",
    },
    {
        "query": "nature landscape vibrant colors details 4k",
        "output_dir": "dataset/raw_hr/natural_landscapes",
    },
    {
        "query": "portrait photography detailed skin detailed hair 8k",
        "output_dir": "dataset/raw_hr/portraits_people",
    },
    {
        "query": "macro photography objects fine details texture",
        "output_dir": "dataset/raw_hr/objects_macro",
    }
]

# --- GLOBAL DOWNLOAD PARAMETERS ---
# These settings apply to all categories defined above.
NUM_IMAGES_PER_CATEGORY = 500 # Number of images to download for each category

# --- DUPLICATION PREVENTION ---
# Each API key uses a different page offset to prevent getting duplicate images:
# Key 0: pages 1, 2, 3, ... | Key 1: pages 51, 52, 53, ... | Key 2: pages 101, 102, 103, ... | Key 3: pages 151, 152, 153, ...
# This ensures different keys never request the same page from Pexels API

# --- TEST MODE (uncomment for testing) ---
# NUM_IMAGES_PER_CATEGORY = 5  # For testing only - download fewer images
# TEST_WAIT_SECONDS = 10       # For testing only - shorter wait time
MIN_WIDTH = 1024              # Minimum image width
MIN_HEIGHT = 1024             # Minimum image height
ORIENTATION = None            # 'landscape', 'portrait', 'square', or None for any

# --- API & SCRIPT CONSTANTS (usually no need to change) ---
BASE_URL = 'https://api.pexels.com/v1/search'
MAX_PER_PAGE = 80             # Pexels API max per_page is 80


def download_pexels_images(query, num_images, output_dir, api_keys, start_key_index,
                           min_width, min_height, orientation):
    """
    Downloads a specified number of images for a given query from Pexels.
    Handles API key rotation if rate limits are hit.

    Returns:
        tuple: (number of images successfully downloaded, index of the last used API key)
    """
    if not any(key and key != 'YOUR_API_KEY_1_HERE' for key in api_keys):
        print("Error: No valid Pexels API keys found. Please add your key(s) to the PEXELS_API_KEYS list.")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    
    images_downloaded = 0
    page = 1
    seen_urls = set()
    current_key_index = start_key_index
    # Track pages processed per API key to avoid duplication
    key_page_usage = {i: 0 for i in range(len(api_keys))}

    while images_downloaded < num_images:
        per_page = min(MAX_PER_PAGE, num_images - images_downloaded)
        params = {
            'query': query,
            'per_page': per_page,
            'page': page  # Will be overridden per key to avoid duplication
        }
        if orientation:
            params['orientation'] = orientation

        # --- Request loop with API key rotation ---
        response = None
        while True: # This loop handles retries with different keys
            if current_key_index >= len(api_keys):
                print("\nError: All API keys have been exhausted or are invalid.")
                return images_downloaded, current_key_index

            # Calculate page offset to avoid duplication across keys
            # Each key gets a different starting page range
            key_offset = current_key_index * 50  # Large offset to ensure different results
            current_page = page + key_offset

            headers = {"Authorization": api_keys[current_key_index]}

            # Update params with key-specific page to avoid duplication
            key_params = params.copy()
            key_params['page'] = current_page

            try:
                response = requests.get(BASE_URL, headers=headers, params=key_params, timeout=15)
                
                if response.status_code == 200:
                    print(f"Using API key {current_key_index + 1}, requesting page {current_page}")
                    break # Success, exit the retry loop

                elif response.status_code == 429: # Rate limit hit
                    print(f"\nWarning: Rate limit hit for key ...{api_keys[current_key_index][-4:]}.")
                    current_key_index += 1
                    print("Switching to the next API key...")
                    time.sleep(1) # Wait a moment before retrying
                    continue # Retry with the new key
                
                else:
                    print(f"\nError fetching page {page} for query '{query}'. Status: {response.status_code} - {response.text}")
                    # For other errors (e.g., 401 Unauthorized), we stop this category.
                    return images_downloaded, current_key_index

            except requests.exceptions.RequestException as e:
                print(f"\nNetwork error: {e}. Stopping download for this category.")
                return images_downloaded, current_key_index
        # --- End of request loop ---
        
        data = response.json()
        photos = data.get('photos', [])

        if not photos:
            print(f'No more photos found for query "{query}".')
            break

        for photo in photos:
            if images_downloaded >= num_images:
                break

            img_url = photo['src']['original']
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            width, height = photo.get('width', 0), photo.get('height', 0)
            if width < min_width or height < min_height:
                continue

            try:
                img_response = requests.get(img_url, timeout=20)
                img_response.raise_for_status() # Raise an exception for bad status codes
                img_data = img_response.content
                
                filename = os.path.join(output_dir, f'{query.split()[0]}_{images_downloaded+1:04d}.jpg')
                with open(filename, 'wb') as f:
                    f.write(img_data)
                
                print(f"({images_downloaded + 1}/{num_images}) Saved: {filename}")
                images_downloaded += 1
                time.sleep(0.1)  # Be polite with image source server

            except requests.exceptions.RequestException as e:
                print(f"Could not download image {img_url}. Error: {e}")

        page += 1
        if images_downloaded < num_images:
            time.sleep(0.5) # Wait before fetching the next page

    print(f"\nFinished category '{query}'. Downloaded {images_downloaded} images to {output_dir}\n")
    return images_downloaded, current_key_index


def main():
    """
    Main function to orchestrate the download of all defined categories.
    """
    current_api_key_index = 0
    total_downloaded = 0
    start_time = time.time()
    
    print("--- Starting Pexels Image Downloader ---")
    
    for i, category in enumerate(CATEGORIES_TO_DOWNLOAD):
        print("--------------------------------------------------")
        print(f"Processing Category {i+1}/{len(CATEGORIES_TO_DOWNLOAD)}: '{category['query']}'")
        print("--------------------------------------------------")
        
        downloaded_count, last_used_key_index = download_pexels_images(
            query=category['query'],
            num_images=NUM_IMAGES_PER_CATEGORY,
            output_dir=category['output_dir'],
            api_keys=PEXELS_API_KEYS,
            start_key_index=current_api_key_index,
            min_width=MIN_WIDTH,
            min_height=MIN_HEIGHT,
            orientation=ORIENTATION
        )
        
        total_downloaded += downloaded_count
        current_api_key_index = last_used_key_index

        # If all API keys were exhausted, stop the script
        if current_api_key_index >= len(PEXELS_API_KEYS):
            print("All API keys have been used up. Stopping the script.")
            break

        # Wait 1.5 hours (5400 seconds) between categories (except after the last one)
        if i < len(CATEGORIES_TO_DOWNLOAD) - 1:  # Don't wait after the last category
            # Check if we're in test mode with shorter wait time
            if 'TEST_WAIT_SECONDS' in globals():
                wait_seconds = TEST_WAIT_SECONDS
                print(f"\nTEST MODE: Waiting {wait_seconds} seconds before processing the next category...")
            else:
                wait_hours = 1.5
                wait_seconds = int(wait_hours * 3600)
                print(f"\nWaiting {wait_hours} hours ({wait_seconds} seconds) before processing the next category...")
                print("This allows API rate limits to reset.")
            time.sleep(wait_seconds)
            print("Wait complete. Continuing with next category.\n")

    end_time = time.time()
    duration = end_time - start_time
    print("==================================================")
    print("All categories processed.")
    print(f"Total images downloaded: {total_downloaded}")
    print(f"Total time elapsed: {duration:.2f} seconds")
    print("==================================================")


if __name__ == '__main__':
    main()