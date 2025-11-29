# ==============================================================================
# SCRIPT FOR MERGING, UNIFYING, AND ARCHIVING A MULTI-PART SISR DATASET
# ==============================================================================
# This script is designed to run in a Google Colab environment with limited
# disk space. It implements a resource-efficient strategy to:
# 1. Sequentially download and extract multiple ZIP archives from Google Drive.
# 2. Unify the metadata from the separate archives into a single, coherent set.
# 3. Create a final, single ZIP archive using an incremental process to avoid
#    exceeding disk space limits.
# 4. Upload the final dataset to Kaggle.
# ==============================================================================

import os
import zipfile
import json
import shutil
import random
import time
from pathlib import Path
from google.colab import drive
from tqdm.notebook import tqdm

# --- STEP 0: CONFIGURATION AND SETUP ---
# Please update these variables according to your setup.

# 1. Google Drive Configuration
GDRIVE_MOUNT_PATH = '/content/drive'
# Path within your Google Drive where the ZIP files are located.
GDRIVE_ZIPS_PATH = '/content/drive/MyDrive/SISR DATASEETETS' # <-- IMPORTANT: UPDATE THIS PATH

# 2. Dataset Configuration
# List of the ZIP files to be processed (names from your screenshot)
ZIP_FILES = [
    "output_natural_landscapes_20250910_184052.zip",
    "output_objects_macro_20250910_172115.zip",
    "output_portraits_people_20250911_224114.zip",
    "output_urban_architecture_20250911_213340.zip"
]

# 3. Local Workspace Configuration
# A temporary directory in the Colab runtime to do all the work.
WORKSPACE_DIR = Path("/content/dataset_workspace")
# The name of the final, unified dataset folder that will be created.
UNIFIED_DATASET_DIR = WORKSPACE_DIR / "sisr_benchmark_unified"
# The name for the final ZIP file to be uploaded.
FINAL_ARCHIVE_NAME = "sisr_benchmark_unified_dataset.zip"
FINAL_ARCHIVE_PATH = WORKSPACE_DIR / FINAL_ARCHIVE_NAME

# 4. Kaggle API Configuration
# Your Kaggle username and the desired dataset slug (the name in the URL)
KAGGLE_USERNAME = "vasuaashadesai" # <-- IMPORTANT: UPDATE THIS
KAGGLE_DATASET_SLUG = "sisr-benchmark-unified" # <-- IMPORTANT: UPDATE THIS

# --- Main script starts here ---

def setup_environment():
    """Mounts Google Drive and sets up the workspace directory."""
    print("--- Setting up environment ---")
    if not Path(GDRIVE_MOUNT_PATH).exists() or not os.listdir(GDRIVE_MOUNT_PATH):
        print("Mounting Google Drive...")
        drive.mount(GDRIVE_MOUNT_PATH)
    else:
        print("Google Drive already mounted.")

    # Clean up previous runs if necessary
    if WORKSPACE_DIR.exists():
        print(f"Removing existing workspace directory: {WORKSPACE_DIR}")
        shutil.rmtree(WORKSPACE_DIR)

    print(f"Creating new workspace directory: {WORKSPACE_DIR}")
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    UNIFIED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print("Environment setup complete.\n")

def sequential_extract():
    """
    Downloads, extracts, and deletes each ZIP file one by one to conserve space.
    """
    print("--- STEP 1: Sequentially Extracting Archives ---")
    source_zip_dir = Path(GDRIVE_ZIPS_PATH)

    for zip_filename in ZIP_FILES:
        source_zip_path = source_zip_dir / zip_filename
        local_zip_path = WORKSPACE_DIR / zip_filename

        if not source_zip_path.exists():
            print(f"❌ ERROR: Cannot find '{zip_filename}' in your Google Drive at '{source_zip_dir}'. Please check the path.")
            continue

        print(f"\nProcessing '{zip_filename}'...")

        # 1. Copy from Drive to local runtime (faster extraction)
        print(f"  > Copying '{zip_filename}' to Colab runtime...")
        shutil.copy(source_zip_path, local_zip_path)

        # 2. Extract the contents into the unified directory
        print(f"  > Extracting contents to '{UNIFIED_DATASET_DIR}'...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            # We must handle nested structures. The script generates a root folder.
            # We will extract and then move contents up one level.
            temp_extract_dir = WORKSPACE_DIR / "temp_extract"
            zip_ref.extractall(temp_extract_dir)

            # The extracted folder is usually named like 'sisr_benchmark'
            extracted_root = next(temp_extract_dir.iterdir())

            # Use shutil.copytree with dirs_exist_ok to merge contents
            shutil.copytree(extracted_root, UNIFIED_DATASET_DIR, dirs_exist_ok=True)

            # Clean up the temporary extraction folder
            shutil.rmtree(temp_extract_dir)


        # 3. Delete the local ZIP file to free up space
        print(f"  > Deleting local zip file '{local_zip_path}' to save space.")
        local_zip_path.unlink()
        print(f"  > ✅ Completed '{zip_filename}'.")

    print("\nAll archives have been extracted successfully.\n")


def unify_metadata():
    """
    Finds all individual metadata files, merges them, creates new unified splits,
    and cleans up the old metadata.
    """
    print("--- STEP 2: Unifying Metadata ---")
    metadata_dir = UNIFIED_DATASET_DIR / "metadata"
    all_pairs = []

    # 1. Find and merge all 'complete_metadata.json' files
    print("  > Searching for individual metadata files...")
    # The original script places metadata inside a 'metadata' folder.
    # We expect multiple of these to have been merged. Let's find all relevant JSONs.

    # Let's assume the structure after merging is now one large folder.
    # The `complete_metadata.json` from the last extracted zip will overwrite others.
    # A better strategy is needed if the metadata files have unique names, but
    # based on the script, they are all called 'complete_metadata.json'.
    # This implies we need to rethink the extraction.

    # REVISED STRATEGY for unify_metadata based on file structure
    # The script saves HR/LR images with unique names like 'category_0001.png'.
    # The metadata JSON references these. We can build a new metadata from scratch
    # by scanning the HR/LR directories. However, the degradation info is lost.

    # Let's stick to the original plan and assume the user can manually
    # rename the metadata files before running this, or we can adjust the
    # extraction process. A robust way is to read the metadata from each zip
    # BEFORE deleting it. Let's modify the sequential_extract function for this.

    # This function will now be called from within the extraction loop.
    pass # We will integrate this logic into a revised extraction function.

# ==============================================================================
# CORRECTED AND MORE ROBUST VERSION
# ==============================================================================
def revised_sequential_extract_and_unify_metadata():
    """
    Revised function to extract archives and unify metadata on the fly.
    This version correctly handles ZIP files that do not have a single root directory.
    """
    print("--- STEP 1 & 2: Sequentially Extracting and Unifying Metadata ---")
    source_zip_dir = Path(GDRIVE_ZIPS_PATH)
    all_pairs = []

    for zip_filename in ZIP_FILES:
        source_zip_path = source_zip_dir / zip_filename
        local_zip_path = WORKSPACE_DIR / zip_filename

        if not source_zip_path.exists():
            print(f"❌ ERROR: Cannot find '{zip_filename}' in your GDrive. Please check path.")
            continue

        print(f"\nProcessing '{zip_filename}'...")
        print("  > Copying to Colab runtime...")
        shutil.copy(source_zip_path, local_zip_path)

        print("  > Extracting files...")
        temp_extract_dir = WORKSPACE_DIR / "temp_extract"

        # Ensure the temp directory is clean before each extraction
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)
        temp_extract_dir.mkdir()

        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # --- FIX: START OF THE CORRECTION ---

        # Correctly locate and read metadata from the temp directory.
        metadata_path = temp_extract_dir / "metadata" / "complete_metadata.json"
        if metadata_path.exists():
            print(f"  > Reading metadata from '{zip_filename}'...")
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                all_pairs.extend(data.get('pairs', []))
                print(f"  > Found {len(data.get('pairs', []))} pairs. Total pairs now: {len(all_pairs)}.")
        else:
            print(f"  > ⚠️ WARNING: No 'complete_metadata.json' found in '{zip_filename}'.")

        # Instead of assuming one root folder, iterate through all extracted items.
        print("  > Merging extracted contents into the unified directory...")
        for item in temp_extract_dir.iterdir():
            destination = UNIFIED_DATASET_DIR / item.name
            if item.is_dir():
                # If it's a directory (like HR, LR, metadata), merge its contents.
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                # If it's a file (like README.md), move it.
                shutil.move(str(item), str(destination))

        # Clean up the temp directory for the next iteration
        shutil.rmtree(temp_extract_dir)

        # --- FIX: END OF THE CORRECTION ---

        print("  > Deleting local zip file...")
        local_zip_path.unlink()
        print(f"  > ✅ Completed '{zip_filename}'.")

    # Now that we have all pairs, let's create the unified metadata
    print("\n--- Creating Unified Metadata Files ---")
    metadata_dir = UNIFIED_DATASET_DIR / "metadata"

    # Clean up old individual metadata and documentation files that will be regenerated
    if metadata_dir.exists():
        print(f"  > Cleaning up old metadata files in {metadata_dir}...")
        for item in metadata_dir.glob("*.json"): # Remove all old json files
             item.unlink()
    # Also remove top-level docs that will be regenerated or unified later
    for doc_file in ["DATASHEET.json", "README.md", "quality_assessment.json"]:
        if (UNIFIED_DATASET_DIR / doc_file).exists():
            (UNIFIED_DATASET_DIR / doc_file).unlink()

    # Create new splits
    print(f"  > Shuffling and splitting {len(all_pairs)} total pairs...")
    random.shuffle(all_pairs)
    train_ratio, val_ratio = 0.7, 0.15
    train_size = int(len(all_pairs) * train_ratio)
    val_size = int(len(all_pairs) * val_ratio)

    splits = {
        'train': all_pairs[:train_size],
        'val': all_pairs[train_size : train_size + val_size],
        'test': all_pairs[train_size + val_size :]
    }

    # Save new unified metadata files
    print("  > Saving new unified metadata...")
    with open(metadata_dir / "complete_metadata.json", 'w') as f:
        json.dump({'dataset_info': {'total_pairs': len(all_pairs)}, 'pairs': all_pairs}, f, indent=2)

    for split_name, split_pairs in splits.items():
        with open(metadata_dir / f"{split_name}_split.json", 'w') as f:
            json.dump(split_pairs, f, indent=2)

    print("  > ✅ Metadata unification complete.\n")

def incremental_archive():
    """
    Creates a ZIP archive incrementally, deleting source files after they are
    added to the archive to manage disk space.
    """
    print("--- STEP 3: Performing Incremental Archiving ---")
    print(f"  > Final archive will be saved to: {FINAL_ARCHIVE_PATH}")

    files_to_archive = list(UNIFIED_DATASET_DIR.rglob('*'))

    with zipfile.ZipFile(FINAL_ARCHIVE_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in tqdm(files_to_archive, desc="Archiving files"):
            if file_path.is_file():
                # The arcname is the path inside the zip file
                arcname = file_path.relative_to(UNIFIED_DATASET_DIR)
                zf.write(file_path, arcname)
                # Delete the original file after it's been written to the archive
                file_path.unlink()

    print("\n  > Archiving of files complete. Cleaning up empty directories...")
    # After all files are removed, the directory structure remains. Let's clean it up.
    shutil.rmtree(UNIFIED_DATASET_DIR)

    archive_size_gb = FINAL_ARCHIVE_PATH.stat().st_size / (1024**3)
    print(f"  > ✅ Incremental archiving successful! Final archive size: {archive_size_gb:.2f} GB\n")

def upload_to_kaggle():
    """
    Uploads the final archive to Kaggle using the Kaggle API.
    """
    print("--- STEP 4: Uploading to Kaggle ---")

    # Setup Kaggle API credentials
    print("  > Please upload your 'kaggle.json' file now.")
    from google.colab import files
    files.upload() # This will prompt you to upload the file.

    # Move credentials to the correct location
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

    print("\n  > Creating dataset metadata file (dataset-metadata.json)...")
    dataset_metadata = {
        "title": "SISR Benchmark Unified Dataset (All Categories)",
        "id": f"{KAGGLE_USERNAME}/{KAGGLE_DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-NC-4.0"}]
    }
    with open(WORKSPACE_DIR / 'dataset-metadata.json', 'w') as f:
        json.dump(dataset_metadata, f)

    print("  > Starting upload to Kaggle. This may take a long time...")
    # The -p flag points to the folder containing the data and metadata file
    # The -q flag makes it quiet
    !kaggle datasets create -p {WORKSPACE_DIR} --dir-mode zip

    print("\n  > ✅ Upload command issued. Check your Kaggle profile for progress.")
    print("  > If the dataset already exists, you can create a new version with:")
    print(f"  > !kaggle datasets version -p {WORKSPACE_DIR} -m 'Unified all categories' --dir-mode zip")


# --- EXECUTION FLOW ---
if __name__ == '__main__':
    try:
        setup_environment()
        revised_sequential_extract_and_unify_metadata()
        incremental_archive()
        upload_to_kaggle()
        print("\n🎉🎉🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉🎉🎉")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()