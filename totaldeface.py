import os
import subprocess
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
import SimpleITK as sitk
import logging
import argparse
from tcia_utils import nbia

def run_totalsegmentator(input_path : str, output_folder : str, gpu_id : int, df : pd.DataFrame, idx : int):
    """
    Function to run TotalSegmentator on a given input file with a specified GPU.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        command = ["TotalSegmentator", "-i", input_path, "-o", output_folder]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run(command, check=True, env=env)

        num_files = len(os.listdir(output_folder))
        df.loc[idx, 'number_files_in_outdir'] = num_files

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        df.loc[idx, 'errors'] = str(e)

def process_files(input_dir, output_dir_base, gpu_ids, df):
    """
    Function to process all files in the input directory.
    """
    # List all files and directories in the input directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    # Filter out only DICOM directories and NIfTI files
    dicom_dirs = [f for f in files if os.path.isdir(f)]
    nifti_files = [f for f in files if os.path.isfile(f) and f.endswith(('.nii', '.nii.gz'))]

    # Create a list of tasks
    tasks = []
    for i, input_path in enumerate(dicom_dirs + nifti_files):
        file_type = 'dcm' if os.path.isdir(input_path) else 'nii'
        output_folder = os.path.join(output_dir_base, os.path.basename(input_path.replace(".gz","").replace(".nii","")))
        output_folder_segs = os.path.join(output_folder, "total_segs")
        df.loc[i] = [input_path, output_folder, output_folder_segs, file_type, 0, None, None]
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((input_path, output_folder_segs, gpu_id, df, i))

    # Run the tasks in parallel
    with Pool(processes=len(gpu_ids)) as pool:
        pool.starmap(run_totalsegmentator, tasks)


def deface(df):
    """
    Deface DICOM images or NIfTI using NIfTI mask files.
    """
    mask_file = "face.nii.gz"
    
    # Filter rows where the segmentation directory contains the mask file and type is 'dcm'
    df_dcm = df[(df['nii_or_dcm'] == 'dcm') & df['output_dir_segs'].apply(lambda x: mask_file in os.listdir(x))]

    for idx, row in df_dcm.iterrows():
        try:
            reader = sitk.ImageSeriesReader()
            dcm_dir = row['input']
            dcm_names = reader.GetGDCMSeriesFileNames(dcm_dir)
            reader.SetFileNames(dcm_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            dcm = reader.Execute()
            dcm_array = sitk.GetArrayViewFromImage(dcm)
            
            nii_path = os.path.join(row['output_dir_segs'], mask_file)
            nii = sitk.ReadImage(nii_path)
            mask = sitk.Resample(nii, dcm, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, nii.GetPixelID())
            mask = sitk.Cast(mask, dcm.GetPixelID())
            zero_mask = sitk.Cast(mask == 0, dcm.GetPixelID())
            mask_value = np.min(dcm_array)
            # masked_image = (mask * mask_value) + (zero_mask * dcm)
            masked_image = sitk.Cast((mask * mask_value) + (zero_mask * dcm), dcm.GetPixelID())

            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()

            for i in range(masked_image.GetDepth()):
                slice_i = masked_image[:, :, i]
                output_file = os.path.join(row['output_dir'], f"defaced_{i:04d}.dcm")
                writer.SetFileName(output_file)
                writer.Execute(slice_i)

            logging.info(f"Defaced DICOMs for patient {os.path.basename(row['input'])}")
        except Exception as e:
            logging.error(f"An exception occurred while defacing file {os.path.basename(row['input'])}: {e}")
            df.loc[idx, 'errors'] = str(e)

    # Filter rows where the segmentation directory contains the mask file and type is 'nii'
    df_nii = df[(df['nii_or_dcm'] == 'nii') & df['output_dir_segs'].apply(lambda x: mask_file in os.listdir(x))]

    for idx, row in df_nii.iterrows():
        try:
            nii = sitk.ReadImage(row['input'])
            nii_mask_path = os.path.join(row['output_dir_segs'], mask_file)
            nii_mask = sitk.ReadImage(nii_mask_path)
            
            mask = sitk.Resample(nii_mask, nii, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, nii_mask.GetPixelID())
            mask = sitk.Cast(mask, nii.GetPixelID())
            zero_mask = sitk.Cast(mask == 0, nii.GetPixelID())
            mask_value = np.min(sitk.GetArrayViewFromImage(nii))
            # masked_image = (mask * mask_value) + (zero_mask * nii)
            masked_image = sitk.Cast((mask * mask_value) + (zero_mask * nii), nii.GetPixelID())
            
            output_filename = os.path.join(row["output_dir"], "defaced.nii.gz")
            sitk.WriteImage(masked_image, output_filename)

            logging.info(f"Defaced NIfTI for patient {os.path.basename(row['input'])}")
        except Exception as e:
            logging.error(f"An exception occurred while defacing file {os.path.basename(row['input'])}: {e}")
            df.loc[idx, 'errors'] = str(e)


def download_test_data(input_directory):
    """
    Download test data for the TotalSegmentator.
    """
    manifest_file = "/scratch/riru10/projects/deface-project/manifest-1719584810202.tcia"
    df = nbia.downloadSeries(manifest_file, input_type="manifest", format="df")

    os.makedirs(input_directory, exist_ok=True)
    # Save the downloaded DICOM files to disk
    for _, row in df.iterrows():
        series_instance_uid = row['Series UID']
        series_data = nbia.downloadSeries(series_instance_uid)
        if isinstance(series_data, str):
            logging.error(f"Failed to download series {series_instance_uid}: {series_data}")
        else:
            logging.info(f"Downloaded series {series_instance_uid} to {input_directory}")
    print(f"Downloaded test data to {input_directory}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TotalSegmentator and deface images.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing DICOM or NIfTI files.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory where results will be saved.")
    parser.add_argument("--totalsegmentator_licence", type=str, help="Your Totalsegmentator licence key.")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory_base = args.output_dir


    if not input_directory:
        input_directory = os.path.join(os.getcwd(), "tciaDownload")
    # download_test_data(input_directory)

    if not output_directory_base:
        output_directory_base = os.path.join(os.getcwd(), "output")
        if not os.path.exists(output_directory_base):
            os.mkdir(output_directory_base)
    
    # Set the license for TotalSegmentator
    if args.totalsegmentator_licence:
        command = [f"totalseg_set_license -l {args.totalsegmentator_licence}"]
        subprocess.run(command, check=True, shell=True)

    # Define dataframe columns
    cols = ['input', 'output_dir', 'output_dir_segs', 'nii_or_dcm', 'number_files_in_outdir', 'face_included', 'errors']
    df = pd.DataFrame(columns=cols)

    # Get the available GPU IDs
    result = subprocess.run("nvidia-smi -L | wc -l", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    num_gpus = int(result.stdout.decode().strip())
    gpu_ids = list(range(num_gpus))

    if num_gpus < 1:
        print("No GPUs available.")

    # Run the processing function
    process_files(input_directory, output_directory_base, gpu_ids, df)
    
    # Print or save the dataframe
    print("DataFrame after running totalsegmentator") 
    print(df.head()) 

    deface(df)

    df.to_csv('segmentation_results.csv', index=False)
    print(f"saved df to: {os.getcwd()}/segmentation_results.csv")