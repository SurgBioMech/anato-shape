# -*- coding: utf-8 -*-
"""
Utility functions

@author: joepugar
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import trimesh
from IPython.display import HTML, display

def progress(value, max=100):
    """A simple progress bar."""
    return HTML(f"""
        <progress value='{value}' max='{max}' style='width: 100%'>{value}</progress>
    """)
    
def GetFilteredMeshPaths(parent_folder, filter_strings, file_filter_strings, ext=".mat"):
    """Generalized function to get a list of mesh file paths (supports .mat and .parquet formats)."""
    filtered_paths = []
    for filter_string in filter_strings:
        specific_folder = Path(parent_folder) / filter_string
        if specific_folder.exists():
            for root, dirs, _ in os.walk(specific_folder):
                for dir_name in dirs:
                    if dir_name == 'mesh':
                        mesh_folder = Path(root) / dir_name
                        for filename in mesh_folder.iterdir():
                            if filename.suffix == ext and any(fil in filename.name for fil in file_filter_strings):
                                filtered_paths.append((str(mesh_folder), filename.name))
                                print(f"Adding {filename.name}")
    return filtered_paths

def GetMeshFromParquet(scan_name):
    """Unpack the mesh data from a parquet file."""
    parquet_data = pd.read_parquet(scan_name)
    v = parquet_data.iloc[:, :3].dropna().values
    f = parquet_data.iloc[:, 3:6].dropna().values
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    pcs = parquet_data.iloc[:, 6:8].dropna().values
    mesh.curvatures = pcs
    return mesh

def SaveResultsToXLSX(directory, file_name, data):
    """Save a pandas dataframe as an .xlsx file."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(Path(directory) / file_name) as writer:
        data.to_excel(writer, index=False)
        
def SaveScanDictToXLSX(directory, file_name, scan_dict):
    """Save a dictionary as an .xlsx file."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(Path(directory) / file_name) as writer:
        for main_key, sub_dict in scan_dict.items():
            data = []
            for key, values, in sub_dict.items():
                df = pd.DataFrame(values)
                df['key'] = key
                data.append(df)
            output_df = pd.concat(data, ignore_index=True)
            output_df.to_excel(writer, sheet_name=str(main_key), index=False)
        
def file_name_not_ext(file_name):
    """Remove file extensions from names."""
    return Path(file_name).stem

def SplitUpScanName(df):
    """Split the file name into three columns: Patient_ID, Scan_Number, and Mesh_Density."""
    df_split = df['Scan_Name'].str.split('_', expand=True)
    df_split.columns = ['Patient_ID', 'Scan_Number', 'Mesh_Density']
    df['Scan_ID'] = df_split['Patient_ID'] + '_' + df_split['Scan_Number']
    return pd.concat([df, df_split], axis=1)

def Normalize(data, xaxis, yaxis, string):
    """Normalize all of the data to the mean of the string group."""
    fdf = data[data['Scan_ID'].str.contains(string)]
    xn, yn = fdf[xaxis].mean(), fdf[yaxis].mean()
    data[f'{xaxis}_Norm'] = data[xaxis] / xn
    data[f'{yaxis}_Norm'] = data[yaxis] / yn
    data['Mean_Radius'] = 1 / data[xaxis]
    return data, xn, yn

def GroupsNormalize(data, xaxis, yaxis, string):
    """Normalize each unique scale space group individually and concatenate results."""
    ndata = pd.DataFrame()
    for density in data['Mesh_Density'].unique():
        for prefactor in data['Partition_Prefactor'].unique():
            group = data[(data['Mesh_Density'] == density) & (data['Partition_Prefactor'] == prefactor)]
            normalized_group, _, _ = Normalize(group, xaxis, yaxis, string)
            ndata = pd.concat([ndata, normalized_group], axis=0)
    return ndata

def MergeMetaData(directory, file_name, cohort_list, results, cat_columns):
    """Integrate all available meta data into the results dataframe."""
    meta_data = pd.concat([pd.read_excel(Path(directory) / file_name, sheet_name=sheet) for sheet in cohort_list], ignore_index=True)
    results_df = results.merge(meta_data, on='Scan_ID', how='left')
    results_df[cat_columns] = results_df[cat_columns].astype(str)
    return results_df