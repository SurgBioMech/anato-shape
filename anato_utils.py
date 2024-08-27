# -*- coding: utf-8 -*-
"""
Utility functions

@author: joepugar
"""

import os
import numpy as np
import pandas as pd
from IPython.display import HTML, display


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))
    
def GetFilteredMatMeshPaths(parent_folder, filter_strings, file_filter_strings):
    filtered_paths = []
    for filter_string in filter_strings:
        specific_folder = os.path.join(parent_folder, filter_string)
        if os.path.exists(specific_folder):
            for root, dirs, files in os.walk(specific_folder):
                for dir_name in dirs:
                    if dir_name == 'mesh':
                        mesh_folder = os.path.join(root, dir_name)
                        for filename in os.listdir(mesh_folder):
                            if filename.endswith('.mat') and any(file_filter in filename for file_filter in file_filter_strings):
                                filtered_paths.append((mesh_folder, filename))
                                print(f'''Adding {filename}''')
    return filtered_paths

def GetFilteredParqMeshPaths(parent_folder, filter_strings, file_filter_strings):
    filtered_paths = []
    for filter_string in filter_strings:
        specific_folder = os.path.join(parent_folder, filter_string)
        if os.path.exists(specific_folder):
            for root, dirs, files in os.walk(specific_folder):
                for dir_name in dirs:
                    if dir_name == 'mesh':
                        mesh_folder = os.path.join(root, dir_name)
                        for filename in os.listdir(mesh_folder):
                            if filename.endswith('.parquet') and any(file_filter in filename for file_filter in file_filter_strings):
                                filtered_paths.append((mesh_folder, filename))
                                print(f'''Adding {filename}''')
    return filtered_paths

def GetMeshFromParquet(scan_name):
    parquet_data = pd.read_parquet(scan_name)
    svs = parquet_data.iloc[:,:3].dropna().values
    sts = parquet_data.iloc[:,3:6].dropna().values
    pcs = parquet_data.iloc[:,6:8].dropna().values
    mesh_features = parquet_data.iloc[:,8:].dropna()
    return svs, sts, pcs, mesh_features

def SaveToXLSX(directory, file_name, data):
    os.chdir(directory)
    with pd.ExcelWriter(file_name) as writer:
        data.to_excel(writer, index=False)
        
def file_name_not_ext(file_name):
    parts = file_name.rsplit(".", 1)
    if len(parts) == 1:
        return file_name
    return parts[0]

def SplitUpScanName(df):
    df['Scan_ID'] = df['Scan_Name'].str.split('_', expand=True, n=2).iloc[:,:2].apply('_'.join, axis=1)
    df_split = df['Scan_Name'].str.split('_', expand=True)
    df_split.columns = ['Patient_ID', 'Scan_Number', 'Mesh_Density']
    df = pd.concat([df, df_split], axis=1)
    return df

def Normalize(data, xaxis, yaxis, string):
    fdf = data[data['Scan_ID'].str.contains(string)]
    xn = np.mean(fdf[xaxis])
    yn = np.mean(fdf[yaxis])
    data[xaxis+'_Norm'] = data[xaxis]/xn
    data[yaxis+'_Norm'] = data[yaxis]/yn
    data['Mean_Radius'] = 1/data[xaxis]
    return data, xn, yn 

def GroupsNormalize(data, xaxis, yaxis, string):
    ndata = pd.DataFrame()
    uv1 = list(data['Mesh_Density'].unique())
    uv2 = list(data['Partition_Prefactor'].unique())
    for i in range(0, len(uv1)):
        for j in range(0, len(uv2)):
            c1 = data['Mesh_Density'] == uv1[i]
            c2 = data['Partition_Prefactor'] == uv2[j]
            gp = data[c1 & c2]
            gpn, xn, yn = Normalize(gp, xaxis, yaxis, string)
            ndata = pd.concat([ndata, gpn], axis=0)
    return ndata

def MergeMetaData(directory, file_name, cohort_list, results, cat_columns):
    os.chdir(directory)
    meta_data = pd.DataFrame()
    for c in cohort_list:
        data = pd.read_excel(file_name, sheet_name=c)
        meta_data = pd.concat([meta_data, data], ignore_index=True)
    results_df = pd.merge(results, meta_data, on='Scan_ID')
    for cat in cat_columns:
        results_df[cat]=results_df[[cat]].astype(str)
    return results_df




