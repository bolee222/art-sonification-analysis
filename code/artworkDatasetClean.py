import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from poseDictionary import poseDic 


def transformCSV(filePath):
    #'../dataset/artworks/A11-Table 1.csv'
    artworkCode = filePath.split("/")[3].split("-")[0]
    print(artworkCode)

    df = pd.read_csv(filePath)

    c_name = df.columns[0] #nickname
    c_sod = df.columns[1] #sod
    c_gRow1 = df.columns[2] #grid_row1
    c_gRow2 = df.columns[3] #grid_row2
    c_gRow3 = df.columns[4] #grid_row3
    c_gRow4 = df.columns[5] #grid_row4
    c_reasons = df.columns[6] #grid_row4
    c_tempo = df.columns[7] #tempo
    c_pitch = df.columns[8] #pitch
    c_density = df.columns[9] #density
    c_others = df.columns[10] #others

    df.rename(columns = {c_name : "name", c_sod : "sod", c_gRow1 : "gRoq1", c_gRow2 : "gRoq2", c_gRow3 : "gRoq3", c_gRow4 : "gRoq4", c_reasons : "reasons", c_tempo : "tempo", c_pitch : "pitch", c_density : "density", c_others : "others"}, inplace = True)

    df["tempo_score"] = df["tempo"].str[0].astype(float)
    df["pitch_score"] = df["pitch"].str[0].astype(float)
    df["density_score"] = df["density"].str[0].astype(float)
    df["artwork"] = artworkCode

    df["poseType"] = df["artwork"].map(poseDic)

    df = df[["name", "artwork", "poseType", "sod","gRoq1", "gRoq2", "gRoq3", "gRoq4", "reasons", "tempo_score", "tempo", "pitch_score","pitch", "density_score", "density"]]

    print(df.columns)
    
    df.to_csv('../dataset_cleanUp/artworks/' + artworkCode + '.csv')

transformCSV('../dataset/artworks/A1-Table 1.csv')
transformCSV('../dataset/artworks/A2-Table 1.csv')


def cleanDataset():
    for i in range(0,21):
        number = i+1
        print(number)
        filePath = '../dataset/artworks/A' + str(number) + '-Table 1.csv'
        print(filePath)
        transformCSV(filePath)

def concatDataset():
    path = r'../dataset_cleanUp/artworks'          
    all_files = glob.glob(os.path.join(path, "*.csv"))     

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df_concat = pd.concat(df_from_each_file, ignore_index=True)
    df_concat.to_csv('../dataset_cleanUp/' + 'Artworks_concat.csv')
    

cleanDataset()
concatDataset()
