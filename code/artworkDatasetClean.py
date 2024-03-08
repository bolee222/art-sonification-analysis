import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from poseDictionary import poseDic 

def gridRowtoNum(row):
    row_num = [0,0,0,0]
    
    if (type(row) == str):
        if "[열A] colA" in row:
            row_num[0] = 1
        if "[열B] colB" in row:
            row_num[1] = 1
        if "[열C] colC" in row:
            row_num[2] = 1
        if "[열D] colD" in row:
            row_num[3] = 1 
    else:
        row_num = [0,0,0,0]
    return row_num 


def heatmapAnalysis(df):
    df["gRoq1_mat"]=None
    df["gRoq2_mat"]=None
    df["gRoq3_mat"]=None
    df["gRoq4_mat"]=None

    for rowNum in range(len(df)):
        gRoq1_ = df.iloc[rowNum]['gRoq1']
        gRoq1_mat = gridRowtoNum(gRoq1_)
        df.at[rowNum,'gRoq1_mat'] = gRoq1_mat

        gRoq2_ = df.iloc[rowNum]['gRoq2']
        gRoq2_mat = gridRowtoNum(gRoq2_)
        df.at[rowNum,'gRoq2_mat'] = gRoq2_mat

        gRoq3_ = df.iloc[rowNum]['gRoq3']
        gRoq3_mat = gridRowtoNum(gRoq3_)
        df.at[rowNum,'gRoq3_mat'] = gRoq3_mat

        gRoq4_ = df.iloc[rowNum]['gRoq4']
        gRoq4_mat = gridRowtoNum(gRoq4_)
        df.at[rowNum,'gRoq4_mat'] = gRoq4_mat
    return df


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

    heatmapAnalysis(df)

    df = df[["name", "artwork", "poseType", "sod", "gRoq1_mat", "gRoq2_mat", "gRoq3_mat", "gRoq4_mat", "reasons", "tempo_score", "tempo", "pitch_score","pitch", "density_score", "density"]]

    #print(df.columns)
    
    df.to_csv('../dataset_cleanUp/artworks/' + artworkCode + '.csv')

#transformCSV('../dataset/artworks/A1-Table 1.csv')
#transformCSV('../dataset/artworks/A2-Table 1.csv')


def cleanDataset():
    for i in range(0,21):
        number = i+1
        #print(number)
        filePath = '../dataset/artworks/A' + str(number) + '-Table 1.csv'
        #print(filePath)
        transformCSV(filePath)



def concatDataset():
    path = r'../dataset_cleanUp/artworks'          
    all_files = glob.glob(os.path.join(path, "*.csv"))     

    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df_concat = pd.concat(df_from_each_file, ignore_index=True)
    df_concat.to_csv('../dataset_cleanUp/' + 'Artworks_concat.csv')
    print("Concat Dataset Generated")
    

cleanDataset()
concatDataset()
