import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from poseDictionary import poseDic 

def scaleToNum(value): 
    output = 0   
    if (value ==  "매우 큰 영향(very huge)"):
        output = 5
    elif (value == "큰 영향 (huge)"):
        output = 4
    elif (value == "보통 (moderate)"):
        output = 3
    elif (value == "작은 영향 (minor)"):
        output = 2
    elif (value == "매우 작은 영향(very minor)"):
        output = 1
    else:
        output = 0
    return output 



def transformCSV():
    filePath = '../dataset/Final-Table 1.csv'
    df = pd.read_csv(filePath)
    df.replace(to_replace= "매우 큰 영향(very huge)", value=5, inplace=True)
    df.replace(to_replace= "큰 영향 (huge)", value=4, inplace=True)
    df.replace(to_replace= "보통 (moderate)", value=3, inplace=True)
    df.replace(to_replace= "작은 영향 (minor)", value=2, inplace=True)
    df.replace(to_replace= "매우 작은 영향(very minor)", value=1, inplace=True)

    nickname = df.columns[0]
    
    element_pose = df.columns[1]
    element_color = df.columns[2]
    element_light = df.columns[3]
    element_stroke = df.columns[4]
    element_style = df.columns[5]
    elemet_add = df.columns[6]

    detail_poseSize = df.columns[7]
    detail_arms = df.columns[8]
    detail_inclination = df.columns[9]
    detail_effort = df.columns[10]
    detail_face = df.columns[11]
    detail_activity = df.columns[12]
    detail_number = df.columns[13]
    detail_location = df.columns[14]
    detail_humanColor = df.columns[15]
    detail_bgColor = df.columns[16]

    sod_a1 = df.columns[17] #nickname
    sod_a2 = df.columns[18] #sod

    df.rename(columns = {nickname : "nickname", sod_a1 : "sod_a1", sod_a2 : "sod_a2",
                         element_pose : "element_pose", element_color : "element_color",
                         element_light : "element_light", element_stroke : "element_stroke",
                         element_style : "element_style", elemet_add : "elemet_add",
                         
                         detail_poseSize : "detail_poseSize", detail_arms : "detail_arms",
                         detail_inclination : "detail_inclination", detail_effort : "detail_effort",
                         detail_face : "detail_face", detail_activity : "detail_activity",
                         detail_number : "detail_number", detail_location : "detail_location",
                         detail_humanColor : "detail_humanColor", detail_bgColor : "detail_bgColor"}, inplace = True)

    df = df[["nickname", "sod_a1", "sod_a2", "element_pose", "element_color", "element_light", "element_stroke", "element_style", "elemet_add",
             "detail_poseSize", "detail_arms", "detail_inclination", "detail_effort", "detail_face", "detail_activity",
             "detail_number", "detail_location", "detail_humanColor", "detail_bgColor"]]
    

    print(df.head())
    df.to_csv('../dataset_cleanUp/final_new.csv')


transformCSV()