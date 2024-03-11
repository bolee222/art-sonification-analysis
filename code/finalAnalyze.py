import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg 
from PIL import Image
#https://r02b.github.io/seaborn_palettes/


df = pd.read_csv('../dataset_cleanUp/final.csv')


def comparePose():

    df_new = pd.melt(df, id_vars=['nickname'], value_vars=['sod_a1', 'sod_a2'], var_name='sod_source', value_name='sod_value' )
    print(df_new.head(16))
    ax = plt.figure(figsize = (8,8))
    ax = sns.barplot(x="sod_source", y="sod_value", data=df_new)
    ax.set(ylim=(0, 10))

    plt.savefig('../plots/final_compare.png', bbox_inches='tight')



def elementGraph():
    df_new = pd.melt(df, id_vars=['nickname'], value_vars=["element_pose", "element_color", "element_light", "element_stroke", "element_style"], var_name='elements', value_name='value')
    print(df_new.head(4))
    ax = plt.figure(figsize = (8,4))

    ax = sns.boxplot(data=df_new, x="elements", y="value",  width=0.3, boxprops=dict(alpha=.2), color="grey")
    ax = sns.stripplot(x="elements", y="value", data=df_new, dodge=True, alpha=.25, zorder=1,color="black") 
    #ax = sns.pointplot(data=df_new, x="elements", y="value", dodge=.8 - .8 / 3, palette="dark", errorbar=None, markers="d")

    plt.savefig('../plots/final_element.png', bbox_inches='tight')


def detailGraph():
    df_new = pd.melt(df, id_vars=['nickname'], value_vars=["detail_poseSize", "detail_arms", "detail_inclination", "detail_effort", "detail_face", "detail_activity",
             "detail_number", "detail_location", "detail_humanColor", "detail_bgColor"], var_name='details', value_name='value')
    print(df_new.head(4))
    ax = plt.figure(figsize = (16,5))

    ax = sns.boxplot(data=df_new, x="details", y="value",  width=0.3, boxprops=dict(alpha=.2), color="grey")
    ax = sns.stripplot(x="details", y="value", data=df_new, dodge=True, alpha=.25, zorder=1,color="black") 
    #ax = sns.pointplot(data=df_new, x="elements", y="value", dodge=.8 - .8 / 3, palette="dark", errorbar=None, markers="d")

    plt.savefig('../plots/final_detail.png', bbox_inches='tight')


comparePose()
elementGraph()
detailGraph()