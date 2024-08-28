import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg 
from PIL import Image


df_post = pd.read_csv('../survey2_integrated/post.csv')
df_post = df_post[df_post['Criteria'].isin(['Performance', 'Enjoyment', 'Empathetic','Concentration'])] 


df_main = pd.read_csv('../survey2_integrated/survey.csv')
df_main = df_main[df_main['Actual Sound'].isin(['Sound A', 'Sound B', 'Sound C'])] 
print(df_main.head(4))


my_colors_2 = {0: 'grey', 1: '#049089'}

order_impact = ['Enjoyment', 'Empathetic','Concentration', 'Performance']
def impactGraph():
    #ax = plt.figure(figsize = (26,8))
    ax = plt.figure(figsize = (13,8))
    sns.set_theme(style="ticks")    

    ax = sns.boxplot(data=df_post, x="Criteria", y="Score", hue = "MusicState", width=0.5, linewidth = 1, order = order_impact, palette=my_colors_2, legend=False, boxprops=dict(alpha=.2),  showmeans=True, gap = 0.2)
    #, meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": "12"}


    #ax.set_yticks(np.arange(1, 5.5, 0.5), minor=True)
    #ax.grid(axis='y', linewidth=1, alpha=0.8)
    #ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/impacts.png', bbox_inches='tight', transparent=True)

def impactGraph_Catplot():
    #ax = plt.figure(figsize = (26,8))

    sns.set_theme(style="ticks")    

    ax = sns.catplot(data=df_post, x="Criteria", y="Score", hue = "MusicState", kind="point", palette=my_colors_2, order = order_impact, errorbar="ci",  capsize=.05, errwidth=1)

    plt.gcf().set_size_inches(14,8)

    ax.set(ylim=(1, 5))

    plt.savefig('../plots_survey2/impact_catplot.png', bbox_inches='tight', transparent=True)

my_colors_3 = {"Sound A": 'grey', "Sound B": '#f3ac64', "Sound C": '#65d4d9'}
def survey2Graph():
    ax = plt.figure(figsize = (13,8))
    sns.set_theme(style="ticks")    

    hue_order = ['Sound A', 'Sound B', 'Sound C']
    ax = sns.boxplot(data=df_main, x="Question Type", y="User Answer", hue = "Actual Sound", width=0.5,linewidth = 1, palette=my_colors_3, boxprops=dict(alpha=.5),  showfliers=False, legend=False, showmeans=True, hue_order=hue_order, gap = 0.2, meanprops={"marker": "^",
                       "markerfacecolor": "black", "markeredgecolor": "black",
                       "markersize": "6"})
    
    #ax.set_yticks(np.arange(1, 5, 0.5), minor=True)
    #ax.grid(axis='y', linewidth=1, alpha=0.8)
    #ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/all.png', bbox_inches='tight', transparent=True)


def survey2Graph_Static():
    ax = plt.figure(figsize = (25,8))
    sns.set_theme(style="ticks")    
    
    df_main_static = df_main[df_main['Artwork Number'].isin(['P7', 'P8', 'P9', 'P10', 'P11', 'P12'])] 
    print(df_main_static.head(10))

    hue_order = ['Sound A', 'Sound B', 'Sound C']
    ax = sns.boxplot(data=df_main_static, x="Question Type", y="User Answer", hue = "Actual Sound", width=0.5,linewidth = 1, palette=my_colors_3, boxprops=dict(alpha=.2),  showfliers=False, showmeans=True, hue_order=hue_order, gap = 0.3, meanprops={"marker": "^",
                       "markerfacecolor": "black", "markeredgecolor": "black",
                       "markersize": "10"})

    ax.set_yticks(np.arange(1, 5, 0.5), minor=True)
    ax.grid(axis='y', linewidth=1, alpha=0.8)
    ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/all(statics).png', bbox_inches='tight', transparent=True)


my_colors_3b = {"Sound A": 'grey', "Sound B": '#dc8302', "Sound C": '#049089'}
def survey2Graph_Catplot():
    ax = plt.figure(figsize = (25,8))
    sns.set_theme(style="ticks")    

    hue_order = ['Sound A', 'Sound B', 'Sound C']
    ax = sns.catplot(data=df_main, x="Question Type", y="User Answer", hue = "Actual Sound", kind="point", palette=my_colors_3b, errorbar="ci",  capsize=.05, errwidth=1)

    plt.gcf().set_size_inches(14,8)

    ax.set(ylim=(1, 5))
    #ax.grid(axis='y', linewidth=1, alpha=0.8)
    #ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/catplot.png', bbox_inches='tight', transparent=True)


def dynamicChecker(input):
    if input == "P1":
        return 1
    if input == "P2":
        return 1
    if input == "P3":
        return 1
    if input == "P4":
        return 1
    if input == "P5":
        return 1
    if input == "P6":
        return 1
    else:
        return 0

my_colors_2_dynamic = {0: '#6cb2f7', 1: '#ff83ac'}
def Static_Dynamic(criteria):
    ax = plt.figure(figsize = (25,8))
    sns.set_theme(style="ticks")    

    
    df_main["DynamicStatus"] = df_main['Artwork Number'].map(dynamicChecker)
    print (df_main.head(10))


    df_here = df_main[df_main['Question Type'].isin([criteria])] 

    hue_order = ['Sound A', 'Sound B', 'Sound C']
    ax = sns.catplot(data=df_here, x="Actual Sound", y="User Answer", hue = "DynamicStatus", kind="point", errorbar="ci",  capsize=.05, errwidth=2, order=hue_order, legend=False, palette=my_colors_2_dynamic, linewidth=3.5)

    plt.gcf().set_size_inches(12,8)

    ax.set(ylim=(1, 5))
    #ax.grid(axis='y', linewidth=1, alpha=0.8)
    #ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/statid-dynamic_' + criteria +'.png', bbox_inches='tight', transparent=True)




#impactGraph()
#impactGraph_Catplot()
#survey2Graph()
#survey2Graph_Catplot()


Static_Dynamic('Harmony')
Static_Dynamic('Dynamism')
Static_Dynamic('Quality')