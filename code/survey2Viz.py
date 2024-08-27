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

def impactGraph():
    ax = plt.figure(figsize = (16,8))
    sns.set_theme(style="ticks")    

    ax = sns.boxplot(data=df_post, x="Criteria", y="Score", hue = "MusicState", width=0.3,boxprops=dict(alpha=.2),  showmeans=True, gap = 0.1, meanprops={"marker": "^",
                       "markerfacecolor": "#DE3A0D", "markeredgecolor": "#DE3A0D",
                       "markersize": "8"})

    ax.set_yticks(np.arange(1, 5, 0.5), minor=True)
    ax.grid(axis='y', linewidth=0.25, alpha=0.8)
    ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/impacts.png', bbox_inches='tight', transparent=True)



def survey2Graph():
    ax = plt.figure(figsize = (16,6))
    sns.set_theme(style="ticks")    

    hue_order = ['Sound A', 'Sound B', 'Sound C']
    ax = sns.boxplot(data=df_main, x="Question Type", y="User Answer", hue = "Actual Sound", width=0.3,boxprops=dict(alpha=.2),  showfliers=False, showmeans=True, hue_order=hue_order, gap = 0.1, meanprops={"marker": "^",
                       "markerfacecolor": "#DE3A0D", "markeredgecolor": "#DE3A0D",
                       "markersize": "8"})

    ax.set_yticks(np.arange(1, 5, 0.5), minor=True)
    ax.grid(axis='y', linewidth=0.25, alpha=0.8)
    ax.grid(which='minor', alpha=0.5)

    plt.savefig('../plots_survey2/all.png', bbox_inches='tight', transparent=True)



#impactGraph()
survey2Graph()