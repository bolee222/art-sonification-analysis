import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg 
from PIL import Image
#https://r02b.github.io/seaborn_palettes/

def sod_graph():
    mean = df.groupby('artwork').sod.mean().reset_index()
    mean['sort'] = mean['artwork'].str.extract('(\d+)', expand=False).astype(int)
    mean.sort_values('sort',inplace=True, ascending=True)
    mean = mean.drop('sort', axis=1)
    mean = mean['sod']

    std = df.groupby('artwork').sod.std() / np.sqrt(df.groupby('artwork').sod.count())

    ax = plt.figure(figsize = (18,7))
    ax = sns.violinplot(x="artwork", y="sod", data=df, hue="poseType", linewidthfloat=0.01, palette="RdYlGn", dodge=False, inner=None)

    violins = [c for i, c in enumerate(ax.collections)]  
    [v.set_edgecolor("white") for v in violins]  # 전체 violin edgecolor 변경
    [v.set_linewidth(0.01) for v in violins]  # 전체 violin edgecolor 변경
    [v.set_alpha(0.3) for v in violins]
    violins[3].set_edgecolor("k")        # Sunday violin edgecolor 변경

    ax = sns.swarmplot (x="artwork", y="sod", data=df, dodge=False, hue="poseType", alpha=.5)
    plt.scatter(x=range(len(mean)),y=mean,c="k")
    plt.errorbar(range(len(mean)), mean, yerr=std, ls='none', c="k")

    ax.set_xlabel("Artworks")
    ax.set_ylabel("Sense of Dynamic Score")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig('../plots/sod_graph.png', bbox_inches='tight')

def tempo_graph():
    df_new = df[['sod', 'tempo_score']].value_counts().reset_index(name='count')
    print(df_new.head(4))
    df_ = df[df["artwork"] == 'A14'] 

    ax = plt.figure(figsize = (4,4))
    #ax = sns.relplot(data=df_new, x="sod", y="tempo_score",  size = 'count', hue="count")
    sns.set_theme(style="ticks")
    ax = sns.jointplot(x="sod", y="tempo_score", data=df, kind="hist", alpha=0.0, cmap="Blues")
    ax = sns.kdeplot(x="sod", y="tempo_score", data=df, fill=True, alpha=.7 ,cmap="Blues")

    #ax.set_xlabel("Tempo Level")
    #ax.set_ylabel("Sense of Dynamic Score")
    plt.savefig('../plots/corr_tempo.png', bbox_inches='tight')

def pitch_graph():
    df_new = df[['sod', 'pitch_score']].value_counts().reset_index(name='count')
    print(df_new.head(4))
    ax = plt.figure(figsize = (4,4))
    #ax = sns.relplot(data=df_new, x="sod", y="pitch_score",  size = 'count', hue="count")
    ax = sns.jointplot(x="sod", y="pitch_score", data=df, kind="hist", alpha=0.0, cmap="Blues")
    ax = sns.kdeplot(x="sod", y="pitch_score", data=df, fill=True, alpha=.7 ,cmap="Blues")
    
    #ax.set_xlabel("Pitch Level")
    #ax.set_ylabel("Sense of Dynamic Score")
    plt.savefig('../plots/corr_pitch.png', bbox_inches='tight')

def density_graph():
    df_new = df[['sod', 'density_score']].value_counts().reset_index(name='count')
    print(df_new.head(4))
    ax = plt.figure(figsize = (4,4))
    #ax = sns.relplot(data=df_new, x="sod", y="density_score",  size = 'count', hue="count")
    ax = sns.jointplot(x="sod", y="density_score", data=df, kind="hist", alpha=0.0, cmap="Blues")
    ax = sns.kdeplot(x="sod", y="density_score", data=df, fill=True, alpha=.7 ,cmap="Blues")

    #ax.set_xlabel("Density Level")
    #ax.set_ylabel("Sense of Dynamic Score")
    plt.savefig('../plots/corr_density.png', bbox_inches='tight')

def parsing(grid):
    grid = grid.replace('[','').replace(']','').split(',')
    grid = [int(x) for x in grid]
    return grid

def grid_heatmap(target_artwork):
    bg = Image.open('../paintings/' + target_artwork + '.jpg').convert('L')
    #bg = mpimg.imread('../paintings/' + target_artwork + '.jpg') 
    df_art = df[df['artwork'] == target_artwork]

    df["grids"]=""
    grid_sum = None

    for i in range(len(df_art)):
        grid_0 = df_art.iloc[i]['gRoq1_mat']
        grid_1 = df_art.iloc[i]['gRoq2_mat']
        grid_2 = df_art.iloc[i]['gRoq3_mat']
        grid_3 = df_art.iloc[i]['gRoq4_mat']

        grids = [parsing(x) for x in [grid_0, grid_1, grid_2, grid_3]]

        if i == 0:
            grid_sum = np.asarray(grids)
        else:
            grid_sum += np.asarray(grids)
    
    ax = plt.figure(figsize = (10,10))
    ax = sns.heatmap(grid_sum, annot=True, cmap="Oranges", cbar=False, yticklabels=False, xticklabels=False)
    ax.tick_params(left=False, bottom=False) ## other options are right and top


    ax.imshow(bg, alpha=0.7, cmap='gray',
          aspect = ax.get_aspect(),
          extent = ax.get_xlim() + ax.get_ylim(),
          zorder = 1) #put the map under the heatmap

    ax = sns.heatmap(grid_sum, annot=True, cmap="Oranges", cbar=False, yticklabels=False, xticklabels=False, alpha=.2)

    plt.savefig('../plots/heatmap_' + target_artwork + '.png', bbox_inches='tight')
    print(target_artwork + " Heatmap Created!")
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv('../dataset_integrated/Artworks_concat.csv')

    df['sort'] = df['artwork'].str.extract('(\d+)', expand=False).astype(int)
    df.sort_values('sort',inplace=True, ascending=True)
    df = df.drop('sort', axis=1)
        
    sod_graph()

    tempo_graph()
    pitch_graph()
    density_graph()

    #run for each artwork 
    for i in range(21):
        grid_heatmap("A" + str(i+1))    







