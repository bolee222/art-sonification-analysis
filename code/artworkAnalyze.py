import pandas as pd
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg 
from PIL import Image
from matplotlib.colors import ListedColormap


#https://r02b.github.io/seaborn_palettes/

colorSets = ['#DD462E','#F07D09', '#B028F4', '#F42863','#4B79C4', '#37AD3B', '#205E22']
colorSets_21 = ['#DD462E', '#DD462E', '#DD462E',
                '#F07D09', '#F07D09', '#F07D09', 
                '#B028F4', '#B028F4', '#B028F4', 
                '#F42863', '#F42863', '#F42863',
                '#4B79C4', '#4B79C4',  '#4B79C4', 
                '#37AD3B', '#37AD3B', '#37AD3B', 
                '#205E22', '#205E22', '#205E22']

def sod_graph():
    mean = df.groupby('artwork').sod.mean().reset_index()
    mean['sort'] = mean['artwork'].str.extract('(\d+)', expand=False).astype(int)
    mean.sort_values('sort',inplace=True, ascending=True)
    mean = mean.drop('sort', axis=1)
    mean = mean['sod']

    std = df.groupby('artwork').sod.std() / np.sqrt(df.groupby('artwork').sod.count())
    
    ax = plt.figure(figsize = (18,7))

    sns.set_palette(sns.color_palette(colorSets))


    #VIOLIN PLOT---------------------------
    ax = sns.violinplot(x="artwork", y="sod", data=df, hue="poseType", dodge=False, inner=None, legend=False)
    violins = [c for i, c in enumerate(ax.collections)]  
    [v.set_edgecolor("white") for v in violins]  # 전체 violin edgecolor 변경
    [v.set_linewidth(0.01) for v in violins]  # 전체 violin edgecolor 변경 
    [v.set_alpha(0.15) for v in violins]
    violins[3].set_edgecolor("k")        # Sunday violin edgecolor 변경

    #SWARMPLOT ---------------------------
    #ax = sns.swarmplot (x="artwork", y="sod", data=df, dodge=False, hue="poseType", alpha=.5)
    #plt.scatter(x=range(len(mean)),y=mean,c="k")
    #plt.errorbar(range(len(mean)), mean, yerr=std, ls='none', c="k")

    #GRIDS ---------------------------
    ax.set_yticks(np.arange(1, 10, 2), minor=True)
    ax.grid(axis='y', linewidth=0.25, alpha=0.8)
    ax.grid(which='minor', alpha=0.3)

    #BOXPLOT ---------------------------
    ax = sns.boxplot(data=df, x="artwork", y="sod", hue="poseType", width=0.6, linecolor="#137", linewidth=1.00, fill=False, showmeans=True, gap = 0.1, meanprops={"marker": "^",
                       "markerfacecolor": "black", "markeredgecolor": "black",
                       "markersize": "8"})
    #for patch in ax.artists:
    #    fc = patch.get_facecolor()
    #    patch.set_facecolor(mpl.colors.to_rgba(fc, 0.3))


    ax.set_xlabel("Artworks")
    ax.set_ylim([-0.5, 10.5])
    ax.set_ylabel("Sense of Dynamic Score")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig('../plots/sod_graph.png', bbox_inches='tight')

palette_colorTwo = sns.color_palette(['#F77855','#89DEF5'])
#palette_colorTwo_r = sns.color_palette(['#4B79C4', '#DD462E'])

def tempo_graph():
    df_new = df[['sod', 'tempo_score']].value_counts().reset_index(name='count')
    print(df_new.head(4))
    df_ = df[df["artwork"] == 'A14'] 

    ax = plt.figure(figsize = (4,4))
    sns.set_theme(style="ticks")
    ax = sns.jointplot(x="sod", y="tempo_score", data=df, kind="hist", alpha=0.0, hue="activityType", palette=palette_colorTwo, legend = False)
    #ax = sns.lmplot(x="sod", y="tempo_score", data=df,  hue="activityType", palette=palette_colorTwo, scatter=False, legend=False)
    #ax.set(ylim=(-2, 12))   
    #ax.set(xlim=(-2, 12))   
    ax = sns.kdeplot(x="sod", y="tempo_score", data=df, fill=True, alpha=.8, levels=8, hue="activityType", bw_adjust=1.8) 
    ax.set_xlim([-2, 12])
    ax.set_ylim([-2, 12])
    #ax.set_ylabel("Sense of Dynamic Score")
    plt.savefig('../plots/corr_tempo.png', bbox_inches='tight', transparent=True)

    df_static = df.loc[df['activityType'] != "static"]
    df_dynamic = df.loc[df['activityType'] != "dynamic"]
    print("RSquared / Tempo / Static:  " + str(df_static['sod'].corr(df_static['tempo_score'], method='pearson')))
    print("RSquared / Tempo / Dynamic:  " + str(df_dynamic['sod'].corr(df_dynamic['tempo_score'], method='pearson')))


def pitch_graph():
    df_new = df[['sod', 'pitch_score']].value_counts().reset_index(name='count')
    print(df_new.head(4))
    ax = plt.figure(figsize = (4,4))
    
    ax = sns.jointplot(x="sod", y="pitch_score", data=df, kind="hist",  alpha=0.0, hue="activityType", palette=palette_colorTwo, legend = False)
    #ax = sns.lmplot(x="sod", y="pitch_score", data=df,  hue="activityType", palette=palette_colorTwo, scatter=False, legend=False)
    #ax.set(ylim=(-2, 12))   
    #ax.set(xlim=(-2, 12))   
    ax = sns.kdeplot(x="sod", y="pitch_score", data=df, fill=True, alpha=.8,  hue="activityType")
    
    ax.set_xlim([-2, 12])
    ax.set_ylim([-2, 12])
    #ax.set_xlabel("Pitch Level")
    #ax.set_ylabel("Sense of Dynamic Score")
    plt.savefig('../plots/corr_pitch.png', bbox_inches='tight', transparent=True)

    df_static = df.loc[df['activityType'] != "static"]
    df_dynamic = df.loc[df['activityType'] != "dynamic"]
    print("RSquared / Pitch / Static:  " + str(df_static['sod'].corr(df_static['pitch_score'], method='pearson')))
    print("RSquared / Pitch / Dynamic:  " + str(df_dynamic['sod'].corr(df_dynamic['pitch_score'], method='pearson')))

def density_graph():
    df_new = df[['sod', 'density_score']].value_counts().reset_index(name='count')
    print(df_new.head(4))
    ax = plt.figure(figsize = (4,4))
    
    ax = sns.jointplot(x="sod", y="density_score", data=df, kind="hist", alpha=0.0, hue="activityType", palette=palette_colorTwo, legend = False)
    #ax = sns.lmplot(x="sod", y="density_score", data=df,  hue="activityType", palette=palette_colorTwo, scatter=False, legend=False)
    #ax.set(ylim=(-2, 12))   
    #ax.set(xlim=(-2, 12)) 
    ax = sns.kdeplot(x="sod", y="density_score", data=df, fill=True, alpha=.8,  levels=8, hue="activityType", bw_adjust=1.8)

    ax.set_xlim([-2, 12])
    ax.set_ylim([-2, 12])
    #ax.set_xlabel("Density Level")
    #ax.set_ylabel("Sense of Dynamic Score")
    plt.savefig('../plots/corr_density.png', bbox_inches='tight', transparent=True)

    df_static = df.loc[df['activityType'] != "static"]
    df_dynamic = df.loc[df['activityType'] != "dynamic"]
    print("RSquared / Density / Static:  " + str(df_static['sod'].corr(df_static['density_score'], method='pearson')))
    print("RSquared / Density / Dynamic:  " + str(df_dynamic['sod'].corr(df_dynamic['density_score'], method='pearson')))

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

    df_yon = pd.read_csv('../dataset_cleanUp/Artworks_concat.csv')
    df_yon['sort'] = df_yon['artwork'].str.extract('(\d+)', expand=False).astype(int)
    df_yon.sort_values('sort',inplace=True, ascending=True)
    df_yon = df_yon.drop('sort', axis=1)


    df = pd.read_csv('../dataset_integrated/Artworks_concat.csv')
    df['sort'] = df['artwork'].str.extract('(\d+)', expand=False).astype(int)
    df.sort_values('sort',inplace=True, ascending=True)
    df = df.drop('sort', axis=1)


        
    #sod_graph()

    #tempo_graph()
    #pitch_graph()
    #density_graph()

    #run for each artwork 
    for i in range(21):
        grid_heatmap("A" + str(i+1))    







