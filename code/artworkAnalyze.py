import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 

df = pd.read_csv('../dataset_cleanUp/Artworks_concat.csv')

df['sort'] = df['artwork'].str.extract('(\d+)', expand=False).astype(int)
df.sort_values('sort',inplace=True, ascending=True)
df = df.drop('sort', axis=1)

print(df.head(4))

mean = df.groupby('artwork').sod.mean().reset_index()
mean['sort'] = mean['artwork'].str.extract('(\d+)', expand=False).astype(int)
mean.sort_values('sort',inplace=True, ascending=True)
mean = mean.drop('sort', axis=1)
mean = mean['sod']

std = df.groupby('artwork').sod.std() / np.sqrt(df.groupby('artwork').sod.count())

ax = plt.figure(figsize = (18,7))
ax = sns.violinplot(x="artwork", y="sod", data=df, hue="poseType", linewidthfloat=0.01, palette='pastel', dodge=False, inner=None)

violins = [c for i, c in enumerate(ax.collections)]  
[v.set_edgecolor("white") for v in violins]  # 전체 violin edgecolor 변경
[v.set_linewidth(0.01) for v in violins]  # 전체 violin edgecolor 변경
violins[3].set_edgecolor("k")        # Sunday violin edgecolor 변경



ax = sns.swarmplot (x="artwork", y="sod", data=df, dodge=False, hue="poseType")
plt.scatter(x=range(len(mean)),y=mean,c="k")
plt.errorbar(range(len(mean)), mean, yerr=std, ls='none', c="k")

ax.set_xlabel("Artworks")
ax.set_ylabel("Sense of Dynamic Score")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


plt.show()