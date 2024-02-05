import pandas as pd

df = pd.read_csv('../dataset/Results-A1.csv')

g_o1 = df.columns[2]
g_o2 = df.columns[3]
g_o3 = df.columns[4]
g_o4 = df.columns[5]

df = df[[g_o1, g_o2, g_o3, g_o4]]

#Column이름 치환
df.rename(columns = {g_o1 : "r1", g_o2 : "r2", g_o3 : "r3", g_o4 : "r4"}, inplace = True)

print(df)




