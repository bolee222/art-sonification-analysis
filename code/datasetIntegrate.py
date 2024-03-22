import pandas as pd
import glob
import os

def integrateArtworks():
    for i in range(0,21):
        number = i+1
        filePath_forms = '../dataset_cleanUp/artworks/A' + str(number) + '.csv'
        filePath_prolific = '../dataset_prolific/artworks/A' + str(number) + '.csv'
        filePath_integrate = '../dataset_integrated/artworks/A' + str(number) + '.csv'

        df = pd.concat(map(pd.read_csv, [filePath_forms, filePath_prolific]), ignore_index=True) 
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        df.to_csv(filePath_integrate)



def concatDataset():
    path = r'../dataset_integrated/artworks'          
    all_files = glob.glob(os.path.join(path, "*.csv"))     
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df_concat = pd.concat(df_from_each_file, ignore_index=True)
    df_concat.to_csv('../dataset_integrated/' + 'Artworks_concat.csv')
    print("Concat Dataset Generated")

integrateArtworks()
concatDataset()