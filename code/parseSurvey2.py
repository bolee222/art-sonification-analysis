import pandas as pd
import re

if __name__ == '__main__':
    df_user = pd.read_csv('./dataset_survey2/datasheet.csv')
    df_ans = pd.read_csv('./dataset_survey2/survey_answer.csv')
    df_ans['_'] = ['' for _ in range(df_ans.shape[0])]
    df_ans = df_ans.values.reshape(-1)

    q_order = [2, 8, 4, 10, 5, 6, 11, 3, 1, 12, 7, 9]

    df_pre_survey = df_user.iloc[:, :16]
    df_survey = df_user.iloc[:,16:208]
    df_post_survey = df_user.iloc[:,208:]

    n,m = df_survey.shape
    y = []
    sound_types = ['A', 'B', 'C', 'D']

    for i in range(1,n):
        res_id = int( df_pre_survey.loc[i, 'Your Prolific ID'] )
            
        for j in range(0, m, 16): # q_order
            n_art = q_order[j // 16]
            art_dyn = df_survey.iloc[i, j]
            if art_dyn == 'Very Dynamic':
                art_dyn = 5
            elif art_dyn == 'Moderate':
                art_dyn = 3
            elif art_dyn == 'Not dynamic at all':
                art_dyn = 1
            else:
                art_dyn = int(art_dyn)

            j+=1

            
            # harmonize
            har_ans = df_survey.iloc[i,j:(j+4)].to_list()
            har_sub = df_survey.iloc[i,j+4]
            har_ans = [int(re.sub(pattern=r'\([^)]*\)', repl='', string= x).replace(' ','')) for x in har_ans]
            dyn_ans = df_survey.iloc[i,(j+5):(j+9)].to_list()
            dyn_sub = df_survey.iloc[i,j+9]
            dyn_ans = [int(re.sub(pattern=r'\([^)]*\)', repl='', string= x).replace(' ','')) for x in dyn_ans]
            qual_ans = df_survey.iloc[i,(j+10):(j+14)].to_list()
            qual_sub = df_survey.iloc[i,j+14]
            qual_ans = [int(re.sub(pattern=r'\([^)]*\)', repl='', string= x).replace(' ','')) for x in qual_ans]
            
        
            jj = 5*(j // 16)
            
            for k, x in enumerate(har_ans):
                t = [res_id, f'P{n_art}', 'Harmony', f'Sound {sound_types[k]}', f'Sound {df_ans[jj+k]}', x, art_dyn, har_sub]
                y.append(t)
                
            for k, x in enumerate(dyn_ans):
                t = [res_id, f'P{n_art}', 'Dynamism', f'Sound {sound_types[k]}', f'Sound {df_ans[jj+k]}', x, art_dyn, dyn_sub]
                y.append(t)
                
            for k, x in enumerate(qual_ans):
                t = [res_id, f'P{n_art}', 'Quality', f'Sound {sound_types[k]}', f'Sound {df_ans[jj+k]}', x, art_dyn, qual_sub]
                y.append(t)

            
    df_y_survey = pd.DataFrame(y, columns=['Your Prolific ID', 'Artwork Number', 'Question Type', 'Survey Column', 'Actual Sound', 'User Answer', 'Artwork Dynamism', 'Comment'])
    df_y_survey.to_csv('survey.csv')

    df_pre_survey.to_csv('pre_survey.csv')
    df_post_survey.insert(0, 'Your Prolific ID', df_pre_survey['Your Prolific ID'], True)
    df_post_survey.to_csv('post_survey.csv')