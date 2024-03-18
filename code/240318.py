import csv
import pandas as pd
import os


'''
240319 BHB @TODO
  - 최신 csv 업데이트 시 처리
  - Intro & Final 문답 처리
'''

art_order = [1,5,12,21,20,14,2,17,3,16,8,6,11,4,15,7,19,18,13,9,10]

art_type = ["",
            "running", "running", "running", 
            "dancing","dancing","dancing", 
            "ballet","ballet","ballet",
            "fight","fight","fight",
            "sit","sit","sit",
            "lying","lying","lying",
            "satnding","satnding","satnding",]


def parseMain(pid, artworks):

        
    m = len(artworks) // len(art_order) # 하나의 Artwork 당 10 개 문항


    # for j in art_order:

    for j in range(len(art_order)):
        
        ao = art_order[j]
        at = art_type[ao]

        fname = f'./dataset_prolific/artworks/A{ao}.csv'

        fcsv = open(fname, 'a')
        writer = csv.writer(fcsv, delimiter=',',
                lineterminator='\r\n',
                # quotechar = "\""
                )
        k = j*m

        sod = artworks[k]

        row = []

        for ii in range(4):
            map = "["

            ans = str(artworks[k+1+ii])
            
            if ans[:3] != 'nan':
                c = ['0','0','0','0']
                for jj, tag in enumerate(['A','B','C','D']):
                    if 'col{}'.format(tag) in ans:
                        c[jj] = '1'
                
                map += ",".join(c)
            else:
                map +="0,0,0,0"
            

            map += "]"
            row += [map]

        map_r = f"{artworks[k+5]}"
        row += [map_r]

        for ii in range(3):
            comp = artworks[k+6+ii].split(' | ')
            if len(comp) > 1:
                comp[1] = f" {comp[1]}"
            else:
                comp += ["nan"]
            row += comp


        percept = f"{artworks[k+9]}" 
        row += [percept]

        row = [i, pid, ao, at, sod] + row
        
        # ret = f"{i}, {pid}, {ao}, {at}, {sod}, {map}, {map_r}, {', '.join(music_component)}, {percept}\n"
        # print(fname, ret)
        # fcsv.write(ret)
        writer.writerow(row)
        # import sys;sys.exit()
        fcsv.close()

if __name__ == "__main__":
    data = pd.read_csv('./dataset_prolific/raw.csv', header=None)

    data = data.to_numpy()

    n = len(data)-1

    for i in art_order:
        fname = f'./dataset_prolific/artworks/A{i}.csv'
        if not os.path.exists(fname):
            fcsv = open(fname, 'w')
            fcsv.write(",name,artwork,poseType,sod,gRoq1_mat,gRoq2_mat,gRoq3_mat,gRoq4_mat,reasons,tempo_score,tempo,pitch_score,pitch,density_score,density,additional_perception\n")
            fcsv.close()


    for i in range(1, n+1):

        intro = data[i, :10]
        pid = intro[2]
        artworks = data[i, 10:-23]
        parseMain(pid, artworks)
        final = data[i, -23:]

        # ,name,artwork,poseType,sod,gRoq1_mat,gRoq2_mat,gRoq3_mat,gRoq4_mat,reasons,tempo_score,tempo,pitch_score,pitch,density_score,density
        #0,나비,A1,running,8,
        #"[0, 0, 0, 0]","[1, 1, 0, 1]","[1, 1, 0, 1]","[0, 0, 0, 0]",
        #뛰어다닌는듯한 동작 -- 액션을 일부러 취하는 행동 ~ 움직임이 많음 + 파도의 모양,9.0,9,6.0,6,9.0,9


import sys;sys.exit()
# artwork_number=
with open('dataset_prolific/A{}.csv'.format(artwork_number), 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)



print(artworks)
# print(data[1,-23:]) # Final Questions

# mId = data[i, 0]


# with open('./dataset_prolific/raw.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in spamreader:
#         print(row)
#         # print(', '.join(row))
#         break 