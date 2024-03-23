import csv
import pandas as pd
import os

if __name__ == "__main__":
    data = pd.read_csv('./dataset_prolific/raw.csv', header=None)

    data = data.to_numpy()

    n = len(data)-1
    fname = f'./dataset_prolific/final_prolific.csv'
    fcsv = open(fname, 'w')
    fcsv.write(",nickname,sod_a1,sod_a2,element_pose,element_color,element_light,element_stroke,element_style,elemet_add,detail_poseSize,detail_arms,detail_inclination,detail_effort,detail_face,detail_activity,detail_number,detail_location,detail_humanColor,detail_bgColor\n")
    writer = csv.writer(fcsv, delimiter=',',
        lineterminator='\r\n',
        # quotechar = "\""
        )

    row = []
    for i in range(1, n+1):

        intro = data[i, :10]
        pid = intro[2]
        # artworks = data[i, 10:-23]
        final = data[i, -23:]

        for j in range(len(final)):
            if final[j] == 'very minor':
                final[j] = 1
            elif final[j] == 'minor':
                final[j] = 2
            elif final[j] == 'moderate':
                final[j] = 3
            elif final[j] == 'huge':
                final[j] = 4
            elif final[j] == 'very huge':
                final[j] = 5
        q1 = final[:16]
        sod1=final[16]
        sod2=final[17]
        q2 = final[17:]
        row = [i-1, pid, sod1, sod2] + list(q1)
        writer.writerow(row)

       
    fcsv.close()