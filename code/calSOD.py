import os
import h5py
import numpy as np
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
# version for GPD
n_coco = 14


def rotation(joints):
    x = joints[5]-joints[2]
    y = (joints[1]-joints[8]) + (joints[1] - joints[11]) 
    y *= 0.5
    z = np.cross(x,y)
    R = np.array(normalize([x,y,z]))
    return R

def length(joints,lines):
    j1,j2 = joints[lines.swapaxes(1,0)]
    L = np.sum(np.sqrt(np.sum(np.square(j1-j2),axis=1)))
    return L

def joint_coordinate(joints,rotation, L):
    J = np.dot(joints-joints[0],rotation)
    J = J/L
    J = np.nan_to_num(J)
    j_c = np.append(J[0][1],J[1:])
    # j_c = (joints-joints[0]).flatten()
    return j_c

def joint_joint_distance(joints):
    D = pdist(joints)
    D = np.nan_to_num(D)
    return D

def joint_joint_orientation(joints,mask):
    J = np.repeat(joints, n_coco, axis=0).reshape(n_coco, n_coco, 3)
    J_T = J.swapaxes(1, 0)
    O = J - J_T
    jj_o = normalize(np.ma.array(O,mask=mask).compressed().reshape(-1,3))
    return jj_o

def get_jj(jj,i,j):
    # jj could be jj_o or jj_d
    j1 = np.minimum(i,j)
    j2 = np.maximum(i,j)
    return jj[n_coco*i-(1+i)*i/2+j-i-1]

def joint_line_distance(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, n_coco, axis=0).reshape(len(lines),n_coco,3)
    J2 = np.repeat(j2, n_coco, axis=0).reshape(len(lines),n_coco,3)
    J3 = np.repeat(joints,len(lines),axis=0).reshape(len(lines),n_coco,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/(C)
    jl_d = np.ma.array(jl_d,mask=mask).compressed()
    jl_d = jl_d
    jl_d = np.nan_to_num(jl_d)
    return jl_d

def line_line_angle(jj_o,lines,mask):
    A = get_jj(jj_o, lines[:,0], lines[:,1])
    J = np.repeat(A, len(lines), axis=0).reshape(len(lines),len(lines),3)
    J_T = J.swapaxes(1,0)
    ll_a = np.ma.array(np.sum(J*J_T,axis=2),mask=mask).compressed()
    ll_a[np.where(ll_a > 1)] = 1
    ll_a[np.where(ll_a < -1)] = -1
    ll_a = np.arccos(ll_a)
    return ll_a


def joint_plane_distance(jj_d,jj_o,planes,index):
    D = get_jj(jj_d,index[:,:,0],index[:,:,1])
    A = get_jj(jj_o,index[:,:,0],index[:,:,1])
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    jp_d = D*np.sum(A*np.cross(B,C),axis=2)
    return jp_d.flatten()


def line_plane_angle(jj_o,planes,index):
    A = get_jj(jj_o,index[:,:,0],index[:,:,1])
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    lp_a = np.sum(A*np.cross(B,C),axis=2)
    lp_a[np.where(lp_a > 1)] = 1
    lp_a[np.where(lp_a < -1)] = -1
    lp_a = np.arccos(lp_a)
    return lp_a.flatten()

def plane_plane_angle(jj_o,planes):
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    pp_a = pdist(np.cross(B,C), lambda u, v: np.dot(u,v))
    pp_a[np.where(pp_a > 1)] = 1
    pp_a[np.where(pp_a < -1)] = -1
    pp_a = np.arccos(pp_a)
    return pp_a

def line_joint_ratio(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, n_coco, axis=0).reshape(len(lines),n_coco,3)
    J2 = np.repeat(j2, n_coco, axis=0).reshape(len(lines),n_coco,3)
    J3 = np.repeat(joints,len(lines),axis=0).reshape(len(lines),n_coco,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/C#???
    lj_r = np.sqrt(np.square(A)-np.square(jl_d))/C
    lj_r = np.nan_to_num(lj_r)
    lj_r = np.ma.array(lj_r,mask=mask).compressed()
    return lj_r

if __name__ == '__main__':

    f = h5py.File("UTKinect-J_c-58.hdf5", 'a')

    skeleton_root = '/home/zsy/data/UT-Kinect'
    index = 0

    # J1 and J2 directly adjacent in the kinectic chain - UTkinect 19

    lines = np.array(
        [[0,1], [1,2], [2,3], [3,4], [1,5],[5,6],[5,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13]] )

    # J1 is end site, J2 is two steps away - UTkinect 5
    lines = np.append(lines, [[0,11], [0,8], [2,4], [5,7], [8,10], [11,13]], axis=0)

    # Both J1 and J2 are end site - UTkinect 10
    lines = np.append(lines, [[0,4], [0,7], [0, 10], [0, 13], [4,7], [4,10], [4,13], [7,10], [7,13], [10,13]], axis=0)

    #- UTkinect 5
    planes = np.array([[0,1,11],[0,1,8],[2,3,4],[5,6,7],[8,9,10],[11,12,13]])


    # 20 : number of indices (14) / 34 : number of lines

     # remove facial points
    n_lines = len(lines)
    n_planes = len(planes)

    jj_mask = np.full((n_coco,n_coco,3), False, dtype=bool)

    for i in range(n_coco):
        jj_mask[i,0:i+1,:] = True

    jl_mask = np.full((n_lines,n_coco), False, dtype=bool)
    for i in range(n_lines):
        jl_mask[i][lines[i]] = True

    ll_mask = np.full((n_lines,n_lines), False, dtype=bool)
    for i in range(n_lines):
        ll_mask[i][0:i+1] = True

    # 17 : # of indices - 3 | 5 : number of planars
    jp_index = np.zeros((n_coco-3, n_planes, 2),dtype='int')
    for i in range(n_planes):
        gen = (k for k in range(n_coco) if k not in planes[i])
        j = 0
        for k in gen:
            jp_index[j,i,0] = min(k,planes[i,0])
            jp_index[j,i,1] = max(k,planes[i,0])
            j = j+1

    # 31 : # of lines - 3 | 5 : number of planars
    lp_index = np.zeros((n_lines-3, n_planes, 2),dtype='int')
    for i in range(n_planes):
        j = 0
        for x,y in lines:
            if x in planes[i] and y in planes[i]:
                continue
            lp_index[j,i,:] = [x,y]
            j = j+1

    for root, dirs, files in os.walk(skeleton_root):
        for file in sorted(files):
            basename = file.split('.')[0][7:]
            label_file = open('actionLabel.txt','r')
            label_lines = label_file.readlines()
            ind = label_lines.index(basename+'\n')
            file_path = os.path.join(root, file)
            action_dict = {1:'a01',2:'a02',3:'a03',4:'a04',5:'a05',6:'a06',7:'a07',8:'a08',9:'a09',10:'a10'}
            for x in range(ind+1,ind+11):
                if label_lines[x][:-1].split(' ')[1] == 'NaN':
                    continue
                start = int(label_lines[x][:-1].split(' ')[1])
                end = int(label_lines[x][:-1].split(' ')[2])
                grp = f.require_group(basename+'_'+action_dict[x-ind])
                pos, frame_count = read_skeleton_file(file_path,start,end)
                grp.attrs.modify('frame_count', frame_count)

                J_C = grp.require_dataset("J_c",shape=(frame_count,58), dtype='float32', chunks=True)
                JJ_D = grp.require_dataset("JJ_d",shape=(frame_count,190), dtype='float32', chunks=True)
                JJ_O = grp.require_dataset("JJ_o",shape=(frame_count,570), dtype='float32', chunks=True)
                JL_D = grp.require_dataset("JL_d",shape=(frame_count,612), dtype='float32', chunks=True)
                LL_A = grp.require_dataset("LL_a",shape=(frame_count,561), dtype='float32', chunks=True)
                JP_D = grp.require_dataset("JP_d",shape=(frame_count,85), dtype='float32', chunks=True)
                LP_A = grp.require_dataset("LP_a",shape=(frame_count,155), dtype='float32', chunks=True)
                PP_A = grp.require_dataset("PP_a",shape=(frame_count,10), dtype='float32', chunks=True)
#                 LJ_r = grp.require_dataset("LJ_r",shape=(frame_count,1127), dtype='float32', chunks=True)

                R = np.array(Parallel(n_jobs=24)(delayed(rotation)(pos[i]) for i in range(frame_count)))
                L = np.array(Parallel(n_jobs=24)(delayed(length)(pos[i],lines[0:19]) for i in range(frame_count)))
                J_C[:] = np.array(Parallel(n_jobs=24)(delayed(joint_coordinate)(pos[i], R[i], L[i]) for i in range(frame_count)))
                JJ_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_joint_distance)(pos[i]) for i in range(frame_count)))
                jj_o = np.array(Parallel(n_jobs=24)(delayed(joint_joint_orientation)(pos[i],jj_mask) for i in range(frame_count)))
                JL_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_line_distance)(pos[i],lines,jl_mask) for i in range(frame_count)))
                LL_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_line_angle)(jj_o[i],lines,ll_mask) for i in range(frame_count)))
                JP_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_plane_distance)(JJ_D[i],jj_o[i],planes,jp_index) for i in range(frame_count)))
                LP_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_plane_angle)(jj_o[i],planes,lp_index) for i in range(frame_count)))
                PP_A[:] = np.array(Parallel(n_jobs=24)(delayed(plane_plane_angle)(jj_o[i],planes) for i in range(frame_count)))
                JJ_O[:] = np.array(Parallel(n_jobs=24)(delayed(np.dot)(jj_o[i],R[i]) for i in range(frame_count))).reshape(frame_count,570)
#                 LJ_r[:] = np.array(Parallel(n_jobs=24)(delayed(line_joint_ratio)(pos[i],lines,jl_mask) for i in range(frame_count)))
            index = index + 1
            print (index,'/',len(files),file)