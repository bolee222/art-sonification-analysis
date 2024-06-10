import os
import h5py
import numpy as np
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from glob import glob

# version for GPD
n_coco = 17
n_lines = 25
n_planes = 4

def rotation(joints):
    x = joints[5]-joints[6]
    y = joints[0]-(joints[11] + joints[12])*0.5
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
    l = n_coco*i-(1+i)*i//2+j-i-1
   
    return jj[l]

def joint_line_distance(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, n_coco, axis=0).reshape(n_lines,n_coco,3)
    J2 = np.repeat(j2, n_coco, axis=0).reshape(n_lines,n_coco,3)
    J3 = np.repeat(joints,n_lines,axis=0).reshape(n_lines,n_coco,3,order='F')


    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/(C) # Heron's formula

    # print(mask)
    jl_d = np.ma.array(jl_d,mask=mask).compressed()

    jl_d = np.nan_to_num(jl_d)
    return jl_d

def line_line_angle(jj_o,lines,mask):
    A = get_jj(jj_o, lines[:,0], lines[:,1])
    J = np.repeat(A, n_lines, axis=0).reshape(n_lines,n_lines,3)
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

    J1 = np.repeat(j1, n_coco, axis=0).reshape(n_lines,n_coco,3)
    J2 = np.repeat(j2, n_coco, axis=0).reshape(n_lines,n_coco,3)
    J3 = np.repeat(joints,n_lines,axis=0).reshape(n_lines,n_coco,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/C
    lj_r = np.sqrt(np.square(A)-np.square(jl_d))/C
    lj_r = np.nan_to_num(lj_r)
    lj_r = np.ma.array(lj_r,mask=mask).compressed()

    
    # for x in lj_r:
    #     x = x.compressed()
    #     print(x)
    # exit()

    return lj_r

if __name__ == '__main__':
    index = 0

    lines = np.array(
        [[5,6], [5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16]] )
    lines = np.append(lines, [[5,9],[6,10],[11,15],[12,16]], axis=0)
    lines = np.append(lines, [[0,9], [0,10], [0,15], [0, 16], [9,10], [9,15], [9,16], [10,15], [10,16], [15,16]], axis=0)
    planes = np.array([[5,7,9],[6,8,10],[11,13,15],[12,14,16]]) 


    print('n_lines :', n_lines)
    print('n_planes :', n_planes)

    jj_mask = np.full((n_coco,n_coco,3), False, dtype=bool)

    for i in range(n_coco):
        jj_mask[i,0:i+1,:] = True

    jl_mask = np.full((n_lines,n_coco), False, dtype=bool)
    for i in range(n_lines):
        jl_mask[i][[0, 1,2,3,4]] = True
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

    flist = sorted(glob('./output/*.npy'))

    for fname in flist:
        if len(os.path.basename(fname)) > 9:
            continue

        bname = os.path.basename(fname)
        pos = np.load(fname)
        pos = np.array([list(x) + [0] for x in pos])

        
        
        xmin, xmax = np.min(pos[:, 0]), np.max(pos[:, 0])
        ymin, ymax = np.min(pos[:, 1]), np.max(pos[:, 1])
        pos[:,0] = (pos[:,0] - xmin) / (xmax - xmin)
        pos[:,1] = (pos[:,1] - ymin) / (ymax - ymin)

        fname = fname.replace('.npy','_jld.npy')

    
        JL_D = joint_line_distance(pos, lines, jl_mask)
        JL_D = sorted(JL_D)[-30:]
        # JL_D = np.log(np.array(sorted(JL_D)[-50:]) / 300 + 1e-5)

        print(bname, np.mean(JL_D), np.std(JL_D))
        # print(JL_D)
        # exit()
        # ljr = line_joint_ratio(pos, lines, jl_mask)
        
        np.save(fname, JL_D)
        
        fname = fname.replace('.npy', '.png')
        plt.figure(figsize=(20,5))
        plt.hist(JL_D, bins=30)
        plt.savefig(fname)
    # R = rotation(pos)
    # L = length(pos, lines)
    # J_C = joint_coordinate(pos, R, L)
    # JJ_D = joint_joint_distance(pos)
    # jj_o = joint_joint_orientation(pos, jj_mask)
    

    
    # LL_A = line_line_angle(jj_o, lines, ll_mask)
    # print(LL_A)
    # JP_D = joint_plane_distance(JJ_D, jj_o, planes, jp_index) # this is for 3D joints
    # LP_A = line_plane_angle(jj_o,planes,lp_index)
    # PP_A = plane_plane_angle(jj_o,planes)
    # JJ_O = np.dot(jj_o,R)


    # plt.figure(figsize=(20,5))
    # plt.hist(JL_D.reshape(-1), bins=30)
    # plt.savefig(f'{fname}_jl_d.png')
    # exit()

    # print(R, L, J_C, JJ_D, jj_o, JL_D, LL_A, JP_D, LP_A, PP_A, JJ_O)
    
    # R = np.array(Parallel(n_jobs=24)(delayed(rotation)(pos[i]) for i in range(frame_count)))
    # L = np.array(Parallel(n_jobs=24)(delayed(length)(pos[i],lines[0:19]) for i in range(frame_count)))
    # J_C[:] = np.array(Parallel(n_jobs=24)(delayed(joint_coordinate)(pos[i], R[i], L[i]) for i in range(frame_count)))
    #     JJ_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_joint_distance)(pos[i]) for i in range(frame_count)))
    #     jj_o = np.array(Parallel(n_jobs=24)(delayed(joint_joint_orientation)(pos[i],jj_mask) for i in range(frame_count)))
    #     JL_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_line_distance)(pos[i],lines,jl_mask) for i in range(frame_count)))
    #     LL_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_line_angle)(jj_o[i],lines,ll_mask) for i in range(frame_count)))
    #     JP_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_plane_distance)(JJ_D[i],jj_o[i],planes,jp_index) for i in range(frame_count)))
    #     LP_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_plane_angle)(jj_o[i],planes,lp_index) for i in range(frame_count)))
    #     PP_A[:] = np.array(Parallel(n_jobs=24)(delayed(plane_plane_angle)(jj_o[i],planes) for i in range(frame_count)))
    #     JJ_O[:] = np.array(Parallel(n_jobs=24)(delayed(np.dot)(jj_o[i],R[i]) for i in range(frame_count))).reshape(frame_count,570)
