import numpy as np
from aufg3 import calc_camera_matrix
import math

def calc_right_null_space(u,s,v_t, rcond=None):
    u_row_shape, v_column_shape = u.shape[0], v_t.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(u_row_shape, v_column_shape)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    return v_t[num:,:].T.conj()

def calc_calibration_matrix(camera_matrix,camera_orientation):
    m = np.delete(camera_matrix,-1,1)
    mmT = m @ m.T

    a = mmT[0][0]
    b = mmT[0][1]
    c = mmT[0][2]
    d = mmT[1][1]
    e = mmT[1][2]

    x_u = c
    x_v = e 
    k_u = math.sqrt(a - ((b-c*e)**2 / (d - e**2)) - c**2)
    k_v = math.sqrt(d-e**2)
    s = (b - c*e)/math.sqrt(d-e**2)

    return np.array([
        [k_u,s  ,x_u],
        [0  ,k_v,x_v],
        [0  ,0  ,1  ]
    ])

def main():
    p = np.array(
        [
            [490,-390,-1500,1300],
            [-590,1400,-600,1300],
            [-0.5*math.sqrt(2),-0.3*math.sqrt(2),-0.4*math.sqrt(2),5]
        ]
    )

    u,s,vh = np.linalg.svd(p,full_matrices=True)
    camera_center = calc_right_null_space(u,s,vh)

    m = np.delete(p,-1,1)
    calibration_matrix = calc_calibration_matrix(p,camera_center)

    camera_orientation = np.linalg.inv(calibration_matrix) @ m

    re_p = calc_camera_matrix(calibration_matrix,camera_orientation,np.delete(camera_center,-1,axis=0))

    np.set_printoptions(suppress=True)
    print("camera center (c) \n",camera_center)
    print()
    print("Test Camera center(c) Pc = 0 \n",p@camera_center)
    print()
    print("calibration matrix (K) \n",calibration_matrix)
    print()
    print("orientation of the camera (R) \n",camera_orientation)
    print()
    print("reconst. Camera Matrix (P_reconst.) \n",re_p)
    print()
    print("compare recost. Ca.Matrix with given Ca.Matrix (0 = same value in P and P_reconst. )\n",p - re_p)

if __name__ == "__main__":
    main()