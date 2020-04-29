import numpy as np
import math

def calc_right_null_space(u,s,v_t, rcond=None):
    u_row_shape, v_column_shape = u.shape[0], v_t.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(u_row_shape, v_column_shape)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    return v_t[num:,:].T.conj()

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
    calibration_matrix = (p@ p.T )@ np.linalg.inv(m.T)

    camera_orientation = np.linalg.inv(calibration_matrix) @ m

    np.set_printoptions(suppress=True)
    print("camera center \n",camera_center)
    print("Test Camera center(c) Pc = 0 \n",p@camera_center)
    print()
    print("calibration matrix \n",calibration_matrix)
    print()
    print("orientation of the camera \n",camera_orientation)

    #print(camera_orientation @ camera_center[:-1] *-1)
    #print("Test P = K[R | t] t = -Rc\n",calibration_matrix @ np.append(camera_orientation,camera_orientation @ camera_center.T *-1),axis=1)

if __name__ == "__main__":
    main()