import numpy as np

def clac_calibration_matrix(k,x):
    return np.array(
        [
            [k[0],0,x[0]],
            [0,k[1],x[1]],
            [0,0,1]
        ]
    )

def calc_camera_matrix(calibration_matrix,camera_orientation,t):
    x = np.append(camera_orientation,t.reshape(3,1),axis=1)
    print("R|t t = -Rc :",x)
    return calibration_matrix @ x

def calc_principal_point_respect_pixel_size(principal_point,pixel_per_unit):
    return principal_point * pixel_per_unit

def calc_focal_length_respect_pixel_size(focal,pixel_per_unit):
    return pixel_per_unit * focal

def main():
    f = 6.0 #mm focal length
    h = np.array([310,250]) #principal point
    c = np.array([100,200,300]) # optical center
    pixel_length = 0.005 #mm
    resultion = np.array([640,480]) #pixels
    camera_orientation = np.identity(3)

    pixel_per_unit = resultion * pixel_length #in px/mm
    k = calc_focal_length_respect_pixel_size(f,pixel_per_unit)
    x = calc_principal_point_respect_pixel_size(h,pixel_per_unit)
    cali = clac_calibration_matrix(k,x)
    t = (camera_orientation*-1) @ c
    p = calc_camera_matrix(cali,camera_orientation,t)

    np.set_printoptions(suppress=True)
    print("k =",k)
    print("x =",x)
    print("K =\n",cali)
    print()
    print("P =\n",p)
    


if __name__ == "__main__":
    main()