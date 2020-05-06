import numpy as np

def main():
    i_f = np.array([
        [0,1,0],
        [1,0,1],
        [0,0,0]])

    i_f = np.flip(i_f,0)
    i_f = np.flip(i_f,1)
 

    i_e = np.array([
        [0,1,1,1],
        [1,1,0,1],
        [0,0,1,1],
        [1,1,0,0]
    ])

    conv = np.zeros((i_e.shape[0],i_e.shape[0]))
    shape_over = int(i_f.shape[0]/2)

    for row in range(conv.shape[0]):
        for column  in range(conv.shape[1]):
            number = 0
            for y in range(i_f.shape[0]):

                c_y = (y-shape_over)+row
                if c_y >= i_e.shape[1]:
                    c_y = c_y - i_e.shape[1]

                for x in range(i_f.shape[1]):

                    c_x = (x-shape_over)+column
                    if c_x >= i_e.shape[0]:
                        c_x = c_x - i_e.shape[0]
                    
                    number += i_e[c_y][c_x] * i_f[y][x]

            conv[row][column] = number
    
    print(conv)

if __name__ == "__main__":
    main()