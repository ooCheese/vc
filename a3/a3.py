import numpy as np


def main():
    p = np.array([
        [[40,0,0],  [0,80,0],   [60,0,0],   [0,100,0]],
        [[0,80,0],  [0,0,100],  [0,100,0],  [0,0,40]],
        [[40,0,0],  [0,100,0],  [100,0,0],  [0,25,0]],
        [[0,25,0],  [0,0,20],   [0,75,0],   [0,0,20]]
    ])

    for row in range(p.shape[0]-2):
        for column in range(p.shape[1]-2):
            p_i = p[row+1][column+1]

            r = p_i[0]
            g = p_i[1]
            b = p_i[2]

            r_count = 0
            b_count = 0
            g_count = 0

            p_i = p[row+1][column+1]
            for x in range(3):
                for y in range(3):
                    c_x = x + row
                    c_y = y + column
                    cell = p[c_x][c_y]
                    if p_i[0] == 0 and cell[0] != 0 :
                        r += cell[0]
                        r_count += 1
                    elif p_i[1] == 0 and cell[1] != 0:
                        g += cell[1]
                        g_count += 1
                    elif p_i[2] == 0 and cell[2] != 0:
                        b += cell[2]
                        b_count += 1
            if r_count != 0:
                r /= r_count
            if b_count != 0:
                b /= b_count
            if g_count != 0:
                g /= g_count
            
            print("rgb in p_",row+1,column+1," => ",r,g,b)

if __name__ == "__main__":
    main()
