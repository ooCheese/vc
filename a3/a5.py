import numpy as np

def main():
    
    rg = (1/2,1/4)
    rgb_g = 1

    rgb_r = (rg[0]*rgb_g)/rg[1]
    rgb_b = ((1 - rg[0] - rg[1])*rgb_g)/rg[1]

    rgb = np.array([rgb_r,rgb_g,rgb_b])
    rgb = rgb/max(rgb)

    print("RGB : ",rgb)

if __name__ == "__main__":
    main()