import numpy as np

def main():
    rgb = np.array([1,0.5,0])
    max_index = np.where(rgb == np.amax(rgb))
    delta = np.max(rgb)-np.min(rgb)

    max_index = np.argmax(rgb)

    if max_index == 0:
        hue = (rgb[1] - rgb[2])/delta
    elif max_index == 1:
        hue = 2.0 + (rgb[2]-rgb[0])/delta
    else:
        hue = 4.0 + (rgb[0]-rgb[1])/delta
    
    print("Hue = ",hue)

if __name__ == "__main__":
    main()