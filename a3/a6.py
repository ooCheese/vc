import numpy as np

def main():
    a = np.array([17,42])
    b = np.array([289,68])

    print("4 way neighborhood ",b[0] - a[0])
    print("8 way neighborhood ",max(b[0] - a[0],b[0] - a[0]))
    print("euclidean distance ",np.linalg.norm(a-b))


if __name__ == "__main__":
    main()