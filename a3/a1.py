import numpy as np

def main():
    i_e = np.array([1,2,1])
    i_f1 = np.array([-1,0,1])

    result_1 = np.correlate(np.correlate(i_e,i_f1),i_f1)
    result_2 = np.correlate(i_e,np.correlate(i_f1,i_f1))
    print(result_1,"!=",result_2)


if __name__ == "__main__":
    main()