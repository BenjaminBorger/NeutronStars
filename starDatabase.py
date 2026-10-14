#radius(km),mass(M_sun)
import numpy as np
#obj = np.array([[7854,0.67]]) #
#obj = np.array([[6000,1],[4000,1.2],[1000,1.1],[1270,1.3],[1260,1.2],[1200,1.35]]) #,[7,1.2],[10,1.1],[12.7,1.3],[12.4,1.2],[12.7,1.356]


def obj():
    values = []
    for i in range(160):
        values.append([7000, (i+1) / 100])

    return np.array(values)


obj = obj()
