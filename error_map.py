import numpy as np


def error_map(fMapY, fMapX, idYX, CONST):
    fMapX = np.reshape(fMapX, (-1,1))
    fMapY = np.reshape(fMapY, (-1,1))

    eMapYX = np.divide((np.abs(fMapX[idYX].reshape(-1,1) - fMapY)).reshape(-1,1), 
                (np.max(np.concatenate((np.abs(fMapX[idYX].reshape(-1,1)), np.abs(fMapY)), axis=1), axis=1) + CONST).reshape(-1,1))
    return eMapYX

