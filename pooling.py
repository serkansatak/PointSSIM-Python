import numpy as np

def pooling(qMap, poolingType):
    if poolingType == 'Mean':
        score = np.nanmean(qMap)
    elif poolingType == 'MSE':
        score = np.nanmean(qMap**2)
    elif poolingType == 'RMS':
        score = np.sqrt(np.nanmean(qMap**2))
    else:
        raise Exception('Wrong pooling type...')
    return score