from error_map import error_map
from feature_map import feature_map
from pooling import pooling
import numpy as np


def ssim_score(quantA, quantB, idBA, idAB, params):
    ssimBA = []
    ssimAB = []
    ssimSym = []
    
    featMapA = feature_map(quantA, params.estimator)
    featMapB = feature_map(quantB, params.estimator)

    if (params.ref == 0) or (params.ref == 1):
        errorMapBA = error_map(featMapB, featMapA, idBA, params.constant)
        ssimMapBA = 1 - errorMapBA
        ssimBA = pooling(ssimMapBA, params.pooling_type)

    if (params.ref == 0) or (params.ref == 2):
        errorMapAB = error_map(featMapA, featMapB, idAB, params.constant)
        ssimMapAB = 1 - errorMapAB
        ssimAB = pooling(ssimMapAB, params.pooling_type)

    if (params.ref == 0):
        ssimSym = min(ssimBA, ssimAB)

    return ssimBA, ssimAB, ssimSym