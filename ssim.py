from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors 
from dataclasses import dataclass
import numpy as np
from ssim_score import ssim_score
from utils import idxWithArray

def getKnnResult(data_A:"trimesh.PointCloud",data_B:"trimesh.PointCloud", k:int) -> "['np.ndarray', 'np.ndarray']":
    """
    Runs k-NN on datas and returns k-neighbors for each point with distances.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data_A.vertices)
    return neigh.kneighbors(data_B.vertices, return_distance=True)

@dataclass
class PointSSIM:
    ssimBA: float
    ssimAB: float
    ssimSym: float


def pc_ssim(pcA, pcB, params):
    # Formulation of neighborhoods in point clouds A and B
    distA, idA = getKnnResult(pcA, pcA, params.neighborhood_size)
    distB, idB = getKnnResult(pcB, pcB, params.neighborhood_size)
    # Association of neighborhoods between point clouds A and B
    _, idBA = getKnnResult(pcA, pcB, 1)
    _, idAB = getKnnResult(pcB, pcA, 1)

    out = {}

    if params.geom:
        geomQuantA = distA[:, 1:]
        geomQuantB = distB[:, 1:]
        
        out["geomSSIM"] = PointSSIM(*ssim_score(geomQuantA, geomQuantB, idBA, idAB, params))
    
    if params.normal:
        sumA = np.sum(np.multiply(idxWithArray(pcA.normal, idA), 
                                    np.tile(pcA.normal, (params.neighborhood_size, 1))), axis=1)
        secondPartA = np.multiply(np.sqrt(np.sum(idxWithArray(pcA.normal, idA)**2, axis=1)),
                                    np.sqrt(np.sum((np.tile(pcA.normal, (params.neighborhood_size, 1)))**2, axis=1)))
        nsA = np.real(1-(2*np.divide(np.arccos(np.abs(np.divide(sumA, secondPartA))),np.pi)))
        # Compansate nan values from 'arccos(pi/2)'
        nsA[np.isnan(nsA)] = 1 
        normQuantA = np.reshape(nsA, (-1,params.neighborhood_size), order='F')

        sumB = np.sum(np.multiply(idxWithArray(pcB.normal, idB), 
                                    np.tile(pcB.normal, (params.neighborhood_size, 1))), axis=1)
        secondPartB = np.multiply(np.sqrt(np.sum(idxWithArray(pcB.normal, idB)**2, axis=1)),
                                    np.sqrt(np.sum((np.tile(pcB.normal, (params.neighborhood_size, 1)))**2, axis=1)))
        nsB = np.real(1-(2*np.divide(np.arccos(np.abs(np.divide(sumB, secondPartB))),np.pi)))
        # Compansate nan values from 'arccos(pi/2)'
        nsB[np.isnan(nsB)] = 1 
        normQuantB = np.reshape(nsB, (-1,params.neighborhood_size), order='F')

        out["normSSIM"] = PointSSIM(*ssim_score(geomQuantA, geomQuantB, idBA, idAB, params))

    return out

        