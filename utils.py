import trimesh
from trimesh.caching import cache_decorator
import numpy as np

class BasePointCloud(trimesh.points.PointCloud):
    """
    We could have import normals with metadata however
    it is more convenient to have our base data class.
    """

    def __init__(self, mesh:trimesh.Trimesh, colors=None, metadata=None, **kwargs):
        vertices = mesh.vertices
        super().__init__(vertices, colors, metadata, **kwargs)
        self.mesh = mesh
        # Find normals at init to get rid of mesh.
        self.normal

    @cache_decorator
    def normal(self):
        normals = self.mesh.vertex_normals
        self.__delattr__('mesh')
        return normals

def idxWithArray(a:np.ndarray, b:np.ndarray):
    """
    This is a feature in Matlab, I couldn't find any similar on Python
    So I wrote it myself for Numpy 2D arrays.
    
    Function tiles array 'a' in a loop of range(b.shape[-1])
    and sorts a with b[:,idx] before concatenation.

    INPUTS
        a: np.ndarray, shape of [n x m]: vertex_normals of point cloud for this matter.
        b: np.ndarray, shape of [n x p] k-neighbors indexes for every point for this matter. 
            p is the number of neighbors for every point.
    OUTPUT
        out: np.ndarray, shape of [n*p x m]
    """
    out = np.zeros((0,3), dtype=np.float64)
    if a.shape[0] == b.shape[0]:
        for idx in range(b.shape[-1]):
            arr = a[b[:,idx]]
            out = np.concatenate((out,arr), axis=0)
    return out


   
