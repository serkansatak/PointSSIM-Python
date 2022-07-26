import numpy as np  

def feature_map(quant:np.ndarray, estType:list):
    """
    In Matlab version you can choose more than one estimator type
    However due to simplicity reasons, you can choose only one here.
    
    INPUTS
        quant: Per-attribute quantities that reflect corresponding local
            properties of a point cloud. The size is LxK, with L the number
            of points of the point cloud, and K the number of points
            comprising the local neighborhood.
        estType: Defines the estimator(s) that will be used to
            compute statistical dispersion, with available options:
            {'STD', 'VAR', 'MeanAD', 'MedianAD', 'COV', 'QCD'}.
            More than one options can be enabled.

    OUTPUTS
        fMap: Feature map of a point cloud, per estimator. The size is LxE,
            with L the number of points of the point cloud and E the length
            of the 'estType'.
    """

    fMap = np.zeros((np.size(quant,0), 1), np.float64)

    if estType == "STD":
        fMap = np.std(quant, 1, ddof=1)
    elif estType == "VAR":
        fMap = np.var(quant, 1, ddof=1)
    elif estType == "MeanAD":
        fMap = np.mean(abs(quant - np.mean(quant, 1).reshape(-1,1)), 1)
    elif estType == "MedianAD":
        fMap = np.median(abs(quant - np.median(quant, 1).reshape(-1, 1)), 1)
    elif estType == "COV":
        fMap = np.std(quant, 1, ddof=1) / np.mean(quant, 1)
    elif estType == "QCD":
        """
        There are no equivalent of Matlab's quantile on Python 3.7 
        
        However on latest Numpy versions there is an option called 'method',
        which is a replacement of 'interpolation' option that can take more 
        arguments than interpolation.
        """
        qq = np.quantile(quant, [.25, .75], 1, interpolation='linear')
        fMap = (qq[1,:] - qq[0,:]) / (qq[1,:] + qq[0,:])
    return fMap