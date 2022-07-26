import trimesh
import argparse
from utils import BasePointCloud
from ssim import pc_ssim

def loadPointCloud(filepath, **kwargs):
    data = trimesh.load(filepath)
    pointCloud = BasePointCloud(mesh=data, **kwargs)
    return pointCloud

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-inA", "--input_A")
    parser.add_argument("-inB", "--input_B")
    parser.add_argument("--geom", default=True, required=False)
    parser.add_argument("--normal", default=True, required=False)
    parser.add_argument("--curvature", default=False, required=False)
    parser.add_argument("--color", default=False, required=False)
    parser.add_argument("--estimator", default="STD", required=False)
    parser.add_argument("--pooling_type", default="Mean", required=False)
    parser.add_argument("--neighborhood_size", default=12, required=False)
    parser.add_argument("--constant", default=2.2204e-16, required=False)
    parser.add_argument("--ref", default=0, required=False)
    args = parser.parse_args()
    return args

def main():
    args = getArgs()
    pcA = loadPointCloud(args.input_A)
    pcB = loadPointCloud(args.input_B)
    pointSSIM = pc_ssim(pcA, pcB, args)
    print(pointSSIM)

if __name__ == "__main__":
    main()