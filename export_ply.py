import trimesh
import sys, os

def main(filepath):
    # load STL file
    basedir, filename = os.path.split(filepath)
    mesh = trimesh.load(filepath) 
    bytearr = trimesh.exchange.ply.export_ply(mesh, encoding='binary', vertex_normal=True, include_attributes=True)
    with open(f"{basedir}/file.ply", "wb+") as f:
        f.write(bytearr)
        f.close()

if __name__ == "__main__":
    path = sys.argv[1]
    main(path)