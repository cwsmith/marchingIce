import numpy as np
import mesh

if __name__ == '__main__':
    thk = np.load("thk.npy")
    topg = np.load("topg.npy")
    x = np.load("x.npy")
    y = np.load("y.npy")
    mesh.mesh_gl(thk,topg,x,y)
