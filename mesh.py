import time
import numpy as np
import marching_cubes_2d as mc
import utils_2d as util

from scipy.interpolate import interpn


def mesh_gl(thk, topg, x, y):
    class Interp:
        def __init__(self, points, values, method='linear'):
            self.points = points
            self.values = values
            self.method = method

        def __call__(self, x, y):
            return interpn(self.points, self.values, (x, y))

    print("mesh_gl start\n")
    assert (thk.shape == (len(y), len(x)))
    tic = time.time()

    dx = x[1] - x[0]  # assumed constant and equal in x and y

    rho_i = 910.0
    rho_w = 1028.0
    # Using the grounding line level set
    # expression 'phi = rho_i * thk + rho_w * topg'
    # results in a runtime overflow warning as 'topg'
    # has values around 1e37 near two of the domain
    # corners (minx,miny) and (maxx,miny).
    # In those corners the level set distance will be set to
    # max distance = max(maxx, maxy)
    (rows, cols) = thk.shape
    max_distance = max(max(x), max(y))
    phi = np.zeros((rows, cols))
    # FIXME this isn't very pythonish
    for i in range(rows):
        for j in range(cols):
            if not np.isclose(thk[i][j], 0) and thk[i][j] < 0:
                phi[i][j] = max_distance
            else:
                phi[i][j] = rho_i * thk[i][j] + rho_w * topg[i][j]
    cgi = Interp((x, y), phi.T, method='linear')
    mcBegin = time.time()
    edges = mc.marching_cubes_2d(cgi, min(x), max(x), min(y), max(y), dx)
    mcEnd = time.time()
    print("marching_cubes_2d done: {:.2f} seconds".format(mcEnd - mcBegin))

    print("len(edges)", len(edges))
    util.writeVtk(edges, "gis.vtk")

    toc = time.time()
    print("mesh_gl done: {:.2f} seconds\n".format(toc - tic))
