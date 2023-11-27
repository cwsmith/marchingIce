import time
import numpy as np
import marching_cubes_2d as mc
import meshio
import matplotlib.pyplot as plt

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
            if np.isclose(thk[i][j], 0) or thk[i][j] < 0:
                phi[i][j] = max_distance
            else:
                phi[i][j] = rho_i * thk[i][j] + rho_w * topg[i][j]
    # plt.imshow(phi, interpolation="nearest", origin="upper")
    # plt.colorbar()
    # plt.show()
    cgi = Interp((x, y), phi.T, method='linear')
    # The following reproduces the plot of phi above...
    #  the previous interpolator did not.
    # interp = np.zeros((rows, cols))
    # for i in range(rows):
    #    for j in range(cols):
    #        d = cgi(x[j], y[i])
    #        interp[i][j] = d
    # plt.imshow(interp, interpolation="nearest", origin="upper")
    # plt.colorbar()
    # plt.show()
    mcBegin = time.time()
    edges = mc.marching_cubes_2d(cgi, min(x), max(x), min(y), max(y), dx)
    mcEnd = time.time()
    print("marching_cubes_2d done" + str(mcEnd - mcBegin))

    print("len(edges)", len(edges))
    points = []
    for e in edges:
        points.append([e.v1.x, e.v1.y, 0])
        points.append([e.v2.x, e.v2.y, 0])

    lineIndices = []
    for i in range(len(edges)):
        lineIndices.append([i, i + 1])
    cells = [("line", lineIndices)]

    mesh = meshio.Mesh(points, cells)

    mesh.write("gis.vtk")

    toc = time.time()
    print("mesh_gl end\n" + str(toc - tic))
