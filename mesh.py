import time
import numpy as np
import marching_cubes_2d as mc
from scipy import ndimage

def mesh_gl(thk, topg, x, y):
    class CartesianGridInterpolator:
        def __init__(self, points, values, method='linear'):
            self.limits = np.array([[min(x), max(x)] for x in points])
            self.values = np.asarray(values, dtype=float)
            self.order = {'linear': 1, 'cubic': 3, 'quintic': 5}[method]

        def __call__(self, x, y):
            """
            `xi` here is an array-like (an array or a list) of points.

            Each "point" is an ndim-dimensional array_like, representing
            the coordinates of a point in ndim-dimensional space.
            """
            # transpose the xi array into the ``map_coordinates`` convention
            # which takes coordinates of a point along columns of a 2D array.
            xi = np.asarray([[x, y]]).T

            # convert from data coordinates to pixel coordinates
            ns = self.values.shape
            coords = [(n - 1) * (val - lo) / (hi - lo)
                      for val, n, (lo, hi) in zip(xi, ns, self.limits)]

            # interpolate
            return ndimage.map_coordinates(self.values, coords,
                                           order=self.order,
                                           cval=np.nan)  # fill_value

    print("mesh_gl start\n")
    tic = time.time()

    np.save("thk", thk)
    np.save("topg", topg)
    np.save("x", x)
    np.save("y", y)

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
    (xSize, ySize) = thk.shape
    max_distance = max(max(x), max(y))
    phi = np.zeros((xSize, ySize))
    # FIXME this isn't very pythonish
    for i in range(xSize):
        for j in range(ySize):
            if thk[i][j] <= 0:
                phi[i][j] = max_distance
            else:
                phi[i][j] = rho_i * thk[i][j] + rho_w * topg[i][j]
    cgi = CartesianGridInterpolator((x, y), phi, method='linear')

    edges = mc.marching_cubes_2d(cgi, min(x), max(x), min(y), max(y), dx)
    edgePointsX = []
    edgePointsY = []
    for e in edges:
        edgePointsX.append(e.v1.x)
        edgePointsY.append(e.v1.y)
        edgePointsX.append(e.v2.x)
        edgePointsY.append(e.v2.y)


    toc = time.time()
    print("mesh_gl end\n" + str(toc - tic))