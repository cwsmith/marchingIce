import time
import numpy as np
import utils_2d as util

from skimage.measure import find_contours
from scipy.interpolate import interpn


def mesh_gl(thk, topg, x, y):
    class Interp:
        def __init__(self, points, values, fill, method='linear'):
            self.points = points
            self.values = values
            self.method = method
            self.fill = fill

        def __call__(self, x, y):
            return interpn(self.points, self.values, (x, y), bounds_error=False, fill_value=self.fill)

    print("mesh_gl start\n")
    assert (thk.shape == (len(y), len(x)))
    tic = time.time()

    cell_size = x[1] - x[0]  # assumed constant and equal in x and y
    half_cell_size = cell_size/2

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
    cgi = Interp((x, y), phi.T, fill=max_distance, method='linear')
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    mid_pt_x, mid_pt_y = np.mgrid[min_x+half_cell_size:max_x:cell_size,
                                  min_y+half_cell_size:max_y:cell_size]
    phi_mid_pt = cgi(mid_pt_x, mid_pt_y)
    phi_mid_pt_grid = np.reshape(phi_mid_pt, mid_pt_x.shape)
    ms_begin = time.time()
    contour_pts = find_contours(phi_mid_pt_grid, 0.0)
    ms_end = time.time()
    print("find_contours done: {:.2f} seconds".format(ms_end - ms_begin))

    print("len(contour_pts)", len(contour_pts))
    #util.writeVtk(edges, "gis.vtk")

    toc = time.time()
    print("mesh_gl done: {:.2f} seconds\n".format(toc - tic))
