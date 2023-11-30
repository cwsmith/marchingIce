import time
import numpy as np
from skimage.measure import find_contours
from scipy.interpolate import interpn
import meshio


def writeContoursToVtk(contours, file):
    """Writes a VTK mesh file from the given contours (chains of vertices that form edges)."""
    points = []
    line_indices = []
    first_point = 0
    contour_id = 0
    contour_ids = []
    for contour_pts in contours:
        print("len(contour_pts) {} closed contour {}",
              len(contour_pts), (contour_pts[-1] == contour_pts[0]).all())
        for i in range(len(contour_pts)-1):
            points.append(contour_pts[i])
            line_indices.append([first_point + i, first_point + i + 1])
            contour_ids.append([contour_id])
        points.append(contour_pts[-1])
        first_point = len(points)
        contour_id = contour_id + 1

    cells = [("line", line_indices)]
    cell_data = {"contour_id": [contour_ids]}

    mesh = meshio.Mesh(points, cells, cell_data=cell_data)

    mesh.write(file)


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
    contours = find_contours(phi_mid_pt_grid, 0.0)
    ms_end = time.time()
    print("find_contours done: {:.2f} seconds".format(ms_end - ms_begin))

    writeContoursToVtk(contours, "gisContours.vtk")

    toc = time.time()
    print("mesh_gl done: {:.2f} seconds\n".format(toc - tic))

    max_contour = max(contours, key=len)
    max_contour_len = len(max_contour)
    print("max sized contour lenth {}\n".format(max_contour_len))

    return max_contour


if __name__ == '__main__':
    thk = np.load("thk.npy")
    topg = np.load("topg.npy")
    x = np.load("x.npy")
    y = np.load("y.npy")
    mesh_gl(thk,topg,x,y)