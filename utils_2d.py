"""Contains utilities common to 2d meshing methods"""
# based on https://github.com/BorisTheBrave/mc-dc
# master@a165b326849d8814fb03c963ad33a9faf6cc6dea

import math
import meshio

from settings import CELL_SIZE, XMAX, XMIN, YMAX, YMIN
import numpy as np

class V2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def normalize(self):
        d = math.sqrt(self.x * self.x + self.y * self.y)
        return V2(self.x / d, self.y / d)


def element(e, **kwargs):
    """Utility function used for rendering svg"""
    s = "<" + e
    for key, value in kwargs.items():
        s += " {}='{}'".format(key, value[0])
    s += "/>\n"
    return s


def make_svg(file, edges, f, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX,
             cellSize=CELL_SIZE):
    """Writes an svg file showing the given edges and solidity"""
    scale = 50
    file.write("<?xml version='1.0' encoding='UTF-8'?>\n")
    url = "http://www.w3.org/2000/svg"
    file.write("<svg version='1.1' xmlns='{}' viewBox='{} {} {} {}'>\n"
               .format(url, xmin * scale,
                       ymin * scale,
                       (xmax - xmin) * scale,
                       (ymax - ymin) * scale))

    file.write("<g transform='scale({})'>\n".format(scale))
    # Draw edges
    for edge in edges:
        file.write(element("line", x1=edge.v1.x, y1=edge.v1.y,
                           x2=edge.v2.x, y2=edge.v2.y,
                           style='stroke:rgb(255,0,0);stroke-width:0.04'))

    file.write("</g>\n")
    file.write("</svg>\n")

def writeVtk(edges, file):
    """Writes a VTK mesh file from the given edges."""
    points = []
    for e in edges:
        points.append([e.v1.x, e.v1.y, 0])
        points.append([e.v2.x, e.v2.y, 0])

    lineIndices = []
    for i in range(len(edges)):
        lineIndices.append([i*2, i*2 + 1])
    cells = [("line", lineIndices)]

    mesh = meshio.Mesh(points, cells)

    mesh.write(file)


def writeContoursToVtk(contours, file):
    """Writes a VTK mesh file from the given contours (chains of vertices that form edges)."""
    points = []
    line_indices = []
    first_point = 0
    for contour_pts in contours:
        print("len(contour_pts)", len(contour_pts))
        for i in range(len(contour_pts)-1):
            points.append(contour_pts[i])
            line_indices.append([first_point + i, first_point + i + 1])
        points.append(contour_pts[-1])
        first_point = len(points)

    cells = [("line", line_indices)]

    mesh = meshio.Mesh(points, cells)

    mesh.write(file)