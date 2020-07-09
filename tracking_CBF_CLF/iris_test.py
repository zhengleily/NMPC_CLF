import irispy
import numpy as np
import matplotlib.pyplot as plt
bounds = irispy.Polyhedron.from_bounds([0, 0, 0], [10, 10, 10])
obstacles = []
for i in range(5):
    center = np.random.random((3,))
    scale = np.random.random() * 0.3
    pts = np.random.random((3,4))
    pts = pts - np.mean(pts, axis=1)[:,np.newaxis]
    pts = scale * pts + center[:,np.newaxis]
    obstacles.append(pts)
    start = np.array([0.5, 0.5, 0.5])

region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)
iter_result = debug.iterRegions()
ellipsoid = list(iter_result)[-1][1]
cov_matrix = ellipsoid.getC()
center_pt = ellipsoid.getD()
lamb, vec = np.linalg.eig(cov_matrix)
axis_x, axis_y, axis_z = cov_matrix.dot(vec).T
print(axis_x.dot(axis_y))

pass