import irispy
import numpy as np

def test_random_obstacles_2d(show=False):
    bounds = irispy.Polyhedron.from_bounds([0, 0], [10, 10])
    obstacles = []
    center = np.array([[3, 3], [7, 7], [5, 3]])
    for i in range(3):
        ct = center[i,:]
        scale = np.random.random() * 3
        pts = np.random.random((2,4))
        pts = pts - np.mean(pts, axis=1)[:,np.newaxis]
        pts = scale * pts + ct[:,np.newaxis]
        obstacles.append(pts)
        start = np.array([4, 3])

    while 1:
        start = np.array(list(map(int,input().split())))
        region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

        debug.animate(pause=0.5, show=show)

if __name__ == '__main__':
    test_random_obstacles_2d(True)