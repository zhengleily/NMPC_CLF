import irispy
import numpy as np

def test_random_obstacles_3d(show=False):
    bounds = irispy.Polyhedron.from_bounds([0, 0, 0], [10, 10, 10])
    barrier = np.array([
        [3, 3, 3, 2],
        [7, 7, 7, 2],
        [3, 5, 1, 2],
        [6, 6, 3, 2]
    ])
    obstacles = []
    for obstacle in barrier:
        center = obstacle[:3]
        scale = obstacle[-1]
        pts = np.array([[1,0,0],[0,-1/3**0.5,0],[0,1/3**0.5,0],[1/3,0,1]]).T
        pts = pts - np.mean(pts, axis=1)[:, np.newaxis]
        pts = scale * pts + center[:, np.newaxis]
        obstacles.append(pts)
    while 1:
        start = list(map(float, input().split()))
        region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)
        debug.animate(pause=0.5, show=show)

if __name__ == '__main__':
    test_random_obstacles_3d(True)