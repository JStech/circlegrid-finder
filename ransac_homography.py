#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cv2 as cv
from itertools import product, combinations
from math import sin, cos, tan, pi, log
from collections import defaultdict

##### code to generate examples #####

def gen_grid_pts(symmetric, grid_size):
    """Makes grid according to specifications, centered at the origin with largest dimension approximately filling [-1, 1]"""
    if symmetric:
        grid = np.array(list(product(range(0, 2*grid_size[0], 2), range(0, 2*grid_size[1], 2), [1]))) - \
                np.array([grid_size[0]-1, grid_size[1]-1, 0], dtype=float)
        grid[:, :2] /= (max(grid_size)-1)
        return grid
    else:
        grid = np.array(list(product(range(0, grid_size[0], 2), range(0, 2*grid_size[1], 2), [1])) +
                        list(product(range(1, grid_size[0], 2), range(1, 2*grid_size[1], 2), [1]))) - \
                                np.array([(grid_size[0]-1)/2, grid_size[1], 0], dtype=float)
        grid[:, :2] /= (max(grid_size[0]/2, grid_size[1])-1)
        return grid

def distort_grid(grid, H, K1, K2):
    """Apply homography (rotation/perspective) and radial distortion to grid points"""

    grid = grid @ H.T
    grid /= grid[:, 2:]

    for i in range(grid.shape[0]):
        r = np.linalg.norm(grid[i, :2])
        grid[i, :2] *= 1 + K1 * r**2 + K2 * r**4

    return grid

def add_noise_points(grid, num_points, limits, exclude_dist=None):
    """add extra false detections"""
    x_range = limits[0][1] - limits[0][0]
    y_range = limits[1][1] - limits[1][0]
    origin = np.array([limits[0][0], limits[1][0]])
    noise_pts = np.random.random((num_points, 2)) * np.array([x_range, y_range]) + origin
    if exclude_dist is not None:
        for _ in range(100):
            dists = np.min(np.linalg.norm(np.expand_dims(noise_pts, 0) - np.expand_dims(grid[:, :2], 1), axis=2), axis=0)
            assert dists.shape == (num_points,)
            keep = dists > exclude_dist
            if np.sum(keep) == num_points: break
            new_pts = np.random.random((num_points - np.sum(keep), 2)) * np.array([x_range, y_range]) + origin
            noise_pts = np.r_[noise_pts[keep, :], new_pts]
    return np.r_[grid, np.pad(noise_pts, ((0, 0), (0, 1)), constant_values=1)]

def add_detection_noise(grid, sigma):
    """add zero-mean normally distributed isotropic noise to detections"""
    grid[:, :2] += sigma * np.random.randn(grid.shape[0], 2)
    return grid

##### circlegrid finding algorithm #####

def estimate_homography(pts):
    """estimate homography from four points in + pattern"""
    A = np.zeros((8, 9))
    pts_plane = np.array([
        [1, 0, 1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, 1],
        ])
    for i, (pt, ppl) in enumerate(zip(pts, pts_plane)):
        A[2*i:2*i+2, :] = np.r_[
                [np.r_[0, 0, 0, -ppl[2] * pt, ppl[1] * pt]],
                [np.r_[ppl[2] * pt, 0, 0, 0, -ppl[0] * pt]],
                ]
    _, _, v = np.linalg.svd(A)
    return v[-1, :].reshape(3, 3)

def find_grid_points(pts, seed, grid_size, symmetric):
    """Identify which points in pts are near integer coordinates"""
    # round
    pts_rounded = np.rint(pts)
    bbox = [np.min(pts_rounded[:, :2], axis=0), np.max(pts_rounded[:, :2])]

    # reject outliers
    outliers = set()
    for i, j in combinations(range(pts.shape[0]), 2):
        if np.linalg.norm(pts_rounded[i, :] - pts_rounded[j, :]) < 1e-6:
            int_dist_i = np.linalg.norm(pts_rounded[i, :] - pts[i, :])
            int_dist_j = np.linalg.norm(pts_rounded[j, :] - pts[j, :])
            outliers.add(i if int_dist_i > int_dist_j else j)

    for i in range(pts.shape[0]):
        neighbors = 0
        for v in (np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0])):
            neighbors += (np.min(np.linalg.norm(pts_rounded - (pts_rounded[i, :] + v), axis=1)) < 1e-6)
        if neighbors == 0:
            outliers.add(i)

    inliers = set(range(pts.shape[0])) - outliers
    if len(inliers) < grid_size[0] * grid_size[1]:
        return

    # lines that might align with grid axes
    axis_lines = [
            np.array([ 0,  1,  0]),
            np.array([ 1,  0,  0]),
            np.array([ 1,  1,  0]),
            np.array([ 1, -1,  0]),
            np.array([ 1,  2,  0]),
            np.array([ 1, -2,  0]),
            np.array([ 2,  1,  0]),
            np.array([ 2, -1,  0]),
            ]

    residual_distributions = []
    x_axis_limits = []
    y_axis_limits = []
    for axis_line in axis_lines:
        distribution = defaultdict(int)
        for pt in pts_rounded[list(inliers), :]:
            distribution[int(np.dot(axis_line, pt))] += 1
        residual_distributions.append(distribution)

        # if this in fact is the x axis, which lines (residuals) have the correct number of points?
        x_enough_pts = [r for r, c in distribution.items() if c >= grid_size[0] // (2 - symmetric)]
        if len(x_enough_pts) >= grid_size[1] * (2 - symmetric) and 0 in x_enough_pts:
            x_axis_limits.append((min(x_enough_pts), max(x_enough_pts)))
        else:
            x_axis_limits.append(None)
        y_enough_pts = [r for r, c in distribution.items() if c >= grid_size[1]]
        if len(y_enough_pts) >= grid_size[0] and 0 in y_enough_pts:
            y_axis_limits.append((min(y_enough_pts), max(y_enough_pts)))
        else:
            y_axis_limits.append(None)

    # usually just one pair of axes meets the above criteria
    grid_ids = None
    for i_x, i_y in product(range(len(axis_lines)), repeat=2):
        if i_x == i_y or x_axis_limits[i_x] is None or y_axis_limits[i_y] is None: continue
        grid_ids = -1 * np.ones(grid_size, dtype=int)

        x_axis = axis_lines[i_x]
        y_axis = axis_lines[i_y]
        for i, j in product(range(grid_size[0]), range(grid_size[1])):
            if symmetric:
                pt = np.cross(
                        np.r_[x_axis[:2], -(x_axis_limits[i_x][0] + j)],
                        np.r_[y_axis[:2], -(y_axis_limits[i_y][0] + i)])
            else:
                pt = np.cross(
                        np.r_[x_axis[:2], -(x_axis_limits[i_x][0] + 2*j + i%2)],
                        np.r_[y_axis[:2], -(y_axis_limits[i_y][0] + i)])
            pt = (pt / pt[2]).astype(int)
            pt_id = np.argmin(np.linalg.norm(pts_rounded - pt, axis=1))
            if np.linalg.norm(pts_rounded[pt_id, :] - pt) > 0.5:
                grid_ids = None
                break
            grid_ids[i, j] = pt_id

    return grid_ids

def ransac_homography(pts, grid_size, symmetric):
    """find + pattern to recover homography"""
    assert len(pts.shape) == 2 and pts.shape[1] == 3, f"{pts.shape = }"

    cos_thresh = 0.99
    found = False
    seeds = list(range(pts.shape[0]))
    while len(seeds) > 0:
        seed_i = np.random.randint(len(seeds))
        seed = seeds.pop(seed_i)
        pt = pts[seed, :]
        dists = np.linalg.norm(pts - pt, axis=1)
        nearest_neighbors = np.argsort(dists)
        assert dists[nearest_neighbors[0]] < 1e-9, f"{nearest_neighbors}"

        for n_plus in combinations(nearest_neighbors[1:11], 4):
            n_pts = pts[n_plus, :]
            intersection = np.cross(np.cross(n_pts[0], n_pts[1]), np.cross(n_pts[2], n_pts[3]))
            intersection /= intersection[2]
            if np.linalg.norm(intersection - pt) > 1e-2 * max(np.linalg.norm(n_pts[0] - n_pts[1]),
                                                              np.linalg.norm(n_pts[2] - n_pts[3])):
                continue

            h = estimate_homography(n_pts)

            pts_plane = pts @ h.T
            pts_plane /= pts_plane[:, 2:]

            # first check: does this map the seed to the origin?
            if np.linalg.norm(pts_plane[seed, :2]) > 1e-2:
                continue

            grid_ids = find_grid_points(pts_plane, seed, grid_size, symmetric)
            if grid_ids is None:
                continue

            return grid_ids

if __name__ == "__main__":

    asymm_grid_size = (np.random.randint(6, 15), np.random.randint(3, 8))
    symm_grid_size = tuple(np.random.randint(3, 15, size=(2,)))

    H = np.array([[ 0.6 + 0.2*np.random.randn(),  0.1*np.random.randn(), 0.5*np.random.randn()],
                  [ 0.1*np.random.randn(),  0.5 + 0.3*np.random.randn(),  0.5*np.random.randn()],
                  [ 0.1*np.random.randn(),  0.1*np.random.randn(),  1.0]])
    if np.linalg.det(H) < 0:
        H[:, 0] *= -1
    theta = np.random.randn()
    H @= np.array([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]])

    K1 = 1e-2 * np.random.randn()
    K2 = 1e-4 * np.random.randn()

    num_noise_pts = 20
    noise_limit = 2
    ex_dist = 0.01
    sigma = 5e-5

    symm_grid = add_detection_noise(
            distort_grid(
                add_noise_points(
                    gen_grid_pts(True, symm_grid_size), num_noise_pts,
                    ((-noise_limit, noise_limit), (-noise_limit, noise_limit)),
                    ex_dist
                    ), H, K1, K2), sigma)
    asymm_grid = add_detection_noise(
            distort_grid(
                add_noise_points(
                    gen_grid_pts(False, asymm_grid_size), num_noise_pts,
                    ((-noise_limit, noise_limit), (-noise_limit, noise_limit)),
                    ex_dist
                    ), H, K1, K2), sigma)

    for pts, grid_size, symm in ((asymm_grid, asymm_grid_size, False), (symm_grid, symm_grid_size, True)):
        expected_num_pts = grid_size[0] * grid_size[1]
        grid_ids = ransac_homography(pts, grid_size, symm)
        if grid_ids is None or set(range(expected_num_pts)) != set(grid_ids.flat):
            print("failure detected")
            with open("failures.txt", "a") as out_file:
                print(grid_size, symm, file=out_file)
                print(pts, file=out_file)
                print("---\n", file=out_file)
        if grid_ids is None:
            continue
        plt.subplot(1, 2, symm+1)
        plt.plot(pts[:expected_num_pts, 0], pts[:expected_num_pts, 1], '.')
        plt.plot(pts[expected_num_pts:, 0], pts[expected_num_pts:, 1], '.')
        plt.plot(pts[grid_ids.flat, 0], pts[grid_ids.flat, 1], ':')
        plt.gca().set_aspect('equal')
    plt.show()
