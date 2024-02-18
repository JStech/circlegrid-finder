#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import cv2 as cv
from itertools import product, combinations
from math import sin, cos, tan, pi, log

#im_size = (600, 800)
#im_center = np.array([300, 400])
#
#img = np.zeros(im_size, dtype='uint8')
#
#dist_coeffs = np.array([2e-6, 1e-12])
#
#for (i, j) in product(range(im_size[0]), range(im_size[1])):
#    px = np.array([i, j])
#    r = np.linalg.norm(px - im_center)
#    c = dist_coeffs[0] * r**2 + dist_coeffs[1] * r**4
#    #px_d = (px + c * im_center) / (1 + c)
#    px_u = px + (px - im_center) * c
#    if int(px_u[0] / 100)%2 ^ int(px_u[1] / 100)%2:
#        img[i, j] = 255
#
#cv.imshow("", img)
#cv.waitKey(0)

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

def fuzzy_hough_transform(points, acc_size, fuzz):
    """detect mostly-straight lines; accumulator is theta x rho"""
    acc = np.zeros(acc_size, dtype=int)
    min_pt = np.min(points[:, :2], axis=0)
    max_pt = np.max(points[:, :2], axis=0)
    max_rho = max(map(np.linalg.norm, [min_pt, max_pt, [min_pt[0], max_pt[1]], [max_pt[0], min_pt[1]]]))
    pt_range = max_pt - min_pt

    for pt in points:
        for theta_i in range(acc_size[0]):
            theta = pi * theta_i / acc_size[0]
            rho = np.dot(pt[:2], [cos(theta), sin(theta)])
            rho_i = int((rho + max_rho) / (2*max_rho) * acc_size[1])
            for offset in range(-fuzz//2, fuzz//2+1):
                if 0 <= rho_i + offset < acc_size[1]:
                    acc[theta_i, rho_i + offset] += 1

    return acc

def find_lines(points, min_pts=3):
    """find sets of points in points that are collinear"""
    lines = {}
    lines2 = {}
    for i, j in combinations(range(points.shape[0]), 2):
        already_found = False
        for line_pt_set in lines.keys():
            if i in line_pt_set and j in line_pt_set:
                already_found = True
                break
        if already_found: continue

        vec1 = points[i, :] - points[j, :]
        dist1 = np.linalg.norm(vec1)
        vec1 /= dist1
        pt_set = {i, j}

        for k in range(points.shape[0]):
            if i==k or j==k: continue
            vec2 = points[i, :] - points[k, :]
            dist2 = np.linalg.norm(vec2)
            vec2 /= dist2

            vec3 = points[j, :] - points[k, :]
            dist3 = np.linalg.norm(vec3)
            vec3 /= dist3

            if np.min(np.abs([np.dot(vec1, vec2), np.dot(vec2, vec3), np.dot(vec3, vec1)])) > 0.99:
                pt_set.add(k)

        if len(pt_set) < min_pts: continue

        pt_idx_list = list(pt_set)
        pt_mat = points[pt_idx_list, :]
        _, _, v = np.linalg.svd(pt_mat)
        line = v[-1, :]
        one_d_coords = pt_mat[:, :2] @ np.array([[0, 1], [-1, 0]]) @ line[:2]
        pt_idx_list = tuple(x for _, x in sorted(zip(one_d_coords, pt_idx_list)))

        lines[pt_idx_list] = line
        lines2[frozenset(pt_set)] = line
        assert len(lines) == len(lines2), f"{list(lines.keys())}\n{list(lines2.keys())}"

    return lines

    #for i, j in combinations(range(points.shape[0]), 2):
    #    already_found = False
    #    for line_pt_set in lines.keys():
    #        if i in line_pt_set and j in line_pt_set:
    #            already_found = True
    #            break
    #    if already_found: continue

    #    line = np.cross(points[i, :], points[j, :])
    #    line /= np.linalg.norm(line[:2])
    #    pt_set = {i, j}
    #    for k in range(points.shape[0]):
    #        if abs(np.dot(line, points[k, :])) < 2*min_dist:
    #            pt_set.add(k)

    #    if len(pt_set) < min_pts: continue
    #    # re-fit line to all points
    #    _, _, v = np.linalg.svd(points[list(pt_set), :])
    #    line = v[-1, :]
    #    line /= np.linalg.norm(line[:2])
    #    pt_set = [i, j]
    #    for k in range(points.shape[0]):
    #        if abs(np.dot(line, points[k, :])) < min_dist:
    #            pt_set.append(k)
    #    pt_mat = points[pt_set, :]
    #    _, _, v = np.linalg.svd(pt_mat)
    #    line = v[-1, :]
    #    line /= np.linalg.norm(line[:2])
    #    one_d_coord = pt_mat[:, :2] @ np.array([[0, 1], [-1, 0]]) @ line[:2]
    #    pt_set = [idx for idx in np.argsort(one_d_coord)]
    #    already_found = False
    #    lines_to_pop = []
    #    for line_pt_set in lines.keys():
    #        if set(line_pt_set).issuperset(set(pt_set)):
    #            already_found = True
    #            continue
    #        if set(line_pt_set).issubset(set(pt_set)):
    #            lines_to_pop.append(line_pt_set)
    #    for line_to_pop in lines_to_pop:
    #        lines.pop(line_to_pop)
    #    if not already_found:
    #        lines[tuple(pt_set)] = line
    #return lines

def find_grids(pts, lines, symmetric, grid_size):
    """Given dict of (point set: line) pairs, determine how to assign points to grid positions"""
    if symmetric:
        large_grid = grid_size
    else:
        large_grid = ((grid_size[0] + 1)//2, grid_size[1])
        small_grid = (grid_size[0]//2, grid_size[1])

    # find lines that could be parallel (have no points in common)
    nonintersecting_lines = []
    for line in lines:
        added = False
        for line_set in nonintersecting_lines:
            if all(set(line).isdisjoint(set(line2)) for line2 in line_set):
                line_set.add(line)
                added = True
                break
        if not added:
            nonintersecting_lines.append({line})

    for line_set_pair in combinations(nonintersecting_lines, 2):
        found, pt_assignment = check_grid(pts, lines, grid_size, symmetric, line_set_pair)
        if found:
            return pt_assignment

    return None

def check_grid(pts, lines, grid_size, symmetric, line_set_pair):

    # estimate vanishing points
    vanishing_pts = []
    for line_set in line_set_pair:
        intersections = []
        for line_a, line_b in combinations(line_set, 2):
            intersection = np.cross(lines[line_a], lines[line_b])
            intersection /= intersection[2]
            intersections.append(intersection)

        intersections = np.stack(intersections)
        centroid = np.mean(intersections, axis=0)
        dists = np.linalg.norm(intersections - centroid, axis=1)
        intersections = np.delete(intersections, np.argmax(dists), axis=0)
        centroid = np.mean(intersections, axis=0)
        # probably better: use clustering, and select the largest cluster first

        vanishing_pts.append(centroid)

    # find a corner
    not_corner = set()
    while True:
        found = False
        for line_a in line_set_pair[0]:
            edge_idx = line_a[0]
            for line_b in line_set_pair[1]:
                if edge_idx in line_b:
                    if line_b[0] in not_corner: continue
                    corner_idx = line_b[0]
                    print(line_a, line_b)
                    found = True
                    break
            if found: break

        grid = -1 * np.ones((max(grid_size)+2, max(grid_size)+2))
        grid[0, 0] = corner_idx
        found_edge = False
        for line_a in line_set_pair[0]:
            if corner_idx in line_a:
                found_edge = True
                start = line_a.index(corner_idx)
                if start < 2:
                    for i, idx in enumerate(line_a[start:]):
                        grid[i, 0] = idx
                else:
                    for i, idx in enumerate(reversed(line_a[:start+1])):
                        grid[i, 0] = idx
        if not found_edge:
            not_corner.add(corner_idx)
            continue

        for line_b in line_set_pair[1]:
            for edge_i, edge_idx in enumerate(grid[:, 0]):
                if edge_idx == -1: continue
                if edge_idx in line_b:
                    start = line_b.index(edge_idx)
                    if start < 2:
                        for i, idx in enumerate(line_b[start:]):
                            grid[edge_i, i] = idx
                    else:
                        for i, idx in enumerate(reversed(line_b[:start+1])):
                            grid[edge_i, i] = idx

        print(grid)
        exit()


    #corner_idx = None
    #long_edge = None
    #short_edge = None
    #long_edge_lines = filter(sorted_lines, lambda l: len(l) == max(large_grid))[0]
    #for selected_line in long_edge_lines:
    #    short_edge = None
    #    for line in sorted_lines:
    #        if selected_line[0] in line and not any(i in line for i in selected_line[1:]):
    #            short_edge = line
    #            break
    #    if short_edge is None: continue
    #    if short_edge[0] == selected_line[0] or short_edge[-1] == selected_line[0]:
    #        corner_idx = short_edge[0]
    #        long_edge = selected_line





#def outlier_rejection(line, pts):
#    """points that are equally-spaced in 3d will be imaged to points where the distance between points forms a geometric
#    sequence (constant ratio), so find and reject points that break this pattern"""
#
#    # project points onto line and convert to 1-d coordinate along line
#    one_d_coord = np.sort(np.dot(pts, line))
#
#    diffs = one_d_coord[1:] - one_d_coord[:-1]
#    log_ratios = np.log(diffs[1:] / diffs[:-1])
#    outliers = []
#    for i in range(pts.shape[0]-2):
#        pass

def cross_ratio(pts, line):
    """check cross ratio of four points on line"""
    assert pts.shape == (4, 3)
    one_d_coords = pts[:, :2] @ np.array([[0, 1], [-1, 0]]) @ line[:2]
    pts = pts[np.argsort(one_d_coords), :]
    return (np.linalg.det(pts[[0, 1], :2]) * np.linalg.det(pts[[2, 3], :2])) / \
           (np.linalg.det(pts[[0, 2], :2]) * np.linalg.det(pts[[1, 3], :2]))

def reject_outliers(pts, lines):
    inlier_votes = np.zeros((pts.shape[0],), dtype=int)
    outlier_votes = np.zeros((pts.shape[0],), dtype=int)
    for pt_set, line in lines.items():
        if len(pt_set) >= 4:
            for pt_subset in combinations(pt_set, 4):
                if abs(cross_ratio(pts[pt_subset, :], line) - 1/4) < 1e-3:
                    inlier_votes[list(pt_subset)] += 1
                else:
                    outlier_votes[list(pt_subset)] += 1
    print(inlier_votes)
    print(outlier_votes)
    print(inlier_votes*10 - outlier_votes)

if __name__ == "__main__":

    asymm_grid_size = (10, 4)
    grid_size = (5, 4)

    H = np.array([[ 0.6,  0.0, -0.4],
                  [ 0.0,  0.8,  0.3],
                  [ 0.0,  0.2,  1.0]])
    theta = 0.1
    H @= np.array([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]])

    K1 = -2e-4
    K2 = 1e-6

    num_noise_pts = 5
    noise_limit = 3
    ex_dist = 0.1
    sigma = 5e-5

    symm_grid = add_detection_noise(
            distort_grid(
                add_noise_points(
                    gen_grid_pts(True, grid_size), num_noise_pts,
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

    symm_lines = find_lines(symm_grid, 4)
    asymm_lines = find_lines(asymm_grid, 4)
    reject_outliers(symm_grid, symm_lines)
    reject_outliers(asymm_grid, asymm_lines)
    #find_grids(symm_grid, symm_lines, True, grid_size)

    #for line in asymm_lines.items():
    #    if len(line[0]) < 6: continue
    #    print(line[1])
    #    print(asymm_grid[list(line[0]), :])
    #    exit()

    plt.subplot(1, 2, 1)
    plt.plot(symm_grid[:-num_noise_pts, 0], symm_grid[:-num_noise_pts, 1], 'o')
    plt.plot(symm_grid[-num_noise_pts:, 0], symm_grid[-num_noise_pts:, 1], 'o')
    for i, pt in enumerate(symm_grid):
        plt.gca().annotate(str(i), pt[:2], fontsize="x-small")
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    left_edge = np.array([1, 0, -xlim[0]])
    right_edge = np.array([1, 0, -xlim[1]])
    for line in symm_lines.items():
        left_pt = np.cross(left_edge, line[1])
        left_pt /= left_pt[2]
        right_pt = np.cross(right_edge, line[1])
        right_pt /= right_pt[2]
        plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], linewidth=0.5)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.gca().set_aspect('equal')
    plt.subplot(1, 2, 2)
    plt.plot(asymm_grid[:-num_noise_pts, 0], asymm_grid[:-num_noise_pts, 1], 'o')
    plt.plot(asymm_grid[-num_noise_pts:, 0], asymm_grid[-num_noise_pts:, 1], 'o')
    for i, pt in enumerate(asymm_grid):
        plt.gca().annotate(str(i), pt[:2], fontsize="x-small")
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    left_edge = np.array([1, 0, -xlim[0]])
    right_edge = np.array([1, 0, -xlim[1]])
    for line in asymm_lines.items():
        left_pt = np.cross(left_edge, line[1])
        left_pt /= left_pt[2]
        right_pt = np.cross(right_edge, line[1])
        right_pt /= right_pt[2]
        plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], linewidth=0.5)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.gca().set_aspect('equal')
    plt.show()

exit()
#centroid = np.mean(dgrid, axis=0)
#c_idx = np.random.randint(dgrid.shape[0]) #np.argmin(np.linalg.norm(dgrid - centroid, axis=1))
#c = dgrid[c_idx, :]
#
#lines = []
#used = {c_idx}
#for i in range(dgrid.shape[0]):
#    if i in used: continue
#    line = np.cross(c, dgrid[i, :])
#    line /= np.linalg.norm(line[:2])
#    num_pts = 2
#    used.add(i)
#    for j in range(dgrid.shape[0]):
#        if j in (i, c_idx): continue
#        if abs(np.dot(line, dgrid[j, :])) <= 1:
#            used.add(j)
#            num_pts += 1
#    if num_pts >= min(asymm_grid_size[0]//2, asymm_grid_size[1]):
#        lines.append((num_pts, line))
#
#lines.sort(key=lambda l: l[0])
#left_edge = np.cross(np.array([-5, -5, 1]), np.array([-5, 5, 1]))
#right_edge = np.cross(np.array([5, -5, 1]), np.array([5, 5, 1]))
#for _, line in lines:
#    left_pt = np.cross(line, left_edge)
#    left_pt /= left_pt[2]
#    right_pt = np.cross(line, right_edge)
#    right_pt /= right_pt[2]
#    plt.plot([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], "-")
#
#plt.show()

acc_size = (45, 200)

acc = np.zeros(acc_size)

blur = 5

for i in range(dgrid.shape[0]):
    pt = dgrid[i, :]
    for theta_i in range(acc_size[0]):
        theta = pi * theta_i / acc_size[0]
        rho = np.dot(pt[:2], [cos(theta), sin(theta)]) + 7
        rho_i = int((rho/14) * acc_size[1])
        for offset in range(-(blur//2), (blur//2)+1):
            if not 0 <= rho_i + offset < acc_size[1]: continue
            acc[theta_i, rho_i+offset] += 1

ax2.imshow(acc.T)

n_lines = 40
for i_line in range(n_lines):
    # get strongest line, zero out it and its neighbors
    (theta_i, rho_i) = np.unravel_index(acc.argmax(), acc.shape)
    for off_th, off_rh in product(range(-blur//2, blur//2 + 1), repeat=2):
        if not (0 <= rho_i + off_rh < acc_size[1] and 0 <= theta_i + off_th < acc_size[0]): continue
        acc[theta_i+off_th, rho_i+off_rh] = 0
    theta = pi * theta_i / acc_size[0]
    rho = rho_i/acc_size[1] * 14 - 7
    line = np.array([cos(theta), sin(theta), -rho])

    pts = []
    for i in range(dgrid.shape[0]):
        if abs(np.dot(line, dgrid[i, :])) < 1e-1:
            pts.append(dgrid[i, :])
    if len(pts) < 3: continue

    # re-fit line
    _, _, v = np.linalg.svd(np.stack(pts))
    line = v[-1, :]
    line /= np.linalg.norm(line[:2])

    # estimate vanishing point from each line

    # plot line
    if pi/4 < theta < 3*pi/4:
        pt1 = np.cross(line, np.array([1, 0, -5]))
        pt2 = np.cross(line, np.array([1, 0, 5]))
    else:
        pt1 = np.cross(line, np.array([0, 1, -5]))
        pt2 = np.cross(line, np.array([0, 1, 5]))
    pt1 /= pt1[2]
    pt2 /= pt2[2]
    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '-', linewidth=0.5,
             color=colormaps['viridis'].colors[255 - 256*i_line//n_lines])

ax3.imshow(acc.T)
plt.show()
