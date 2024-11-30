"""
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
"""

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import argparse


"""
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
"""


def transform_points(points, H):
    """
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication

    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix

    return:
      new_points: Nx3 matrix with each row being a 3-D point
    """
    # compute pt_w = H * pt_c
    n, m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n, 1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3, :]
    new_points = new_points[:3, :].transpose()
    return new_points


def check_dup_locations(y, z, loc_list):
    for loc_y, loc_z in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == "__main__":

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num", dest="num", type=int, default=6, help="number of spheres"
    )
    parser.add_argument(
        "--sph_rad_min",
        dest="sph_rad_min",
        type=int,
        default=10,
        help="min sphere  radius x10",
    )
    parser.add_argument(
        "--sph_rad_max",
        dest="sph_rad_max",
        type=int,
        default=16,
        help="max sphere  radius x10",
    )
    parser.add_argument(
        "--sph_sep_min",
        dest="sph_sep_min",
        type=int,
        default=4,
        help="min sphere  separation",
    )
    parser.add_argument(
        "--sph_sep_max",
        dest="sph_sep_max",
        type=int,
        default=8,
        help="max sphere  separation",
    )
    parser.add_argument(
        "--display_centre",
        dest="bCentre",
        action="store_true",
        help="open up another visualiser to visualise centres",
    )
    parser.add_argument("--coords", dest="bCoords", action="store_true")

    args = parser.parse_args()

    if args.num <= 0:
        print("invalidnumber of spheres")
        exit()

    if args.sph_rad_min >= args.sph_rad_max or args.sph_rad_min <= 0:
        print("invalid max and min sphere radii")
        exit()

    if args.sph_sep_min >= args.sph_sep_max or args.sph_sep_min <= 0:
        print("invalid max and min sphere separation")
        exit()

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=h, height=0.05, depth=w)
    box_H = np.array(
        [[1, 0, 0, -h / 2], [0, 1, 0, -0.05], [0, 0, 1, -w / 2], [0, 0, 0, 1]]
    )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ["plane"]
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f"sphere_{i}")

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2) / 10
        sph_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0.0, 0.5, 0.5])

        # create random sphere location
        step = random.randrange(int(args.sph_sep_min), int(args.sph_sep_max), 1)
        x = random.randrange(int(-h / 2 + 2), int(h / 2 - 2), step)
        z = random.randrange(int(-w / 2 + 2), int(w / 2 - 2), step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(int(-h / 2 + 2), int(h / 2 - 2), step)
            z = random.randrange(int(-w / 2 + 2), int(w / 2 - 2), step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.0]))
        GT_rads.append(size)
        sph_H = np.array([[1, 0, 0, x], [0, 1, 0, size], [0, 0, 1, z], [0, 0, 0, 1]])
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for mesh, H, rgb in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        obj_meshes = obj_meshes + [coord_frame]
        RGB_list.append([1.0, 1.0, 1.0])
        name_list.append("coords")

    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init
    # # placed at the world origin, and looking at z-positive direction,
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45 * 5 + random.uniform(-5, 5)) / 180.0
    H0_wc = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 20],
            [0, 0, 0, 1],
        ]
    )

    # camera_1 (world to camera)
    theta = np.pi * (80 + random.uniform(-10, 10)) / 180.0
    H1_0 = np.array(
        [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    theta = np.pi * (45 * 5 + random.uniform(-5, 5)) / 180.0
    H1_1 = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), -4],
            [0, np.sin(theta), np.cos(theta), 20],
            [0, 0, 0, 1],
        ]
    )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [
        (H0_wc, "view0.png", "depth0.png"),
        (H1_wc, "view1.png", "depth1.png"),
    ]

    #####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.

    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width = 640
    img_height = 480
    f = 415  # focal length
    # image centre in pixel coordinates
    ox = img_width / 2 - 0.5
    oy = img_height / 2 - 0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, f, f, ox, oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for H_wc, name, dname in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam, True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
    ##################################################

    # load in the images for post processings
    img0 = cv2.imread("view0.png", -1)
    dep0 = cv2.imread("depth0.png", -1)
    img1 = cv2.imread("view1.png", -1)
    dep1 = cv2.imread("depth1.png", -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1.0, 0.0, 0.0])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    ###################################
    """
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    """

    ###################################
    def circle_detection(
        img,
        output_path,
        dp=1.2,
        min_dist=30,
        param1=50,
        param2=30,
        min_radius=10,
        max_radius=50,
    ):
        """
        Detect circles using Hough Circle Transform.
        INPUT:
            image_path: Path of the input image.
            output_path: Path to save the output image.
            dp: Inverse ratio of the accumulator resolution to the image resolution.
            min_dist: Minimum distance between the centers of detected circles.
            param1: Gradient value for edge detection.
            param2: Accumulator threshold for circle centers.
            min_radius: Minimum radius of detected circles.
            max_radius: Maximum radius of detected circles.
        OUTPUT:
            detected_circles: List of detected circles [(x_center, y_center, radius), ...].
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)  # REDUCE NOISE
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        detected_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                detected_circles.append((x, y, r))
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)  # CIRCLE
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)  # CENTER

        cv2.imwrite(output_path, img)
        return detected_circles

    view0_circles = circle_detection(
        img0,
        "view0_circles.png",
        dp=1,
        min_dist=20,
        param1=60,
        param2=30,
        min_radius=10,
        max_radius=70,
    )
    view1_circles = circle_detection(
        img1,
        "view1_circles.png",
        dp=1,
        min_dist=20,
        param1=60,
        param2=30,
        min_radius=10,
        max_radius=70,
    )

    print("View 0 Circles:", view0_circles)
    print("View 1 Circles:", view1_circles)

    ###################################
    """
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
            Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2

    Write your code here
    """

    ###################################
    def compute_relative_pose(H0_wc, H1_wc):
        """
        Compute the relative rotation matrix (R) and translation vector (T) between two cameras.
        """
        R0 = H0_wc[:3, :3]
        T0 = H0_wc[:3, 3]

        R1 = H1_wc[:3, :3]
        T1 = H1_wc[:3, 3]
        R = R1 @ R0.T
        T = T1 - R @ T0

        return R, T

    def compute_essential_matrix(R, T):
        """
        Compute the essential matrix E = [T]_x R.
        """
        T_x = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
        E = T_x @ R
        return E

    def draw_epipolar_lines(img, points, E, K, reference_view):
        """
        Draw epipolar lines on the image given points in the reference view.
        """
        F = (
            np.linalg.inv(K).T @ E @ np.linalg.inv(K)
        )  # Fundamental matrix from Essential matrix
        h, w = img.shape[:2]
        print(points)
        for point in points:
            x, y, r = point
            line = F @ np.array([x, y, 1]).T
            a, b, c = line

            x_start, y_start = 0, int(-c / b) if b != 0 else 0
            x_end, y_end = w, int(-(c + a * w) / b) if b != 0 else h

            cv2.line(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        return img

    K = np.array(
        [
            [f, 0, ox],
            [0, f, oy],
            [0, 0, 1],
        ]
    )

    R, T = compute_relative_pose(H0_wc, H1_wc)
    E = compute_essential_matrix(R, T)
    view1_img = cv2.imread("view1.png")
    view1_with_lines = draw_epipolar_lines(
        view1_img, view0_circles, E, K, reference_view="view0"
    )

    # Save or display the output
    cv2.imwrite("view1_with_epipolar_lines.png", view1_with_lines)

    ###################################
    """
    Task 5: Find correspondences

    Write your code here
    """

    ###################################
    def find_corresponding_points(F):
        """
        Use epipolar lines to find corresponding points between two images.
        """
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        refined_matches = []

        for i, k1 in enumerate(kp1):
            pt1 = np.array([k1.pt[0], k1.pt[1], 1])
            epiline = F @ pt1

            # Extract keypoints in image 2 and calculate distance to the epipolar line
            for j, k2 in enumerate(kp2):
                pt2 = np.array([k2.pt[0], k2.pt[1], 1])
                # Calculate the distance of pt2 to the epipolar line
                a, b, c = epiline
                d = abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a**2 + b**2)

                # Add match if distance is within threshold
                if d < 0.8:  # lower -> harsher, higher -> leanient
                    refined_matches.append(cv2.DMatch(i, j, d))

        return kp1, kp2, refined_matches

    def compute_fundamental_matrix(K, H0_wc, H1_wc):
        R1, t1 = H0_wc[:3, :3], H0_wc[:3, 3]
        R2, t2 = H1_wc[:3, :3], H1_wc[:3, 3]
        R = R2 @ R1.T
        t = t2 - R @ t1
        t_x = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        E = t_x @ R
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
        return F / F[2, 2]

    K = np.array(
        [
            [f, 0, ox],
            [0, f, oy],
            [0, 0, 1],
        ]
    )
    F = compute_fundamental_matrix(K, H0_wc, H1_wc)
    img1 = cv2.imread("view0.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("view1.png", cv2.IMREAD_GRAYSCALE)

    kp1, kp2, refined_matches = find_corresponding_points(F)
    output_img = cv2.drawMatches(img1, kp1, img2, kp2, refined_matches, None, flags=2)
    cv2.imwrite("refined_matches_epipolar.png", output_img)

    ###################################
    """
    Task 6: 3-D locations of sphere centres

    Write your code here
    """
    ###################################

    def reconstruct_3d_points(matches, kp1, kp2, R, T, K):
        """
        Reconstruct 3D points according to 3d reconstruction algorithm
        INPUT:
            matches: List of cv2.DMatch objects (refined matches).
            kp1: Keypoints from the first image.
            kp2: Keypoints from the second image.
            R: Rotation Matrix
            T: Translation Matrix
            K: Fundamental Matrix
        OUTPUT:
            points_3d: Nx3 numpy array of reconstructed 3D points.
        """
        points_3d = []
        for match in matches:
            # Extract matched keypoints
            pt1 = kp1[match.queryIdx].pt  # POINT FROM CAM 0
            pt2 = kp2[match.trainIdx].pt  # POINT FROM CAM 1

            # Convert from raw pixel coordinates to image coordinates
            p_L = np.linalg.inv(K) @ np.array([pt1[0], pt1[1], 1])
            p_R = np.linalg.inv(K) @ np.array([pt2[0], pt2[1], 1])

            # Following the equation: a * p_L - b * (R @ p_R) - T = 0
            H = np.column_stack([p_L, -(R.T @ p_R), -np.cross(p_L, R.T @ p_R)])

            # Solve for [a, b, c] such that H @ [a, b, c] = T
            # Use the pseudo-inverse to solve for a, b, c
            abc = np.linalg.lstsq(H, T, rcond=None)[0]  # Solves for a, b, c

            # Calculate 3D point P
            a, b, c = abc
            P = (a * p_L + b * (R.T @ p_R) + T) / 2

            points_3d.append(P)

        return np.array(points_3d)

    K = np.array(
        [
            [f, 0, ox],
            [0, f, oy],
            [0, 0, 1],
        ]
    )

    points_3d = reconstruct_3d_points(refined_matches, kp1, kp2, R, T, K)
    print("Reconstructed 3D Points:")
    print(points_3d)

    ###################################
    """
    Task 7: Evaluate and Display the centres

    Write your code here
    """
    ###################################

    def compute_errors(reconstructed_points, ground_truth_points):
        """
        Compute the error between reconstructed 3D points and ground truth points
        with 1-to-1 unique matching using Hungarian Algorithm.
        INPUT:
            reconstructed_points: Nx3 array of reconstructed 3D sphere centers.
            ground_truth_points: Mx3 array of ground truth 3D sphere centers.
        OUTPUT:
            matched_pairs: List of tuples (reconstructed_idx, ground_truth_idx) indicating the matches.
            errors: List of Euclidean distances (errors) for each matched point.
            mean_error: Mean error across all points.
            max_error: Maximum error.
            std_error: Standard deviation of the errors.
        """
        dist_matrix = distance_matrix(reconstructed_points, ground_truth_points)
        reconstructed_indices, ground_truth_indices = linear_sum_assignment(dist_matrix)

        matched_pairs = []
        errors = []
        for rec_idx, gt_idx in zip(reconstructed_indices, ground_truth_indices):
            matched_pairs.append((rec_idx, gt_idx))
            errors.append(dist_matrix[rec_idx, gt_idx])

        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)

        return matched_pairs, errors, mean_error, max_error, std_error

    ground_truth_centers = [cent[:3] for cent in GT_cents]
    pairs, errors, mean_error, max_error, std_error = compute_errors(
        points_3d, ground_truth_centers
    )

    for i, error in enumerate(errors):
        print(
            f"PAIR {ground_truth_centers[pairs[i][1]]} and {points_3d[pairs[i][0]]}: Error = {error:.4f} units"
        )

    print(f"Mean Error: {mean_error:.4f} units")
    print(f"Max Error: {max_error:.4f} units")
    print(f"Standard Deviation of Errors: {std_error:.4f} units")

    def visualize_centers(ground_truth, estimated):
        """
        Visualize ground truth and estimated sphere centers in 3D.
        INPUT:
            ground_truth (list): Ground-truth sphere centers [(x, y, z), ...].
            estimated (list): Estimated sphere centers [(x, y, z), ...].
        """
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(np.array(ground_truth))
        pcd_gt.paint_uniform_color([1, 0, 0])  # Red for ground-truth

        pcd_est = o3d.geometry.PointCloud()
        pcd_est.points = o3d.utility.Vector3dVector(np.array(estimated))
        pcd_est.paint_uniform_color([0, 1, 0])  # Green for estimated
        o3d.visualization.draw_geometries(
            [pcd_gt, pcd_est], window_name="Sphere Centers", point_show_normal=False
        )

    # Visualize centers
    visualize_centers(ground_truth_centers, points_3d)

    ###################################
    """
    Task 8: 3-D radius of spheres

    Write your code here
    """
    ###################################

    def estimate_sphere_radii(
        detected_circles_ref,
        detected_circles_view,
        reconstructed_points,
        focal_length,
    ):
        """
        Estimate the real-world radius of each sphere.
        INPUT:
            detected_circles_ref: List of detected circles in the reference image [(x, y, r), ...].
            detected_circles_view: List of detected circles in the viewing image [(x, y, r), ...].
            reconstructed_points: List of 3D reconstructed points [(x, y, z), ...].
            focal_length: Camera focal length.

        OUTPUT:
            radii: List of estimated radii of the spheres.
        """
        radii = []
        if len(detected_circles_ref) != len(reconstructed_points):
            raise ValueError(
                "Mismatch between detected circles and reconstructed points."
            )

        for circle_ref, circle_view, point_3d in zip(
            detected_circles_ref, detected_circles_view, reconstructed_points
        ):
            r_ref = circle_ref[2]  # Radius from reference image
            r_view = circle_view[2]  # Radius from viewing image

            # Extract depth (z-coordinate) from 3D reconstructed point
            z_ref = point_3d[2]
            z_view = point_3d[2]
            # Convert pixel radii to real-world radii using depth and focal length
            R_ref = r_ref * (z_ref / focal_length)
            R_view = r_view * (z_view / focal_length)

            # Average the radii estimates
            R_avg = (R_ref + R_view) / 2
            radii.append(R_avg)

        return radii

    focal_length = f
    detected_circles_ref = view0_circles
    detected_circles_view = view1_circles
    ### only want reconstructed SPHERES
    reconstructed_spheres = [pair[0] for pair in pairs]
    reconstructed_points = points_3d[reconstructed_spheres]

    # Estimate radii
    radii = estimate_sphere_radii(
        detected_circles_ref,
        detected_circles_view,
        reconstructed_points,
        focal_length,
    )

    # Print estimated radii
    for i, radius in enumerate(radii):
        print(f"Sphere {i + 1}: Estimated Radius = {radius:.4f} units")

    ###################################
    """
    Task 9: Display the spheres

    Write your code here:
    """
    ###################################

    ###################################
    """
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    """
    ###################################
