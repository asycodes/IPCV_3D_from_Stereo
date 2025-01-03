'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True

def o3d_2_nparray(img):
    '''
    Change from open3d images to numpy array
    '''
    img = np.asarray(img)
    if len(img.shape) > 2:
        # rgb 2 bgr
        img = img[..., ::-1]
    img = np.ascontiguousarray(img)   # make it contiguously stored in memory, otherwise errors triggered when drawing circles with cv2.
    return img


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sephere with random radius
        size = random.randrange(10, 14, 2) / 10.
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = 6
        x = random.randrange(-h/2+2, h/2-2, step)
        z = random.randrange(-w/2+2, w/2-2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
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
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


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
    theta = np.pi * 45*5/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * 80/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * 45*5/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)

    #########################################


    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

    # Rendering RGB-D frames given camera poses
    render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    for m, rgb, name in zip(obj_meshes, RGB_list, name_list):
        colour = o3d.visualization.rendering.MaterialRecord()
        colour.base_color = [rgb[0], rgb[1], rgb[2], 1.0]
        colour.shader = "defaultLit"
        render.scene.add_geometry(name, m, colour)
    
    render.setup_camera(K, H0_wc)
    img0 = o3d_2_nparray(render.render_to_image())
    cv2.imwrite('view0.png', img0)
    dep0 = o3d_2_nparray(render.render_to_depth_image(z_in_view_space=True))
    cv2.imwrite('depth0.png', dep0)
    render.setup_camera(K, H1_wc)
    img1 = o3d_2_nparray(render.render_to_image())
    cv2.imwrite('view1.png', img1)
    dep1 = o3d_2_nparray(render.render_to_depth_image(z_in_view_space=True))
    cv2.imwrite('depth1.png', dep1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()


    ###################################
    '''
    Task 3: Circle detection
    Hint: check cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################
    # Task 3: Circle Detection using HoughCircles
    def detect_circles(image_path, output_path, dp=1.2, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=50):
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
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2) #REDUCE NOISE

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        detected_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                detected_circles.append((x, y, r))
                cv2.circle(img, (x, y), r, (0, 255, 0), 2) #CIRCLE
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3) #CENTER

        cv2.imwrite(output_path, img)
        return detected_circles


    view0_circles = detect_circles(
        "view0.png", "view0_circles.png", dp=1.2, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=50
    )
    view1_circles = detect_circles(
        "view1.png", "view1_circles.png", dp=1.2, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=50
    )

    print("View 0 Circles:", view0_circles)
    print("View 1 Circles:", view1_circles)

    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################



def draw_epipolar_lines(image1, image2, points1, points2, fundamental_matrix):
    """
    Draw epipolar lines on two images given corresponding points and the fundamental matrix.
    INPUT:
        image1: First image (numpy array).
        image2: Second image (numpy array).
        points1: Corresponding points in the first image.
        points2: Corresponding points in the second image.
        fundamental_matrix: Fundamental matrix computed from the point correspondences.
    OUTPUT:
        img1_with_lines, img2_with_lines: image with epipolar
    """
    points1 = np.int32(points1)
    points2 = np.int32(points2)
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img1_with_lines = image1.copy()

    for r, pt in zip(lines1, points1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1_with_lines.shape[1], -(r[2] + r[0] * img1_with_lines.shape[1]) / r[1]])
        cv2.line(img1_with_lines, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1_with_lines, tuple(pt), 5, color, -1)
    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    img2_with_lines = image2.copy()

    for r, pt in zip(lines2, points2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img2_with_lines.shape[1], -(r[2] + r[0] * img2_with_lines.shape[1]) / r[1]])
        cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img2_with_lines, tuple(pt), 5, color, -1)

    return img1_with_lines, img2_with_lines


def task4_epipolar_lines(view0_path, view1_path, view0_points, view1_points):
    """
    Task 4: FIND Fundamental Matrix and draw the epipolar lines
    INPUT:
        view0_path: Path to the first image.
        view1_path: Path to the second image.
        view0_points: List of points (x, y) in the first image.
        view1_points: List of points (x, y) in the second image.
    """
    img1 = cv2.imread(view0_path)
    img2 = cv2.imread(view1_path)
    points1 = np.array(view0_points, dtype=np.float32)
    points2 = np.array(view1_points, dtype=np.float32)
    fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    img1_with_lines, img2_with_lines = draw_epipolar_lines(img1, img2, points1, points2, fundamental_matrix)
    cv2.imwrite("view0_epilines.png", img1_with_lines)
    cv2.imwrite("view1_epilines.png", img2_with_lines)

    print("Fundamental Matrix:\n", fundamental_matrix)


view0_points = [(320, 240), (300, 200), (340, 220)] 
view1_points = [(310, 250), (290, 210), (330, 230)]
task4_epipolar_lines("view0.png", "view1.png", view0_points, view1_points)


    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################


    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################
    
    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
