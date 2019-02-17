import cv2
import math
import sys
from math import sqrt
import csv
import skimage
from skimage.feature import blob_log
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from matplotlib import collections as mc
import seaborn as sns 
import os


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)

    return np.argmin(dist_2)


def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup


def onclick_add(event):
    global scat

    global x_list
    global y_list
    global xy_list

    x_list.append(int(event.xdata))
    y_list.append(int(event.ydata))
    xy_list.append([int(event.xdata), int(event.ydata)])

    scat.set_offsets(xy_list)
    fig.canvas.draw()


def onclick_remove(event):
    global scat

    global x_list
    global y_list
    global xy_list

    index_of_closest_point = closest_node((event.xdata, event.ydata), xy_list)

    del xy_list[index_of_closest_point]
    del x_list[index_of_closest_point]
    del y_list[index_of_closest_point]

    scat.set_offsets(xy_list)
    fig.canvas.draw()


def onclick_add_vertices(event):
    global scat2

    global vertices_x_list
    global vertices_y_list
    global vertices_xy_list

    vertices_x_list.append(float(event.xdata))
    vertices_y_list.append(float(event.ydata))
    vertices_xy_list.append([float(event.xdata), float(event.ydata)])

    scat2.set_offsets(vertices_xy_list)
    fig.canvas.draw()


def onclick_remove_vertices(event):
    global scat2

    global vertices_x_list
    global vertices_y_list
    global vertices_xy_list
    global list_of_connected_points
    global list_of_disconnected_points

    global green_lines

    temp_list_of_connected_points = []

    index_of_closest_point2 = closest_node((event.xdata, event.ydata), vertices_xy_list)

    ################################################################################################

    # Have to clear possible manual connected LINE points that user added as well so that there not an extra lines.
    # Not just vertices_xy_list.

    temp_list_of_connected_points = list_of_connected_points[:]

    if (len(temp_list_of_connected_points) != 0):

        for stuff in temp_list_of_connected_points:

            new_indice1 = vertices_xy_list.index(stuff[0])  # Pair of new indices that indicate line connection
            new_indice2 = vertices_xy_list.index(stuff[1])

            if ((new_indice1 == index_of_closest_point2) or (new_indice2 == index_of_closest_point2)):
                list_of_connected_points.remove(stuff)

    ################################################################################################

    del vertices_xy_list[index_of_closest_point2]
    del vertices_x_list[index_of_closest_point2]
    del vertices_y_list[index_of_closest_point2]

    scat2.set_offsets(vertices_xy_list)
    fig.canvas.draw()


def onclick_add_lines(event):
    global vertices_x_list
    global vertices_y_list
    global vertices_xy_list
    global list_of_connected_points
    global list_of_disconnected_points
    global line_collection
    global green_lines
    global temp_index
    global linepoints

    index_of_closest_point3 = closest_node((event.xdata, event.ydata), vertices_xy_list)

    temp_index.append(index_of_closest_point3)
    linepoints.append([vertices_xy_list[index_of_closest_point3][0],
                       vertices_xy_list[index_of_closest_point3][1]])  # Store actual VALUES not INDICES

    if (np.size(temp_index) == 2):

        tuple_index = tuple(temp_index)

        if tuple_index in green_lines:  # Line already exists

            print ("Line exists already. Try Again.\n")

            temp_index.clear()
            linepoints.clear()

        elif Reverse(tuple_index) in green_lines:  # Check other way

            print ("Line exists already. Try Again.\n")

            temp_index.clear()
            linepoints.clear()

        else:

            list_of_connected_points.append(
                [linepoints[0], linepoints[1]])  # Dont use indice values since vertices_xy_list length CHANGES...

            if ([linepoints[0], linepoints[1]]) in list_of_disconnected_points:
                list_of_disconnected_points.remove([linepoints[0], linepoints[1]])

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

            temp_index.clear()
            linepoints.clear()


def onclick_remove_lines(event):
    global vertices_x_list
    global vertices_y_list
    global vertices_xy_list
    global list_of_connected_points
    global list_of_disconnected_points
    global line_collection
    global green_lines
    global temp_index
    global linepoints

    index_of_closest_point4 = closest_node((event.xdata, event.ydata), vertices_xy_list)

    temp_index.append(index_of_closest_point4)
    linepoints.append([vertices_xy_list[index_of_closest_point4][0],
                       vertices_xy_list[index_of_closest_point4][1]])  # Store actual VALUES not INDICES

    if (np.size(temp_index) == 2):

        tuple_index = tuple(temp_index)

        if tuple_index in green_lines:  # Line already exists

            list_of_disconnected_points.append([linepoints[0], linepoints[1]])

            if ([linepoints[0], linepoints[1]]) in list_of_connected_points:
                list_of_connected_points.remove([linepoints[0], linepoints[1]])

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

            temp_index.clear()
            linepoints.clear()

        elif Reverse(tuple_index) in green_lines:  # Check other way

            list_of_disconnected_points.append([linepoints[0], linepoints[1]])

            if ([linepoints[0], linepoints[1]]) in list_of_connected_points:
                list_of_connected_points.remove([linepoints[0], linepoints[1]])

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

            temp_index.clear()
            linepoints.clear()

        else:

            print ("Line does not exist. Cannot Delete. Try Again.\n")

            temp_index.clear()
            linepoints.clear()


def calculate_vertice_lines():
    global green_lines
    global vertices_xy_list
    global list_of_disconnected_points
    global list_of_connected_points
    global cell_wall_lengths

    data = np.asarray(vertices_xy_list)

    del cell_wall_lengths[:]

    tree = spatial.cKDTree(data)
    g = {i: set(tree.query(data[i, :], 4)[-1][1:]) for i in range(data.shape[0])}

    for node, candidates in g.items():
        for node2 in candidates:

            if node2 < node:
                # avoid double-counting
                continue

            if (len(list_of_disconnected_points) != 0):

                if ([[vertices_xy_list[node2][0], vertices_xy_list[node2][1]],
                     [vertices_xy_list[node][0], vertices_xy_list[node][1]]]) in list_of_disconnected_points:
                    continue

                if ([[vertices_xy_list[node][0], vertices_xy_list[node][1]],
                     [vertices_xy_list[node2][0], vertices_xy_list[node2][1]]]) in list_of_disconnected_points:
                    continue

            if node in g[node2] and spatial.distance.euclidean(data[node, :], data[node2, :]) < 55:
                green_lines.append((node, node2))  # Nodes are indices of vertices_xy_list.
                cell_wall_lengths.append(spatial.distance.euclidean(data[node, :], data[node2, :]))

    if (len(list_of_connected_points) != 0):

        for stuff in list_of_connected_points:
            new_indice1 = vertices_xy_list.index(stuff[0])  # Pair of new indices that indicate line connection
            new_indice2 = vertices_xy_list.index(stuff[1])

            green_lines.append((new_indice1, new_indice2))
            cell_wall_lengths.append(spatial.distance.euclidean(data[new_indice1], data[new_indice2]))


def draw_update_vertice_lines():
    global line_collection
    global green_lines
    global vertices_xy_list
    global axes

    lines = [[vertices_xy_list[i], vertices_xy_list[j]] for i, j in green_lines]
    line_collection = mc.LineCollection(lines, color='green')
    axes.add_collection(line_collection)

    fig.canvas.draw()


xy_tuplelist = []

x_list = []
y_list = []
xy_list = []

vertices_x_list = []
vertices_y_list = []
vertices_xy_list = []

cell_wall_lengths = []

list_of_lines_references = []
list_of_connected_points = []
list_of_disconnected_points = []  # Tuples of indices of points that should be disconnected.

temp_index = []
linepoints = []

green_lines = []

angles = []
wall_len = []
areas = []

# groundtruth_x, groundtruth_y = Read_Two_Column_File('hc_junction_test_groundtruth.txt')

user_input_menu1 = input(
    "\nType 1, 2, or 3 to choose from the options below.\n1. Automated Blob Detection Image Analysis Mode.\n2. Old Image Data Mode.\n3. BLANK Slate Mode.\nEnter Input: ")

##################################################################################################################################################################################################################################################################

if (user_input_menu1 == '1'):

    print (
    "\n################################## Automated Blob Detection Image Analysis Mode ##################################")

    inputfile = input("\nEnter input image name: ")

    image = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE)

    image2 = rgb2gray(image)

    image_gray = skimage.util.invert(image2)

    user_input1 = input(
        "\nEnter DESIRED thresholding method\n ~ Adaptive Gaussian ('ag')\n ~ Adaptive Mean ('am')\n ~ Binary ('b')\n ~ No Threshold ('no')\nEnter Input: ")

    ################################################ Different Thresholding Methods ################################################

    if (user_input1 == 'b'):

        retval, image_g = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)

    elif (user_input1 == 'ag'):

        image_g = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif (user_input1 == 'am'):

        image_g = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    elif (user_input1 == 'no'):

        image_g = image_gray

    else:

        sys.exit()

    ################################################################################################################################

    # image_g = exposure.equalize_hist(image_g)

    blobs_log = blob_log(image_g, min_sigma=23, max_sigma=25, num_sigma=10, threshold=.01,
                         overlap=0.1)  # Slows down the program a lot in this 1 line.

    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)  # Compute radii in the 3rd column.

    fig, axes = plt.subplots()

    axes.set_title('LoG Detection - ' + user_input1)
    plt.xticks([]), plt.yticks([])
    axes.imshow(image, interpolation='nearest', cmap='gray')

    for blob in blobs_log:

        y, x, r = blob

        y = int(y)
        x = int(x)

        intensity_value = int(image2[y][x])

        if (intensity_value > 20):

            x_list.append(x)
            y_list.append(y)

            xy_list.append([x, y])

        # plt.plot(x, y, 'red')
        # c = plt.Circle((x, y), r, color='blue', linewidth=1, fill=False)
        # axes.add_patch(c)

        else:

            pass

    ############################################################################################################################################

    scat = plt.scatter(x_list, y_list, color='red')

    fig.show()
    fig.canvas.draw()

    while True:

        print ("\n#############################################################################################")
        print ("#############################################################################################")
        print ("\nUser Blob RED Point Adjustment Interface\n")
        print ("Only Interact with the plot (e.g. Zoom and Pan) when not in Add/Remove points mode.\n")
        print ("Please type 1, 2, 3, or 4 to select from the options below.\n")
        print ("1. Manually Add Points\n")
        print ("2. Manually Remove Points\n")
        print ("3. Save Blob Center Points Data\n")
        print ("4. Done \n")

        user_input_interface = input(":")

        if (user_input_interface == '1'):

            print("\nPlease click to add points.")

            cid = fig.canvas.mpl_connect('button_press_event', onclick_add)

            user_input2 = input("\nPress ENTER to move on.")

            fig.canvas.mpl_disconnect(cid)

        elif (user_input_interface == '2'):

            print("\nPlease click to remove points.")

            cid2 = fig.canvas.mpl_connect('button_press_event', onclick_remove)

            user_input2 = input("\nPress ENTER to move on.")

            fig.canvas.mpl_disconnect(cid2)

        elif (user_input_interface == '3'):

            user_input_filename = input("Please enter the desired blob file name: ")

            file_blob_new = open(user_input_filename, "w")

            for f, b in zip(x_list, y_list):
                file_blob_new.write(str(f))
                file_blob_new.write(",")
                file_blob_new.write(str(b))
                file_blob_new.write("\n")

            file_blob_new.close()

        elif (user_input_interface == '4'):

            break

        else:

            print ("\nIncorrect input. Please try again.")

    ############################################################################################################################################

    PATH = inputfile
    image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    image2 = rgb2gray(image)
    thresh1 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
    kernel1 = np.ones((6, 6), np.uint8)
    img = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    min_size = 250

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    kernel2 = np.ones((5, 5), np.uint8)
    image_g = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel2)

    all_vertex = []
    at = []
    # all_points = list(xy_list)
    #angles = []
    #wall_len = []
    #areas = []
    sobelx = cv2.Sobel(image_g, cv2.CV_64F, 1, 0)  # Find x and y gradients  image_g
    sobely = cv2.Sobel(image_g, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)

    for i in range(len(xy_list)):
        x = xy_list[i][0]  # for this point
        y = xy_list[i][1]

        distance = []

        around_cells = []

        for d in xy_list:
            xi = d[0]  # k
            yi = d[1]
            distance.append(((x - xi) ** 2 + (y - yi) ** 2) ** 0.5)

        sorted_distance = np.argsort(distance)

        for j in range(7):
            around_cells.append(xy_list[sorted_distance[j]])

        vertex = []
        wall = []
        points = []
        max_temp = []
        center = around_cells.pop(0)

        for c in around_cells:
            if (center[0] - c[0] != 0):
                k = (center[1] - c[1]) / (center[0] - c[0])
                b = center[1] - k * center[0]
            else:
                k = 10000
                b = 10000

            max_gradient = 0
            final = []
            line = []
            maxline = []

            ind = 0
            ind1 = 0

            for m in range(min(center[0], c[0]), max(center[0], c[0])):  # x
                for n in range(min(center[1], c[1]), max(center[1], c[1])):  # y
                    if (n == round(k * m + b)):
                        l = [m, n]
                        line.append(l)
                        ind = ind + 1

                        if (max_gradient < magnitude[n][m]):
                            max_gradient = magnitude[n][m]
                            ind1 = ind

            ind2 = ind1
            ind3 = ind1 - 1

            while (ind2 <= len(line) - 1):

                if image_g[line[ind2][1], line[ind2][0]] == 255:
                    maxline.append(line[ind2])

                ind2 = ind2 + 1

            while (ind3 >= 0):

                if image_g[line[ind3][1], line[ind3][0]] == 255:
                    maxline.insert(0, line[ind3])

                ind3 = ind3 - 1

            if (len(maxline) != 0):
                leng = len(maxline)
                r = int(leng / 2)

                final = maxline[r]
                if (len(final) != 0):
                    if (k == 0):
                        k_1 = 10000
                        b_1 = (float)(final[1] - k_1 * final[0])
                    else:
                        k_1 = (float)(-1 / (k))
                        b_1 = (float)(final[1] - k_1 * final[0])

                points.append(final)
                w = [k_1, b_1]
                wall.append(w)

        # print(len(points))
        #del angles[:]
        for slop in wall:
            sk = slop[0]
            sb = slop[1]

            for slop2 in wall:
                sk2 = slop2[0]
                sb2 = slop2[1]

                a = np.abs(180 * math.atan(sk) / np.pi - 180 * math.atan(sk2) / np.pi)
                if (a > 100 and a < 150):
                    angles.append(a)


        for p1 in points:
            dis = []
            i1 = points.index(p1)
            k0 = wall[i1][0]
            b0 = wall[i1][1]
            for p2 in points:
                d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                dis.append(d)

            sorted_dis = np.argsort(dis)

            if (len(sorted_dis) >= 3):
                index1 = sorted_dis[1]
                index2 = sorted_dis[2]

                k1 = wall[index1][0]
                b1 = wall[index1][1]
                k2 = wall[index2][0]
                b2 = wall[index2][1]

                if (k0 != k1):
                    vx1 = (b1 - b0) / (k0 - k1)
                    vy1 = k1 * vx1 + b1
                    v1 = [vx1, vy1]
                # print(v1)
                if v1 not in all_vertex:
                    if (v1[0] > 0 and v1[1] > 0 and v1[0] < image_g.shape[1] and v1[1] < image_g.shape[0] and image_g[
                        round(v1[1]), round(v1[0])] == 255):
                        all_vertex.append(v1)

                if (k0 != k2):
                    vx2 = (b2 - b0) / (k0 - k2)
                    vy2 = k2 * vx2 + b2
                    v2 = [vx2, vy2]
                # print(v2)
                if v2 not in all_vertex:
                    if (v2[0] > 0 and v2[1] > 0 and v2[0] < image_g.shape[1] and v2[1] < image_g.shape[0] and image_g[
                        round(v2[1]), round(v2[0])] == 255):
                        all_vertex.append(v2)

    for m in all_vertex:
        temp = []
        tempx = []
        tempy = []
        temp.append(m)
        tempx.append(m[0])
        tempy.append(m[1])
        for n in all_vertex:
            if (((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2) ** 0.5 < 20):
                temp.append(n)
                tempx.append(n[0])
                tempy.append(n[1])
                all_vertex.remove(n)

        vertices_x_list.append(np.mean(tempx))
        vertices_y_list.append(np.mean(tempy))
        vertices_xy_list.append([np.mean(tempx), np.mean(tempy)])
        




    scat2 = plt.scatter(vertices_x_list, vertices_y_list, color='blue')

    fig.canvas.draw()

    if (len(xy_list) < 6 or len(vertices_xy_list) < 4):
        print ("\nThere were either less than 6 RED center points or less than 4 BLUE vertice points. \n")
        print ("Please Try Again.")
        plt.close()
        sys.exit()

    ############################################################################################################################################
    # Calculate first cycle of green lines.

    calculate_vertice_lines()  # Drawing the actual green lines.

    # Initialize first automatic set of lines and line_collection.
    lines = [[vertices_xy_list[i], vertices_xy_list[j]] for i, j in green_lines]
    line_collection = mc.LineCollection(lines, color='green')
    axes.add_collection(line_collection)

    fig.canvas.draw()

    ############################################################################################################################################

    while True:

        print ("\n#############################################################################################")
        print ("\n*** Please do NOT go below 6 RED center points or 4 BLUE vertice points. ***\n")
        print ("#############################################################################################")
        print ("\nUser Vertices Line Adjustment Interface\n")
        print ("Adjustment of lines (GREEN) through adjustment of vertice points (BLUE).\n")
        print ("Only Interact with the plot (e.g. Zoom and Pan) when not in Add/Remove points mode.\n")
        print ("Please type 1, 2, 3, or 4 to select from the options below.\n")
        print ("1. Manually Add Points\n")
        print ("2. Manually Remove Points\n")
        print ("3. Manually Add Lines\n")
        print ("4. Manually Remove Lines\n")
        print ("5. Save Line and Vertice Data\n")
        print ("6. Done \n")

        user_input_interface2 = input(":")

        if (user_input_interface2 == '1'):

            print("\nPlease click to add points.")

            cid_vert = fig.canvas.mpl_connect('button_press_event', onclick_add_vertices)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert)

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

        elif (user_input_interface2 == '2'):

            print("\nPlease click to remove points.")

            cid_vert2 = fig.canvas.mpl_connect('button_press_event', onclick_remove_vertices)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert2)

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

        elif (user_input_interface2 == '3'):

            print("\nPlease click 2 BLUE points to add lines.")

            cid_vert3 = fig.canvas.mpl_connect('button_press_event', onclick_add_lines)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert3)

        elif (user_input_interface2 == '4'):

            print("\nPlease click 2 BLUE points to remove lines.")

            cid_vert4 = fig.canvas.mpl_connect('button_press_event', onclick_remove_lines)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert4)

        elif (user_input_interface2 == '5'):

            vertices_x_list.clear()
            vertices_y_list.clear()

            vertices_x_list, vertices_y_list = map(list, zip(*vertices_xy_list))

            ####################################################################################################

            user_input_filename = input("Please enter the desired vertices file name: ")

            file_vertices_new = open(user_input_filename, "w")

            for f, b in zip(vertices_x_list, vertices_y_list):
                # Saves COORDINATE pairs of vertices_xy_list.
                file_vertices_new.write(str(f))
                file_vertices_new.write(",")
                file_vertices_new.write(str(b))
                file_vertices_new.write("\n")

            file_vertices_new.close()

            ####################################################################################################

            user_input_filename = input("Please enter the desired lines file name: ")

            file_lines_new = open(user_input_filename, "w")

            for (f, b) in green_lines:
                # Saves tuple of INDICES of all points on vertices_xy_list that should be connected.
                file_lines_new.write(str(f))
                file_lines_new.write(",")
                file_lines_new.write(str(b))
                file_lines_new.write("\n")

            file_lines_new.close()

            user_input_filename = input("Please enter the desired removed lines file name: ")

            file_removed_lines_new = open(user_input_filename, "w")

            for f, b in list_of_disconnected_points:
                file_removed_lines_new.write(str(f[0]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(f[1]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(b[0]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(b[1]))
                file_removed_lines_new.write("\n")

            file_removed_lines_new.close()

            user_input_filename = input("Please enter the desired added lines file name: ")

            file_added_lines_new = open(user_input_filename, "w")

            for f, b in list_of_connected_points:
                file_added_lines_new.write(str(f[0]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(f[1]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(b[0]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(b[1]))
                file_added_lines_new.write("\n")

            file_added_lines_new.close()

        ####################################################################################################

        elif (user_input_interface2 == '6'):
            print ("Do you wnat to see distribution map [y/n] \n")
            user_input = input(":")
            if (user_input == 'n'):
                break
            elif (user_input == 'y'):
                #del wall_len[:]
                for p1 in vertices_xy_list:
                    for p2 in vertices_xy_list:
                        b = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                        if (b > 25 and b < 50):
                            wall_len.append(b)
                            
                print(len(wall_len))
                print(len(angles))  
                
                for c in wall_len:
                    a=3*(3**0.5)*(c**2)/2
                    areas.append(a)
                    
                print(len(areas))
                

                plt.figure(7)
                sns.distplot(angles, rug=True)
                plt.xlabel("degree")
                plt.ylabel("percentage")
                plt.title("angular distribution diagram")
                print(7)

                plt.figure(8)
                sns.distplot(wall_len, rug=True)
                plt.xlabel("wall length")
                plt.ylabel("percentage")
                plt.title("wall length distribution diagram")
                print(8)
                
                plt.figure(9)
                sns.distplot(areas, rug=True)
                plt.xlabel("area")
                plt.ylabel("percentage")
                plt.title("area distribution diagram")
                print(9)
                plt.show()
                #sys.exit()
                #break
            break

        else:

            print ("\nIncorrect input. Please try again.")

    print ("\nSaving cell wall pixel lengths in cell_wall_lengths.csv")

    file_cellwall_lengths = open("cell_pixel_wall_lengths.csv", "w")

    for f in cell_wall_lengths:
        file_cellwall_lengths.write(str(f))
        file_cellwall_lengths.write("\n")
        
    file_cellwall_lengths.close()
    fig.canvas.draw()

##################################################################################################################################################################################################################################################################

if (user_input_menu1 == '2'):

    print ("\n################################## Old Image Data Mode ##################################")

    user_input_menu1_1 = input("\nEnter Old Image File: ")

    image = cv2.imread(user_input_menu1_1, cv2.IMREAD_GRAYSCALE)

    image2 = rgb2gray(image)

    image_gray = skimage.util.invert(image2)

    user_input1 = input(
        "\nEnter OLD thresholding method\n ~ Adaptive Gaussian ('ag')\n ~ Adaptive Mean ('am')\n ~ Binary ('b')\n ~ No Threshold ('no')\nEnter Input: ")

    ################################################ Different Thresholding Methods ################################################

    if (user_input1 == 'b'):

        retval, image_g = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)

    elif (user_input1 == 'ag'):

        image_g = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif (user_input1 == 'am'):

        image_g = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    elif (user_input1 == 'no'):

        image_g = image_gray

    else:

        sys.exit()

    ################################################################################################################################

    fig, axes = plt.subplots()

    axes.set_title('LoG Detection - ' + user_input1)
    plt.xticks([]), plt.yticks([])
    axes.imshow(image, interpolation='nearest', cmap='gray')

    user_input_menu1_blobcsv = input("\nEnter Old blob.csv File: ")
    file_blob = open(user_input_menu1_blobcsv, "r")

    for line in file_blob:
        p = line.split(",")

        x_list.append(float(p[0]))
        y_list.append(float(p[1]))

        xy_list.append([float(p[0]), float(p[1])])

    file_blob.close()

    ################################################################################################################################

    scat = plt.scatter(x_list, y_list, color='red')

    fig.show()
    fig.canvas.draw()

    while True:

        print ("\n#############################################################################################")
        print ("#############################################################################################")
        print ("\nUser Blob RED Point Adjustment Interface\n")
        print ("Only Interact with the plot (e.g. Zoom and Pan) when not in Add/Remove points mode.\n")
        print ("Please type 1, 2, 3, or 4 to select from the options below.\n")
        print ("1. Manually Add Points\n")
        print ("2. Manually Remove Points\n")
        print ("3. Save Blob Center Points Data\n")
        print ("4. Done \n")

        user_input_interface = input(":")

        if (user_input_interface == '1'):

            print("\nPlease click to add points.")

            cid = fig.canvas.mpl_connect('button_press_event', onclick_add)

            user_input2 = input("\nPress any key to move on.")

            fig.canvas.mpl_disconnect(cid)

        elif (user_input_interface == '2'):

            print("\nPlease click to remove points.")

            cid2 = fig.canvas.mpl_connect('button_press_event', onclick_remove)

            user_input2 = input("\nPress any key to move on.")

            fig.canvas.mpl_disconnect(cid2)

        elif (user_input_interface == '3'):

            user_input_filename = input("Please enter the desired blob file name: ")

            file_blob_new = open(user_input_filename, "w")

            for f, b in zip(x_list, y_list):
                file_blob_new.write(str(f))
                file_blob_new.write(",")
                file_blob_new.write(str(b))
                file_blob_new.write("\n")

            file_blob_new.close()

        elif (user_input_interface == '4'):

            break

        else:

            print ("\nIncorrect input. Please try again.")

    ############################################################################################################################################

    user_input_menu1_verticescsv = input("\nEnter Old vertices.csv File: ")
    file_vertices = open(user_input_menu1_verticescsv, "r")

    for line in file_vertices:
        p = line.split(",")

        vertices_x_list.append(float(p[0]))
        vertices_y_list.append(float(p[1]))

        vertices_xy_list.append([float(p[0]), float(p[1])])

    file_vertices.close()

    scat2 = plt.scatter(vertices_x_list, vertices_y_list, color='blue')

    ############################################################################################################################################

    user_input_menu1_linescsv = input("\nEnter Old lines.csv File: ")
    file_lines = open(user_input_menu1_linescsv, "r")

    for line in file_lines:
        p = line.split(",")

        green_lines.append((int(p[0]), int(p[1])))

    file_lines.close()

    user_input_menu1_removedlinescsv = input("\nEnter Old removed_lines.csv File: ")
    file_removedlines = open(user_input_menu1_removedlinescsv, "r")

    for line in file_removedlines:
        p = line.split(",")

        list_of_disconnected_points.append([[float(p[0]), float(p[1])], [float(p[2]), float(p[3])]])

    file_removedlines.close()

    user_input_menu1_addedlinescsv = input("\nEnter Old added_lines.csv File: ")
    file_addedlines = open(user_input_menu1_addedlinescsv, "r")

    for line in file_addedlines:
        p = line.split(",")

        list_of_connected_points.append([[float(p[0]), float(p[1])], [float(p[2]), float(p[3])]])

    file_addedlines.close()

    ############################################################################################################################################

    fig.canvas.draw()

    if (len(xy_list) < 6 or len(vertices_xy_list) < 4):
        print ("\nThere were either less than 6 RED center points or less than 4 BLUE vertice points. \n")
        print ("Please Try Again.")
        plt.close()
        sys.exit()

    ############################################################################################################################################
    # Calculate first cycle of green lines.

    calculate_vertice_lines()  # Drawing the actual green lines.

    # Initialize first automatic set of lines and line_collection.
    lines = [[vertices_xy_list[i], vertices_xy_list[j]] for i, j in green_lines]
    line_collection = mc.LineCollection(lines, color='green')
    axes.add_collection(line_collection)

    fig.canvas.draw()

    ############################################################################################################################################

    while True:

        print ("\n#############################################################################################")
        print ("\n*** Please do NOT go below 6 RED center points or 4 BLUE vertice points. ***\n")
        print ("#############################################################################################")
        print ("\nUser Vertices Line Adjustment Interface\n")
        print ("Adjustment of lines (GREEN) through adjustment of vertice points (BLUE).\n")
        print ("Only Interact with the plot (e.g. Zoom and Pan) when not in Add/Remove points mode.\n")
        print ("Please type 1, 2, 3, or 4 to select from the options below.\n")
        print ("1. Manually Add Points\n")
        print ("2. Manually Remove Points\n")
        print ("3. Manually Add Lines\n")
        print ("4. Manually Remove Lines\n")
        print ("5. Save Line and Vertice Data\n")
        print ("6. Done \n")

        user_input_interface2 = input(":")

        if (user_input_interface2 == '1'):

            print("\nPlease click to add points.")

            cid_vert = fig.canvas.mpl_connect('button_press_event', onclick_add_vertices)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert)

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

        elif (user_input_interface2 == '2'):

            print("\nPlease click to remove points.")

            cid_vert2 = fig.canvas.mpl_connect('button_press_event', onclick_remove_vertices)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert2)

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

        elif (user_input_interface2 == '3'):

            print("\nPlease click 2 BLUE points to add lines.")

            cid_vert3 = fig.canvas.mpl_connect('button_press_event', onclick_add_lines)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert3)

        elif (user_input_interface2 == '4'):

            print("\nPlease click 2 BLUE points to remove lines.")

            cid_vert4 = fig.canvas.mpl_connect('button_press_event', onclick_remove_lines)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert4)

        elif (user_input_interface2 == '5'):

            vertices_x_list.clear()
            vertices_y_list.clear()

            vertices_x_list, vertices_y_list = map(list, zip(*vertices_xy_list))

            ####################################################################################################

            user_input_filename = input("Please enter the desired vertices file name: ")

            file_vertices_new = open(user_input_filename, "w")

            for f, b in zip(vertices_x_list, vertices_y_list):
                # Saves COORDINATE pairs of vertices_xy_list.
                file_vertices_new.write(str(f))
                file_vertices_new.write(",")
                file_vertices_new.write(str(b))
                file_vertices_new.write("\n")

            file_vertices_new.close()

            ####################################################################################################

            user_input_filename = input("Please enter the desired lines file name: ")

            file_lines_new = open(user_input_filename, "w")

            for (f, b) in green_lines:
                # Saves tuple of INDICES of all points on vertices_xy_list that should be connected.
                file_lines_new.write(str(f))
                file_lines_new.write(",")
                file_lines_new.write(str(b))
                file_lines_new.write("\n")

            file_lines_new.close()

            user_input_filename = input("Please enter the desired removed lines file name: ")

            file_removed_lines_new = open(user_input_filename, "w")

            for f, b in list_of_disconnected_points:
                file_removed_lines_new.write(str(f[0]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(f[1]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(b[0]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(b[1]))
                file_removed_lines_new.write("\n")

            file_removed_lines_new.close()

            user_input_filename = input("Please enter the desired added lines file name: ")

            file_added_lines_new = open(user_input_filename, "w")

            for f, b in list_of_connected_points:
                file_added_lines_new.write(str(f[0]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(f[1]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(b[0]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(b[1]))
                file_added_lines_new.write("\n")

            file_added_lines_new.close()

        ####################################################################################################

        elif (user_input_interface2 == '6'):
            print("Do you want to see the distribution of wall length and cell area? [y/n] \n")
            user_input = input(":")
            if (user_input == 'n'):
                break
            elif (user_input == 'y'):
                #del wall_len[:]
                for p1 in vertices_xy_list:
                    for p2 in vertices_xy_list:
                        b = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                        if (b > 25 and b < 50):
                            wall_len.append(b)
                            
                print(len(wall_len))
                print(len(angles))  
                
                for c in wall_len:
                    a=3*(3**0.5)*(c**2)/2
                    areas.append(a)
                    
                print(len(areas))
                
                print(7)

                plt.figure(8)
                sns.distplot(wall_len, rug=True)
                plt.xlabel("wall length")
                plt.ylabel("percentage")
                plt.title("wall length distribution diagram")
                print(8)
                
                plt.figure(9)
                sns.distplot(areas, rug=True)
                plt.xlabel("area")
                plt.ylabel("percentage")
                plt.title("area distribution diagram")
                print(9)
                plt.show()
            break


        else:

            print ("\nIncorrect input. Please try again.")

    print ("\nSaving cell wall pixel lengths in cell_wall_lengths.csv")

    file_cellwall_lengths = open("cell_pixel_wall_lengths.csv", "w")

    for f in cell_wall_lengths:
        file_cellwall_lengths.write(str(f))
        file_cellwall_lengths.write("\n")

    file_cellwall_lengths.close()

    fig.canvas.draw()

##################################################################################################################################################################################################################################################################

if (user_input_menu1 == '3'):

    print ("\n################################## BLANK Slate Mode ##################################")

    user_input_menu1_1 = input("\nEnter Image File: ")

    image = cv2.imread(user_input_menu1_1, cv2.IMREAD_GRAYSCALE)

    image2 = rgb2gray(image)

    image_gray = skimage.util.invert(image2)

    user_input1 = input(
        "\nEnter thresholding method\n ~ Adaptive Gaussian ('ag')\n ~ Adaptive Mean ('am')\n ~ Binary ('b')\n ~ No Threshold ('no')\nEnter Input: ")

    ################################################ Different Thresholding Methods ################################################

    if (user_input1 == 'b'):

        retval, image_g = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)

    elif (user_input1 == 'ag'):

        image_g = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif (user_input1 == 'am'):

        image_g = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    elif (user_input1 == 'no'):

        image_g = image_gray

    else:

        sys.exit()

    ################################################################################################################################

    fig, axes = plt.subplots()

    axes.set_title('LoG Detection - ' + user_input1)
    plt.xticks([]), plt.yticks([])
    axes.imshow(image, interpolation='nearest', cmap='gray')

    ################################################################################################################################

    scat = plt.scatter(x_list, y_list, color='red')

    fig.show()
    fig.canvas.draw()

    while True:

        print ("\n#############################################################################################")
        print ("\n*** Please plot AT LEAST 6 RED center points that results in AT LEAST 4 blue vertice points. ***\n")
        print ("#############################################################################################")
        print ("\nUser Blob RED Point Adjustment Interface\n")
        print ("Only Interact with the plot (e.g. Zoom and Pan) when not in Add/Remove points mode.\n")
        print ("Please type 1, 2, 3, or 4 to select from the options below.\n")
        print ("1. Manually Add Points\n")
        print ("2. Manually Remove Points\n")
        print ("3. Save Blob Center Points Data\n")
        print ("4. Done \n")

        user_input_interface = input(":")

        if (user_input_interface == '1'):

            print("\nPlease click to add points.")

            cid = fig.canvas.mpl_connect('button_press_event', onclick_add)

            user_input2 = input("\nPress any key to move on.")

            fig.canvas.mpl_disconnect(cid)

        elif (user_input_interface == '2'):

            print("\nPlease click to remove points.")

            cid2 = fig.canvas.mpl_connect('button_press_event', onclick_remove)

            user_input2 = input("\nPress any key to move on.")

            fig.canvas.mpl_disconnect(cid2)

        elif (user_input_interface == '3'):

            user_input_filename = input("Please enter the desired blob file name: ")

            file_blob_new = open(user_input_filename, "w")

            for f, b in zip(x_list, y_list):
                file_blob_new.write(str(f))
                file_blob_new.write(",")
                file_blob_new.write(str(b))
                file_blob_new.write("\n")

            file_blob_new.close()

        elif (user_input_interface == '4'):

            break

        else:

            print ("\nIncorrect input. Please try again.")

    ############################################################################################################################################

    PATH = user_input_menu1_1
    image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    image2 = rgb2gray(image)
    thresh1 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
    kernel1 = np.ones((6, 6), np.uint8)
    img = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    min_size = 250

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    kernel2 = np.ones((5, 5), np.uint8)
    image_g = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel2)

    all_vertex = []
    at = []
    # all_points = list(xy_list)
    angles = []
    wall_len = []
    areas = []
    sobelx = cv2.Sobel(image_g, cv2.CV_64F, 1, 0)  # Find x and y gradients  image_g
    sobely = cv2.Sobel(image_g, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)

    for i in range(len(xy_list)):
        x = xy_list[i][0]  # for this point
        y = xy_list[i][1]

        distance = []

        around_cells = []

        for d in xy_list:
            xi = d[0]  # k
            yi = d[1]
            distance.append(((x - xi) ** 2 + (y - yi) ** 2) ** 0.5)

        sorted_distance = np.argsort(distance)

        for j in range(7):
            around_cells.append(xy_list[sorted_distance[j]])

        vertex = []
        wall = []
        points = []
        max_temp = []
        center = around_cells.pop(0)

        for c in around_cells:
            if (center[0] - c[0] != 0):
                k = (center[1] - c[1]) / (center[0] - c[0])
                b = center[1] - k * center[0]
            else:
                k = 10000
                b = 10000

            max_gradient = 0
            final = []
            line = []
            maxline = []

            ind = 0
            ind1 = 0

            for m in range(min(center[0], c[0]), max(center[0], c[0])):  # x
                for n in range(min(center[1], c[1]), max(center[1], c[1])):  # y
                    if (n == round(k * m + b)):
                        l = [m, n]
                        line.append(l)
                        ind = ind + 1

                        if (max_gradient < magnitude[n][m]):
                            max_gradient = magnitude[n][m]
                            ind1 = ind

            ind2 = ind1
            ind3 = ind1 - 1

            while (ind2 <= len(line) - 1):

                if image_g[line[ind2][1], line[ind2][0]] == 255:
                    maxline.append(line[ind2])

                ind2 = ind2 + 1

            while (ind3 >= 0):

                if image_g[line[ind3][1], line[ind3][0]] == 255:
                    maxline.insert(0, line[ind3])

                ind3 = ind3 - 1

            if (len(maxline) != 0):
                leng = len(maxline)
                r = int(leng / 2)

                final = maxline[r]
                if (len(final) != 0):
                    if (k == 0):
                        k_1 = 10000
                        b_1 = (float)(final[1] - k_1 * final[0])
                    else:
                        k_1 = (float)(-1 / (k))
                        b_1 = (float)(final[1] - k_1 * final[0])

                points.append(final)
                w = [k_1, b_1]
                wall.append(w)

        # print(len(points))
        for slop in wall:
            sk=slop[0]
            sb=slop[1]
        
            for slop2 in wall:
                sk2=slop2[0]
                sb2=slop2[1]
                
                a=np.abs(180*math.atan(sk)/np.pi-180*math.atan(sk2)/np.pi)  
                if (a > 100 and a < 150):
                    angles.append(a)

        for p1 in points:
            dis = []
            i1 = points.index(p1)
            k0 = wall[i1][0]
            b0 = wall[i1][1]
            for p2 in points:
                d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                dis.append(d)

            sorted_dis = np.argsort(dis)

            if (len(sorted_dis) >= 3):
                index1 = sorted_dis[1]
                index2 = sorted_dis[2]

                k1 = wall[index1][0]
                b1 = wall[index1][1]
                k2 = wall[index2][0]
                b2 = wall[index2][1]

                if (k0 != k1):
                    vx1 = (b1 - b0) / (k0 - k1)
                    vy1 = k1 * vx1 + b1
                    v1 = [vx1, vy1]
                # print(v1)
                if v1 not in all_vertex:
                    if (v1[0] > 0 and v1[1] > 0 and v1[0] < image_g.shape[1] and v1[1] < image_g.shape[0] and image_g[
                        round(v1[1]), round(v1[0])] == 255):
                        all_vertex.append(v1)

                if (k0 != k2):
                    vx2 = (b2 - b0) / (k0 - k2)
                    vy2 = k2 * vx2 + b2
                    v2 = [vx2, vy2]
                # print(v2)
                if v2 not in all_vertex:
                    if (v2[0] > 0 and v2[1] > 0 and v2[0] < image_g.shape[1] and v2[1] < image_g.shape[0] and image_g[
                        round(v2[1]), round(v2[0])] == 255):
                        all_vertex.append(v2)

    for m in all_vertex:
        temp = []
        tempx = []
        tempy = []
        temp.append(m)
        tempx.append(m[0])
        tempy.append(m[1])
        for n in all_vertex:
            if (((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2) ** 0.5 < 20):
                temp.append(n)
                tempx.append(n[0])
                tempy.append(n[1])
                all_vertex.remove(n)

        vertices_x_list.append(np.mean(tempx))
        vertices_y_list.append(np.mean(tempy))
        vertices_xy_list.append([np.mean(tempx), np.mean(tempy)])

    scat2 = plt.scatter(vertices_x_list, vertices_y_list, color='blue')

    fig.canvas.draw()

    if (len(xy_list) < 6 or len(vertices_xy_list) < 4):
        print ("\nThere were either less than 6 RED center points or less than 4 BLUE vertice points. \n")
        print ("Please Try Again.")
        plt.close()
        sys.exit()

    ############################################################################################################################################
    # Calculate first cycle of green lines.

    calculate_vertice_lines()  # Drawing the actual green lines.

    # Initialize first automatic set of lines and line_collection.
    lines = [[vertices_xy_list[i], vertices_xy_list[j]] for i, j in green_lines]
    line_collection = mc.LineCollection(lines, color='green')
    axes.add_collection(line_collection)

    fig.canvas.draw()

    ############################################################################################################################################

    while True:

        print ("\n#############################################################################################")
        print ("\n*** Please do NOT go below 6 RED center points or 4 BLUE vertice points. ***\n")
        print ("#############################################################################################")
        print ("\nUser Vertices Line Adjustment Interface\n")
        print ("Adjustment of lines (GREEN) through adjustment of vertice points (BLUE).\n")
        print ("Only Interact with the plot (e.g. Zoom and Pan) when not in Add/Remove points mode.\n")
        print ("Please type 1, 2, 3, or 4 to select from the options below.\n")
        print ("1. Manually Add Points\n")
        print ("2. Manually Remove Points\n")
        print ("3. Manually Add Lines\n")
        print ("4. Manually Remove Lines\n")
        print ("5. Save Line and Vertice Data\n")
        print ("6. Done \n")

        user_input_interface2 = input(":")

        if (user_input_interface2 == '1'):

            print("\nPlease click to add points.")

            cid_vert = fig.canvas.mpl_connect('button_press_event', onclick_add_vertices)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert)

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

        elif (user_input_interface2 == '2'):

            print("\nPlease click to remove points.")

            cid_vert2 = fig.canvas.mpl_connect('button_press_event', onclick_remove_vertices)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert2)

            line_collection.remove()
            del line_collection
            # fig.canvas.draw()

            green_lines.clear()

            calculate_vertice_lines()

            draw_update_vertice_lines()

        elif (user_input_interface2 == '3'):

            print("\nPlease click 2 BLUE points to add lines.")

            cid_vert3 = fig.canvas.mpl_connect('button_press_event', onclick_add_lines)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert3)

        elif (user_input_interface2 == '4'):

            print("\nPlease click 2 BLUE points to remove lines.")

            cid_vert4 = fig.canvas.mpl_connect('button_press_event', onclick_remove_lines)

            user_input3 = input("\nPress ENTER to move on.\n")

            fig.canvas.mpl_disconnect(cid_vert4)

        elif (user_input_interface2 == '5'):

            vertices_x_list.clear()
            vertices_y_list.clear()

            vertices_x_list, vertices_y_list = map(list, zip(*vertices_xy_list))

            ####################################################################################################

            user_input_filename = input("Please enter the desired vertices file name: ")

            file_vertices_new = open(user_input_filename, "w")

            for f, b in zip(vertices_x_list, vertices_y_list):
                # Saves COORDINATE pairs of vertices_xy_list.
                file_vertices_new.write(str(f))
                file_vertices_new.write(",")
                file_vertices_new.write(str(b))
                file_vertices_new.write("\n")

            file_vertices_new.close()

            ####################################################################################################

            user_input_filename = input("Please enter the desired lines file name: ")

            file_lines_new = open(user_input_filename, "w")

            for (f, b) in green_lines:
                # Saves tuple of INDICES of all points on vertices_xy_list that should be connected.
                file_lines_new.write(str(f))
                file_lines_new.write(",")
                file_lines_new.write(str(b))
                file_lines_new.write("\n")

            file_lines_new.close()

            user_input_filename = input("Please enter the desired removed lines file name: ")

            file_removed_lines_new = open(user_input_filename, "w")

            for f, b in list_of_disconnected_points:
                file_removed_lines_new.write(str(f[0]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(f[1]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(b[0]))
                file_removed_lines_new.write(",")
                file_removed_lines_new.write(str(b[1]))
                file_removed_lines_new.write("\n")

            file_removed_lines_new.close()

            user_input_filename = input("Please enter the desired added lines file name: ")

            file_added_lines_new = open(user_input_filename, "w")

            for f, b in list_of_connected_points:
                file_added_lines_new.write(str(f[0]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(f[1]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(b[0]))
                file_added_lines_new.write(",")
                file_added_lines_new.write(str(b[1]))
                file_added_lines_new.write("\n")

            file_added_lines_new.close()

        ####################################################################################################

        elif (user_input_interface2 == '6'):
            print ("Do you wnat to see distribution map [y/n] \n")
            user_input = input(":")
            if (user_input == 'n'):
                break
            elif (user_input == 'y'):
                #del wall_len[:]
                for p1 in vertices_xy_list:
                    for p2 in vertices_xy_list:
                        b = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                        if (b > 25 and b < 50):
                            wall_len.append(b)
                            
                print(len(wall_len))
                print(len(angles))  
                
                for c in wall_len:
                    a=3*(3**0.5)*(c**2)/2
                    areas.append(a)
                    
                print(len(areas))
                

                plt.figure(7)
                sns.distplot(angles, rug=True)
                plt.xlabel("degree")
                plt.ylabel("percentage")
                plt.title("angular distribution diagram")
                print(7)

                plt.figure(8)
                sns.distplot(wall_len, rug=True)
                plt.xlabel("wall length")
                plt.ylabel("percentage")
                plt.title("wall length distribution diagram")
                print(8)
                
                plt.figure(9)
                sns.distplot(areas, rug=True)
                plt.xlabel("area")
                plt.ylabel("percentage")
                plt.title("area distribution diagram")
                print(9)
                plt.show()
            break

        else:

            print ("\nIncorrect input. Please try again.")

    print ("\nSaving cell wall pixel lengths in cell_wall_lengths.csv")

    file_cellwall_lengths = open("cell_pixel_wall_lengths.csv", "w")

    for f in cell_wall_lengths:
        file_cellwall_lengths.write(str(f))
        file_cellwall_lengths.write("\n")

    file_cellwall_lengths.close()

    fig.canvas.draw()


