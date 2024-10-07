# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:02:26 2024

@author: pb910
"""

import math
import matplotlib.pyplot as plt
import copy

def rotate(points, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in degrees.
    """
    angle = math.radians(angle) # degrees to radians
    cosx = math.cos(angle)
    sinx = math.sin(angle)
    newpoints = []
        
    for point in points:
        ox, oy = origin
        px, py = point
        qx = ox + cosx * (px - ox) - sinx * (py - oy)
        qy = oy + sinx * (px - ox) + cosx * (py - oy)
        newpoints.append( [qx, qy] )
    return(newpoints)

def make_tripod(xcords,ycords,start_angle=0):
    basic = []
    if len(xcords) != len(ycords):
        print("mismatched number of xcords and ycords")
        return 
    else:
        for i in range (len(xcords)):
            basic.append( (float(xcords[i]), float(ycords[i])))
    origin = (xcords[0], ycords[0])
    L1 = rotate(basic, origin, start_angle)
    L2 = rotate(basic, origin, start_angle + 120)
    L3 = rotate(basic, origin, start_angle + 240)
    return [L1,L2,L3]

def translate(geometry, vector):
    #expected format is row = [T1, T2,T3]
    # where T1 is [L1, L2, L3]
    # and L1 is [(x,y),(x,y) ect]
    #use recursion. Only the cordinates is length 2
    if len(geometry[0]) == 2:
        for cord in geometry:
            cord[0] = cord[0] + vector[0]
            cord[1] = cord[1] + vector[1]
    else:
        for g in geometry:
            translate(g, vector)
    return geometry

def plot_geometry(geometry, colour):
    if len(geometry[0]) == 2:
        ax.plot(*zip(*geometry), c= colour)
    else:
        for g in geometry:
            plot_geometry(g, colour)
    return
    



hex_side = 5.0

red_x = [0,1,2, 3, 4,5]
red_y = [0,1,1,-1,-1,0]

green_x = [0, 1, 5]
green_y = [0, 0, 0]

blue_x = [0, 1, 2, 2, 3, 3, 4, 4, 5]
blue_y = [0, 0, 0, 3, 3, -1, -1, 0, 0]


hex_width = hex_side * 3**(0.5)
hex_height = hex_side * 2


r_tripod = make_tripod(red_x, red_y, -90)
b_tripod = make_tripod(blue_x, blue_y, -90)
g_tripod = make_tripod(green_x, green_y, -90)

b_tripod = translate(b_tripod, (hex_width, 0) )
g_tripod = translate(g_tripod, (2*hex_width, 0) )
one_row = [r_tripod, b_tripod, g_tripod]

# becuase we're deep in recursion now, need to deepcopy
# to avoid affecting one_row
third_row = copy.deepcopy(one_row)
offset_row = copy.deepcopy(one_row)
third_row = translate(third_row, (0,-hex_height-hex_side) )
offset_row = translate(offset_row, (1.5*hex_width, -3/2*hex_side) )
offset2_row = copy.deepcopy(offset_row)
fourth_row = translate(offset2_row, (0,-hex_height-hex_side) )
block = [one_row, offset_row, third_row, offset2_row]

blocks_v = 10
blocks_h = 10 

blocksv = []
for i in range(0, blocks_v):
    new_block = copy.deepcopy(block)
    new_block = translate(new_block, (0,i*(-hex_height-hex_side)) )
    blocksv.append(new_block)

blocks = []
for j in range(0, blocks_h):
    new_col = copy.deepcopy(blocksv)
    new_col = translate(new_col, (j*3*hex_width, 0))
    blocks.append(new_col)

fig, ax = plt.subplots()
plot_geometry(blocks, 'green')
#plot_geometry(offset_row, 'black')
#plot_geometry(third_row, 'blue')

