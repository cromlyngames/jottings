# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:34:47 2022

@author: pb910
"""
#!/usr/bin/python
import matplotlib.pyplot as plt

import random as r



#very eleglant intersection from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)



def get_rand_line(x_max, y_max):
    rx1 = r.randint(0, x_max)
    ry1 = r.randint(0, y_max)
    rx2 = r.randint(0, x_max)
    ry2 = r.randint(0, y_max)
    
    a = Point(rx1, ry1)
    b = Point(rx2, ry2)
    return [a,b]
    

def one_map(xmax, ymax, linecount):
    # append first line
    intersectcount = 0
    lines = []
    line = get_rand_line(xmax, ymax)
    lines.append(line)

    for i in range(1, linecount, 1):
        nextline = get_rand_line(xmax, ymax)
        for lin in lines:
            a = lin[0]
            b = lin[1]
            c = nextline[0]
            d = nextline[1]
            #print(intersect(a,b,c,d))
            if intersect(a,b,c,d):
                intersectcount = intersectcount +1
        lines.append(nextline)
    return intersectcount, lines

#useability functions

def print_lines(lines):
    #prints off all of the cordinates in a table for error checking
    for lin in lines:
        p1 = lin[0]
        p2 = lin[1]
        print(p1.x, p1.y, p2.x, p2.y)
    

def plot_map(lines):
    #plots 
    # creating an empty canvas
    fig = plt.figure(1)
     
    # defining the axes with the projection
    # as 3D so as to plot 3D graphs
    ax1 = plt.axes()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    for lin in lines:
        p1 = lin[0]
        p2 = lin[1]
        p1x = p1.x
        p2x = p2.x
        xplot = [p1x, p2x]
        yplot = [p1.y, p2.y]
        ax1.plot(xplot, yplot) 

    plt.show()
    

### main


xmax = 100
ymax = 100

for i in range (3, 100):
    intersections, lines = one_map(xmax, ymax, i)    
    print(intersections)




    