#!/usr/bin/env python
# coding: utf-8

# # Toboggan Trajectory Challenge
# 
# |||
# |-|-|
# |**Author**|Ángel García Alcántara<br><angelgarcan@gmail.com>|
# |**Date**|March 5th 2021|
# 
# <br>
# 
# **Instructions:** https://docs.google.com/document/d/1Dzx6X71KsXjJrfeQr3UmTSD8RbYIOfeuSEvjUE0FM1g/edit#
# 
# **Input (map):** https://drive.google.com/file/d/1ReBBHgI1ol5eKFM6WN-2PFLhie_B2AOT/view

# ### Behaivor
# The program's input is the path to the file with the forest map.<br>
# It reads the map pattern, shows it, then calculates and prints the solution.

# ##### Reading and printing map
# mapfilename="map_forest_sample.txt"
mapfilename="map_forest.txt"


with open(mapfilename, 'r', encoding='utf-8') as f:
    forest=[list(line.replace('\n','')) for line in f]

h=len(forest)
w=len(forest[0])

print(f">>>> FOREST (height={h}, width={w}) <<<<\n")
for line in forest:
    print(''.join(line))


# ##### Calculating trajectory
# Starting in the top-left corner.
row=0
col=0
trajectory=""
XOs=""

while row<h: # Until reaching the bottom-most row.
    trajectory+=forest[row][col] # Slope in the position (row,col).
    XOs+='X' if trajectory[-1]=='#' else 'O' # X if there is a tree, O otherwise.
    print(f"({row},{col})::({trajectory[-1]},{XOs[-1]})",end=' -> ') # Print all steps.
    
    # Moving right 3 and down 1.
    row+=1
    col+=3
    if col>=w: col=col%w # At the rigth edge continue on the left.


# ##### Counting trees in trajectory
trees_count=XOs.count('X')


# ##### Printing solution
print(f"TRAJECTORY :\n{trajectory}")
print(f"XOs :\n{XOs}")

print(f"Number of trees: {trees_count}")

