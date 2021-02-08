# Drone-Marker-Detection

This Project aims to detect specific Drone landing marker. 
The drone is hovering over the air, looking for the corresponding marker on the ground. 
When it finds a marker, it lands at that location.
The basic code is the histogram of gradients & Support vector machine, which is an object detection technique in dlib.
Since you need to mount the code on the drone, the code is assumed to work on the CPU.

# Method 

This project used a technique based on [Histograms of oriented gradients for human detection.](https://ieeexplore.ieee.org/document/1467360)
+ Normalize gamma & colour
+ Compute gradients
+ Weighted vote into spatial & orientation cells
+ Contrast normalize over overlapping spatial blocks 
+ Collect HOG's over detection window 
+ Train Linear SVM
+ Marker / Non-Marker classification

# How to use

+ Train 

''' 

python3 train.py

usage: train.py  [--train_db_dir TRAIN_DB_DIR] [--test_db_dir TEST_DB_DIR]

optional arguments:
      --train_db_dir TRAIN_DB_DIR Directory where training data is stored
      --test_db_dir TEST_DB_DIR Directory where testing data is stored
      
'''
