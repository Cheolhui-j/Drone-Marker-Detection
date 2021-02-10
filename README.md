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

+ Annotation

   The training data was annotated using 'Dlib Imglab Tool' from the following [link](https://github.com/davisking/dlib/tree/master/tools/imglab).
   
   ![Annotation_example](./example/annotation.png)

+ Train 

      python3 train.py

      usage: train.py  [--train_db_dir TRAIN_DB_DIR] [--valid_db_dir VALID_DB_DIR]

      optional arguments:
            --train_db_dir TRAIN_DB_DIR Directory where training data is stored
            --valid_db_dir VALID_DB_DIR Directory where validation data is stored
      
+ Test 

      python3 main.py

      usage: train.py  [--test_dir Test_DIR] [--test_type TEST_TYPE]

      optional arguments:
            --test_dir Test_DIR Directory where testing file is stored
            --test_type TEST_TYPE Type what testing data is image(.jpg) or video(.mp4)


# Visualization


# TO DO
+ add image about img tool & visualization 
+ add ver. C++ 

# Reference 
Dlib example code [link](http://dlib.net/)
