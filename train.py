import os
import sys
import glob

import dlib

if len(sys.argv) < 2:
    print(
        "Give the path to the train dataset & test dataset directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train.py ../TRAIN_DATA_PATH ../TEST_DATA_PATH")
    exit()
path = sys.argv[1]
if sys.argv[2] is not None:
    test_path = sys.argv[2]

# before train detector using train_simple_object_detector() function.
# set some train parameters
options = dlib.simple_object_detector_training_options()

# parameters
# 1. C : usual SVM C regularization parameter. Larger values of C  might lead to overfit. 
# 2. add_left_right_image_flips : if true, train left/right flip images
# 3. be_verbose : If true, print process while training.
# 4. detection_window_size : sliding window size. if wanna fine small size, try small window size. minium is 30x30
# 5. epsilon : stopping value. Smaller values make more accurate. 
# 6. max_runtime_seconds : smiliar maximum epoch.
# 7. nuclear_norm_regularization_strength : 
# 8. num_threads :  Set the number of CPU cores while training.
# 9. upsample_limit : upsample time for images.  0 forbid trainer to upsample any images. 2 (default) is recommend. 

options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 6
options.be_verbose = True
options.detection_window_size = 30 * 30

# train Hog detector using train option & train data.
dlib.train_simple_object_detector(path, "detector.svm", options)

# test detector for test images. print(the precision, recall, and then)
# average precision.
print("") 
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(path, "detector.svm")))

if sys.argv[2] is not None :
    print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(test_path, "detector.svm")))
