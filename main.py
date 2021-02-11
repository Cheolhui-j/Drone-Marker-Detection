import imutils
import dlib
import cv2
import time
import sys
import os.path

if len(sys.argv) < 2:
    print(
        "Give the path to the examples directory & the type as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ./TEST_DATA_PATH TEST_DATA_TYPE")
    exit()

# test file directory & test file type
file_dir = sys.argv[1]
file_type = sys.argv[2]

# if file is image, create test image list
if file_type == 'jpg':
    if file_dir.endswith('.jpg'):
        file_list = [file_dir]
    else :
        file_list = os.listdir(file_dir)
        for i in range(len(file_list)):
            file_list[i] = file_dir + '/' + file_list[i]
        file_list.sort()
    #print (file_list)
# if file is video, create Video capture source
elif file_type == 'mp4':
    cap= cv2.VideoCapture(file_dir) 
# if file is neither jpg nor mp4, terminate program
else :
    print('Incorrect file is inserted!!')
    sys.exit()


start = time.time()
# Load marker detector model 
# detector1 : detect big or mediun size obects
# detector2 : detect small size objects 
detector1 = dlib.fhog_object_detector("./model/detector.svm")
detector2 = dlib.fhog_object_detector("./model/detector_small.svm")
detectors = [detector1, detector2]


# show detector HOG 
win_det = dlib.image_window()
win_det.set_image(detector1)


frame_count = 0
rects_count = 0

win = dlib.image_window()

# Load image from file
if file_type == 'jpg':
    for i in file_list:
        path = i
        image = cv2.imread(path)

        # Resize image for size control
        image = imutils.resize(image, width=1280, height=960)
        #image = imutils.resize(image, width=640, height=480)

        # if use one detector, use line below. 
        # but default is multiple. 
        #rects = detector1(image)

        # Run multiple detector  
        # requires
        #   1. list of detectors.
        #   2. image
        #   3. upsample_num_times >= 0 
        # return
        #   a tuple of (list of detections, list of scores, list of weight_indices).
        [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(
            detectors, image, upsample_num_times=0, adjust_threshold=0.0)

        # Draw detected boxes to image using 'add_overlay' function
        for i in range(len(boxes)):
            print("\n detector {} found box {} with confidence {}."
                .format(detector_idxs[i], boxes[i], confidences[i]))
            rects_count += 1
            win.clear_overlay()
            win.set_image(image)
            win.add_overlay(boxes[i])

        print("\n 단일 이미지 소요 시간 : ", time.time() - start)

        frame_count += 1

elif file_type == 'mp4':
    while True:
        ret, image = cap.read()

        if ret == False:
            break

        # Resize image for size control
        image = imutils.resize(image, width=1280, height=960)
        #image = imutils.resize(image, width=640, height=480)

        # if use one detector, use line below. 
        # but default is multiple. 
        #rects = detector1(image)

        # Run multiple detector  
        # requires
        #   1. list of detectors.
        #   2. image
        #   3. upsample_num_times >= 0 
        # return
        #   a tuple of (list of detections, list of scores, list of weight_indices).
        [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(
            detectors, image, upsample_num_times=0, adjust_threshold=0.0)

        # Draw detected boxes to image using 'add_overlay' function
        for i in range(len(boxes)):
            print("\n detector {} found box {} with confidence {}."
                .format(detector_idxs[i], boxes[i], confidences[i]))
            rects_count += 1
            win.clear_overlay()
            win.set_image(image)
            win.add_overlay(boxes[i])

        print("\n 단일 이미지 소요 시간 : ", time.time() - start)

        frame_count += 1

print("총 소요 시간 : ", time.time()-start)

print("정확도 : ", rects_count/frame_count)
