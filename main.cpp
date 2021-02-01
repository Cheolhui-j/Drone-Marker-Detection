#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <opencv2/opencv.hpp>
#include <dlib/opencv/cv_image.h>

#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;
using namespace dlib;

int main(int argc, char** argv)
{
		time_t start, end;

		start = time(NULL);

		int count_frame = 0;


		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

		object_detector<image_scanner_type> detector1, detector2;
		deserialize("./model/detector.svm") >> detector1;
		deserialize("./model/detector_small.svm") >> detector2;


		cv::VideoCapture cap(argv[1]);

		while (1)
		{
			cv::Mat temp;
			// wait for a new frame from camera and store it into 'frame'
			bool isframe = cap.read(temp);
			// check if we succeeded
			if (temp.empty())
				break;
			
			cv::resize(temp, temp, cv::Size(640,480));
	
			dlib::array2d<dlib::bgr_pixel> dlibImage;
			dlib::assign_image(dlibImage, dlib::cv_image<dlib::bgr_pixel>(temp));

			// Upsample images if the user asked us to do that.
			for (unsigned long i = 0; i < 1; ++i)
			{
				pyramid_up(dlibImage);
			}
			// Test the detector on the images we loaded and display the results
			// in a window.

			std::vector<object_detector<image_scanner_type> > my_detectors;
			my_detectors.push_back(detector1);
			my_detectors.push_back(detector2);

			image_window win;
			// Run the detector on images[i] 
			const std::vector<rectangle> rects = dlib::evaluate_detectors(my_detectors, dlibImage);
			cout << "Number of detections  : " << rects.size() << endl;;
			// Put the image and detections into the window.
			win.clear_overlay();
			win.set_image(dlibImage);
			win.add_overlay(rects, rgb_pixel(255, 0, 0));
			count_frame++;
		}

		end = time(NULL);

		double total = end - start;

		std::cout << std::endl << "total : " << total << std::endl;

}
