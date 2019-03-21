#include <opencv2/core/utility.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

using namespace std;
using namespace cv;

int __main(int argc, char** argv) {
	std::string trackingAlg = "MOSEE";

	//MultiTracker trackers;
	vector<Rect2d> objects;

	std::string video = "E:\\work_min\\1.avi";
	VideoCapture cap(video);

	Mat frame;

	// get bounding box
	cap >> frame;
	vector<Rect> ROIs;
	selectROIs("tracker", frame, ROIs);

	//quit when the tracked object(s) is not provided
	if (ROIs.size()<1)
		return 0;

	// initialize the tracker
	//std::vector<Ptr<Tracker> > algorithms;
	for (size_t i = 0; i < ROIs.size(); i++)
	{
		//algorithms.push_back(TrackerMOSSE::create());
		//objects.push_back(ROIs[i]);
	}

	//trackers.add(algorithms, frame, objects);

	// do the tracking
	printf("Start the tracking process, press ESC to quit.\n");
	while (1)
	{
		cap >> frame;
		if (frame.rows == 0 || frame.cols == 0)
			break;

		//trackers.update(frame);

		// draw the tracked object
		//for (unsigned i = 0; i<trackers.getObjects().size(); i++)
			//rectangle(frame, trackers.getObjects()[i], Scalar(255, 0, 0), 2, 1);


		imshow("tracker", frame);

		if (waitKey(1) == 27)break;
	}

}
