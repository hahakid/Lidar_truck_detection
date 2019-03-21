#include <opencv2/core/utility.hpp>
//#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "yolo_v2_class.hpp"    // imported functions from DLL
using namespace std;
using namespace cv;

int _main(int argc, char** argv) {
	//
	int obj_n = 1;
	string  names_file = "E:\\work_min\\data\\lidar\\lidar_data\\lidar_data\\names.list";
	string  cfg_file = "E:\\work_min\\data\\lidar\\lidar_data\\lidar_data\\lidar.cfg";
	string  weights_file = "E:\\work_min\\data\\lidar\\lidar_model\\model_480_480\\lidar_252700.weights";
	//string filename = "E:\\work_min\\data\\lidar\\lidar_data\\datas\\feature_3c_anno1\\feature_3c\\2\\1499.png";
	string filename = "E:\\work_min\\1.avi";
	float const thresh = 0.25;
	Detector_YOLO detector(cfg_file, weights_file);

	clock_t startTime, endTime;
	// 初始化，创建
	//Ptr<TrackerMOSSE> tracker = TrackerMOSSE::create();
	Rect2d objects;


	//设置video
	//string video_name = "E:\\work_min\\test.mp4";
	string video_name = "E:\\work_min\\1.avi";
	//string video_name = "E:\\work_min\\demo.mp4";


	string write_video_name = "E:\\work_min\\res_mosse.avi";
	VideoCapture cap(video_name);
	VideoWriter writer;
	if (!cap.isOpened())
	{
		cout << "open video error\n" << endl;
		return 0;
	}


	Mat frame;
	cap >> frame;
	bool isColor = (frame.type() == CV_8UC3);
	writer.open(write_video_name, CV_FOURCC('M', 'J', 'P', 'G'), 10.0, frame.size(), true);
	if (!writer.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}
	vector<bbox_t> obj_result_vec = detector.detect(frame);
	objects.height = (double)obj_result_vec[0].h;
	objects.width = (double)obj_result_vec[0].w;
	objects.x = (double)obj_result_vec[0].x;
	objects.y = (double)obj_result_vec[0].y;
	//objects = selectROI("tracker", frame, false, false);
	//tracker->init(frame, objects);
	//   !!!!! do
	int count = 0;
	bool first_frame = false;
	double time_all = 0.0;
	while (1)
	{
		cap >> frame;
		if (frame.rows == 0 || frame.cols == 0) break;
		count++;
		if (count < 0)
		{
			continue;
		}
		if (first_frame)
		{// 获取 目标框, 初始化 tracker
			vector<bbox_t> obj_result_vec = detector.detect(frame);
			objects.height = (double)obj_result_vec[0].h;
			objects.width = (double)obj_result_vec[0].w;
			objects.x = (double)obj_result_vec[0].x;
			objects.y = (double)obj_result_vec[0].y;
			//objects = selectROI("tracker", frame, false, false);
			//tracker->init(frame, objects);
			first_frame = false;
		}
		else
		{
			startTime = clock();
			//tracker->update(frame, objects);
			endTime = clock();
			time_all += (double)(endTime - startTime) / CLOCKS_PER_SEC;
			//画框
			rectangle(frame, objects, Scalar(0, 255, 0), 2);
			//show
			imshow("tracker", frame);
			writer << frame;
			if (waitKey(1) == 27)break; //按esc退出
		}

	}
	cout << "time average cost " << time_all / count << " s" << endl;
	writer.release();
	cap.release();
	return 1;
}
