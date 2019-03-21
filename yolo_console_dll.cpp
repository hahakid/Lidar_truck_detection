#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // mutex, unique_lock
#include <condition_variable> // condition_variable

#define OPENCV
// To use tracking - uncomment the following line. Tracking is supported only by OpenCV 3.x
//#define TRACK_OPTFLOW

//#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include\cuda_runtime.h"
//#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.1/lib/x64/cudart.lib")
//static shared_ptr<image_t> device_ptr(NULL, [](void *img) { cudaDeviceReset(); });

#include "yolo_v2_class.hpp"    // imported functions from DLL
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>

#include <opencv2/videoio/videoio.hpp>
//#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)



using namespace std;
using namespace cv;



class track_kalman {
public:
    KalmanFilter kf;
    int state_size, meas_size, contr_size;


    track_kalman(int _state_size = 10, int _meas_size = 10, int _contr_size = 0)
        : state_size(_state_size), meas_size(_meas_size), contr_size(_contr_size)
    {
        kf.init(state_size, meas_size, contr_size, CV_32F);

        setIdentity(kf.measurementMatrix);
        setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(kf.processNoiseCov, Scalar::all(1e-5));
        setIdentity(kf.errorCovPost, Scalar::all(1e-2));
        setIdentity(kf.transitionMatrix);
    }

    void set(vector<bbox_t> result_vec) {
        for (size_t i = 0; i < result_vec.size() && i < state_size*2; ++i) {
            kf.statePost.at<float>(i * 2 + 0) = result_vec[i].x;
            kf.statePost.at<float>(i * 2 + 1) = result_vec[i].y;
        }
    }

    // Kalman.correct() calculates: statePost = statePre + gain * (z(k)-measurementMatrix*statePre);
    // corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    vector<bbox_t> correct(vector<bbox_t> result_vec) {
        Mat measurement(meas_size, 1, CV_32F);
        for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
            measurement.at<float>(i * 2 + 0) = result_vec[i].x;
            measurement.at<float>(i * 2 + 1) = result_vec[i].y;
        }
        Mat estimated = kf.correct(measurement);
        for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
            result_vec[i].x = estimated.at<float>(i * 2 + 0);
            result_vec[i].y = estimated.at<float>(i * 2 + 1);
        }
        return result_vec;
    }

    // Kalman.predict() calculates: statePre = TransitionMatrix * statePost;
    // predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    vector<bbox_t> predict() {
        vector<bbox_t> result_vec;
        Mat control;
        Mat prediction = kf.predict(control);
        for (size_t i = 0; i < prediction.rows && i < state_size * 2; ++i) {
            result_vec[i].x = prediction.at<float>(i * 2 + 0);
            result_vec[i].y = prediction.at<float>(i * 2 + 1);
        }
        return result_vec;
    }

};

class extrapolate_coords_t {
public:
    vector<bbox_t> old_result_vec;
    vector<float> dx_vec, dy_vec, time_vec;
    vector<float> old_dx_vec, old_dy_vec;

    void new_result(vector<bbox_t> new_result_vec, float new_time) {
        old_dx_vec = dx_vec;
        old_dy_vec = dy_vec;
        if (old_dx_vec.size() != old_result_vec.size()) cout << "old_dx != old_res \n";
        dx_vec = vector<float>(new_result_vec.size(), 0);
        dy_vec = vector<float>(new_result_vec.size(), 0);
        update_result(new_result_vec, new_time, false);
        old_result_vec = new_result_vec;
        time_vec = vector<float>(new_result_vec.size(), new_time);
    }

    void update_result(vector<bbox_t> new_result_vec, float new_time, bool update = true) {
        for (size_t i = 0; i < new_result_vec.size(); ++i) {
            for (size_t k = 0; k < old_result_vec.size(); ++k) {
                if (old_result_vec[k].track_id == new_result_vec[i].track_id && old_result_vec[k].obj_id == new_result_vec[i].obj_id) {
                    float const delta_time = new_time - time_vec[k];
                    if (abs(delta_time) < 1) break;
                    size_t index = (update) ? k : i;
                    float dx = ((float)new_result_vec[i].x - (float)old_result_vec[k].x) / delta_time;
                    float dy = ((float)new_result_vec[i].y - (float)old_result_vec[k].y) / delta_time;
                    float old_dx = dx, old_dy = dy;

                    // if it's shaking
                    if (update) {
                        if (dx * dx_vec[i] < 0) dx = dx / 2;
                        if (dy * dy_vec[i] < 0) dy = dy / 2;
                    } else {
                        if (dx * old_dx_vec[k] < 0) dx = dx / 2;
                        if (dy * old_dy_vec[k] < 0) dy = dy / 2;
                    }
                    dx_vec[index] = dx;
                    dy_vec[index] = dy;

                    //if (old_dx == dx && old_dy == dy) cout << "not shakin \n";
                    //else cout << "shakin \n";

                    if (dx_vec[index] > 1000 || dy_vec[index] > 1000) {
                        //cout << "!!! bad dx or dy, dx = " << dx_vec[index] << ", dy = " << dy_vec[index] <<
                        //    ", delta_time = " << delta_time << ", update = " << update << endl;
                        dx_vec[index] = 0;
                        dy_vec[index] = 0;
                    }
                    old_result_vec[k].x = new_result_vec[i].x;
                    old_result_vec[k].y = new_result_vec[i].y;
                    time_vec[k] = new_time;
                    break;
                }
            }
        }
    }

    vector<bbox_t> predict(float cur_time) {
        vector<bbox_t> result_vec = old_result_vec;
        for (size_t i = 0; i < old_result_vec.size(); ++i) {
            float const delta_time = cur_time - time_vec[i];
            auto &bbox = result_vec[i];
            float new_x = (float) bbox.x + dx_vec[i] * delta_time;
            float new_y = (float) bbox.y + dy_vec[i] * delta_time;
            if (new_x > 0) bbox.x = new_x;
            else bbox.x = 0;
            if (new_y > 0) bbox.y = new_y;
            else bbox.y = 0;
        }
        return result_vec;
    }

};

void draw_boxes(Mat mat_img, vector<bbox_t> result_vec, vector<string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        Scalar color = obj_id_to_color(i.obj_id);
        rectangle(mat_img, Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + to_string(i.track_id);
            Size const text_size = getTextSize(obj_name, FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            rectangle(mat_img, Point2f(max((int)i.x - 1, 0), max((int)i.y - 30, 0)),
                Point2f(min((int)i.x + max_width, mat_img.cols-1), min((int)i.y, mat_img.rows-1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, Point2f(i.x, i.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(0, 0, 0), 2);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        string fps_str = "FPS detection: " + to_string(current_det_fps) + "   FPS capture: " + to_string(current_cap_fps);
        putText(mat_img, fps_str, Point2f(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(50, 255, 0), 2);
    }
}

void show_console_result(vector<bbox_t> const result_vec, vector<string> const obj_names) {
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) cout << obj_names[i.obj_id] << " - ";
        cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << setprecision(3) << ", prob = " << i.prob << endl;
    }
}

vector<string> objects_names_from_file(string const filename) {
    ifstream file(filename);
    vector<string> file_lines;
    if (!file.is_open()) return file_lines;
    for(string line; getline(file, line);) file_lines.push_back(line);
    cout << "object names loaded \n";
    return file_lines;
}


int sample_main(int argc, char *argv[])
{
    string  names_file = "/home/kid/min/lidar_dl/data/names.list";
    string  cfg_file = "/home/kid/min/lidar_dl/data/model_480_480_tiny/lidar_tiny.cfg";
    string  weights_file = "/home/kid/min/lidar_dl/data/model_480_480_tiny/lidar_tiny_final.weights";
    //string filename = "E:\\work_min\\data\\lidar\\lidar_data\\datas\\feature_3c_anno1\\feature_3c\\2\\1499.png";
	string filename = "/home/kid/min/1.avi";
    float const thresh = 0.25;

    Detector_YOLO detector(cfg_file, weights_file);

    auto obj_names = objects_names_from_file(names_file);
    string out_videofile = "result.avi";
    bool const save_output_videofile = true;
#ifdef TRACK_OPTFLOW
    Tracker_optflow tracker_flow;
    detector.wait_stream = true;
#endif

    while (true)
    {
        cout << "input image or video filename: ";
        if(filename.size() == 0) cin >> filename;
        if (filename.size() == 0) break;

        try {
            extrapolate_coords_t extrapolate_coords;
            bool extrapolate_flag = false;
            float cur_time_extrapolate = 0, old_time_extrapolate = 0;
            preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
            bool show_small_boxes = false;

            string const file_ext = filename.substr(filename.find_last_of(".") + 1); //用来判断输入文件的类型
            string const protocol = filename.substr(0, 7);
            if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" ||     // video file
                protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/")    // video network stream
            {
                Mat cap_frame, cur_frame, det_frame, write_frame;
                queue<Mat> track_optflow_queue;
                int passed_flow_frames = 0;
                shared_ptr<image_t> det_image;
                vector<bbox_t> result_vec, thread_result_vec;
                detector.nms = 0.02;    // comment it - if track_id is not required
                atomic<bool> consumed, videowrite_ready;
                bool exit_flag = false;
                consumed = true;
                videowrite_ready = true;
                atomic<int> fps_det_counter, fps_cap_counter;
                fps_det_counter = 0;
                fps_cap_counter = 0;
                int current_det_fps = 0, current_cap_fps = 0;
                thread t_detect, t_cap, t_videowrite;
                mutex mtx;
                condition_variable cv_detected, cv_pre_tracked;
                chrono::steady_clock::time_point steady_start, steady_end;
                VideoCapture cap(filename); cap >> cur_frame;
                int const video_fps = cap.get(CV_CAP_PROP_FPS);
                Size const frame_size = cur_frame.size();
                VideoWriter output_video;
                if (save_output_videofile)
                    output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), max(35, video_fps), frame_size, true);

                while (!cur_frame.empty())
                {
                    // always sync 抓取
                    if (t_cap.joinable()) {
                        t_cap.join();
                        ++fps_cap_counter;
                        cur_frame = cap_frame.clone();
                    }
                    t_cap = thread([&]() { cap >> cap_frame; });
                    ++cur_time_extrapolate;

                    // 交换 result bouned-boxes 和 input-frame
                    if(consumed)
                    {
                        unique_lock<mutex> lock(mtx);
                        det_image = detector.mat_to_image_resize(cur_frame);
                        auto old_result_vec = detector.tracking_id(result_vec);
                        auto detected_result_vec = thread_result_vec;
                        result_vec = detected_result_vec;
#ifdef TRACK_OPTFLOW
                        // track optical flow
                        if (track_optflow_queue.size() > 0) {
                            //cout << "\n !!!! all = " << track_optflow_queue.size() << ", cur = " << passed_flow_frames << endl;
                            Mat first_frame = track_optflow_queue.front();
                            tracker_flow.update_tracking_flow(track_optflow_queue.front(), result_vec);

                            while (track_optflow_queue.size() > 1) {
                                track_optflow_queue.pop();
                                result_vec = tracker_flow.tracking_flow(track_optflow_queue.front(), true);
                            }
                            track_optflow_queue.pop();
                            passed_flow_frames = 0;

                            result_vec = detector.tracking_id(result_vec);
                            auto tmp_result_vec = detector.tracking_id(detected_result_vec, false);
                            small_preview.set(first_frame, tmp_result_vec);

                            extrapolate_coords.new_result(tmp_result_vec, old_time_extrapolate);
                            old_time_extrapolate = cur_time_extrapolate;
                            extrapolate_coords.update_result(result_vec, cur_time_extrapolate - 1);
                        }
#else
                        result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
                        extrapolate_coords.new_result(result_vec, cur_time_extrapolate - 1);
#endif
                        // add old tracked objects
                        for (auto &i : old_result_vec) {
                            auto it = find_if(result_vec.begin(), result_vec.end(),
                                [&i](bbox_t const& b) { return b.track_id == i.track_id && b.obj_id == i.obj_id; });
                            bool track_id_absent = (it == result_vec.end());
                            if (track_id_absent) {
                                if (i.frames_counter-- > 1)
                                    result_vec.push_back(i);
                            }
                            else {
                                it->frames_counter = min((unsigned)3, i.frames_counter + 1);
                            }
                        }
#ifdef TRACK_OPTFLOW
                        tracker_flow.update_cur_bbox_vec(result_vec);
                        result_vec = tracker_flow.tracking_flow(cur_frame, true);    // track optical flow
#endif
                        consumed = false;
                        cv_pre_tracked.notify_all();
                    }
                    // launch thread once - Detection
                    if (!t_detect.joinable()) {
                        t_detect = thread([&]() {
                            auto current_image = det_image;
                            consumed = true;
                            while (current_image.use_count() > 0 && !exit_flag) {
                                auto result = detector.detect_resized(*current_image, frame_size.width, frame_size.height,
                                    thresh, false);    // true
                                ++fps_det_counter;
                                unique_lock<mutex> lock(mtx);
                                thread_result_vec = result;
                                consumed = true;
                                cv_detected.notify_all();
                                if (detector.wait_stream) {
                                    while (consumed && !exit_flag) cv_pre_tracked.wait(lock);
                                }
                                current_image = det_image;
                            }
                        });
                    }
                    //while (!consumed);    // sync detection

                    if (!cur_frame.empty()) {
                        steady_end = chrono::steady_clock::now();
                        if (chrono::duration<double>(steady_end - steady_start).count() >= 1) {
                            current_det_fps = fps_det_counter;
                            current_cap_fps = fps_cap_counter;
                            steady_start = steady_end;
                            fps_det_counter = 0;
                            fps_cap_counter = 0;
                        }

                        large_preview.set(cur_frame, result_vec);
#ifdef TRACK_OPTFLOW
                        ++passed_flow_frames;
                        track_optflow_queue.push(cur_frame.clone());
                        result_vec = tracker_flow.tracking_flow(cur_frame);    // track optical flow
                        extrapolate_coords.update_result(result_vec, cur_time_extrapolate);
                        small_preview.draw(cur_frame, show_small_boxes);
#endif
                        auto result_vec_draw = result_vec;
                        if (extrapolate_flag) {
                            result_vec_draw = extrapolate_coords.predict(cur_time_extrapolate);
                            putText(cur_frame, "extrapolate", Point2f(10, 40), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(50, 50, 0), 2);
                        }
                        draw_boxes(cur_frame, result_vec_draw, obj_names, current_det_fps, current_cap_fps);
                        //show_console_result(result_vec, obj_names);
                        large_preview.draw(cur_frame);

                        imshow("window name", cur_frame);
                        int key = waitKey(3);    // 3 or 16ms
                        if (key == 'f') show_small_boxes = !show_small_boxes;
                        if (key == 'p') while (true) if(waitKey(100) == 'p') break;
                        if (key == 'e') extrapolate_flag = !extrapolate_flag;
                        if (key == 27) { exit_flag = true; break; }

                        if (output_video.isOpened() && videowrite_ready) {
                            if (t_videowrite.joinable()) t_videowrite.join();
                            write_frame = cur_frame.clone();
                            videowrite_ready = false;
                            t_videowrite = thread([&]() {
                                 output_video << write_frame; videowrite_ready = true;
                            });
                        }
                    }

#ifndef TRACK_OPTFLOW
                    // wait detection result for video-file only (not for net-cam)
                    if (protocol != "rtsp://" && protocol != "http://" && protocol != "https:/") {
                        unique_lock<mutex> lock(mtx);
                        while (!consumed) cv_detected.wait(lock);
                    }
#endif
                }
                exit_flag = true;
                if (t_cap.joinable()) t_cap.join();
                if (t_detect.joinable()) t_detect.join();
                if (t_videowrite.joinable()) t_videowrite.join();
                cout << "Video ended \n";
                break;
            }
            else if (file_ext == "txt") {    // list of image files
                ifstream file(filename);
                if (!file.is_open()) cout << "File not found! \n";
                else
                    for (string line; file >> line;) {
                        cout << line << endl;
                        Mat mat_img = imread(line);
                        vector<bbox_t> result_vec = detector.detect(mat_img);
                        show_console_result(result_vec, obj_names);
                        //draw_boxes(mat_img, result_vec, obj_names);
                        //imwrite("res_" + line, mat_img);
                    }

            }
            else {    // image file
                Mat mat_img = imread(filename);
                auto start = chrono::steady_clock::now();
                vector<bbox_t> result_vec = detector.detect(mat_img);
                auto end = chrono::steady_clock::now();
                chrono::duration<double> spent = end - start;
                cout << " Time: " << spent.count() << " sec \n";

                //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
                draw_boxes(mat_img, result_vec, obj_names);
                imshow("window name", mat_img);
                show_console_result(result_vec, obj_names);
                waitKey(0);
            }

        }
        catch (exception &e) { cerr << "exception: " << e.what() << "\n"; getchar(); }
        catch (...) { cerr << "unknown exception \n"; getchar(); }
        filename.clear();
    }

    return 0;
}
