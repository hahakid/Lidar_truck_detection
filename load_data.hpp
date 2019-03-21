#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include<iostream>
#include<fstream>
#include<string>
#include<cmath>

//#include<boost/thread.hpp>
//#include<boost/timer.hpp>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/point_types.h>

//#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/ModelCoefficients.h>
//#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>

//#include <pcl/common/transforms.h>
//#include <pcl/correspondence.h>

#include <opencv2/opencv.hpp>


#include <proj_api.h>

//#define OVERLOP_NUM 3
////点云边
#define MINX -10.0
#define MAXX 50.0
#define MINY -30.0
#define MAXY 30.0
#define MINZ -1.6
#define MAXZ 2.4

#define GROUND_LIMIT_MIN 0.5
#define GROUND_LIMIT_MAX 5
#define OVERLOP_NUM 3//
#define RES 480 //创建图片像素


struct velo_data_t {
    int counts;
	float *x;
	float *y;
	float *z;
	int *r;
};
struct imu_origin_t{
    double x;
    double y;
    //imu_origin_t(){};
};
struct imu_data_t{
    double lat;
    double lon;
    double direction;
    //imu_origin_t(){};
};


pcl::PointCloud<pcl::PointXYZI>::Ptr process_merged(std::vector<velo_data_t> velo_points,std::vector<imu_data_t> imu_data);
imu_data_t trans_imu_data(double lat, double lon, double direction);


//class LOAD_LIDAR_DATA{
//
//
// public:
//	imu_origin_t imu_origin;
//	double lan_origin = 0.0;
//	double lon_origin = 0.0;
//	std::vector<velo_data_t> points_velo_list;
//
//    LOAD_LIDAR_DATA();
//    ~LOAD_LIDAR_DATA();
//    int main_process(velo_data_t velo_points);
//    int process_imu_data(double lat, double lon, double direction);
//    int process_merged();
//
//};


void get_img(cv::Mat& img_src,pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo);

velo_data_t read_velo_data(std::string velo_filename160,std::string velo_filename161);
imu_data_t read_imu_data(std::string imu_filename);

//pcl::PointCloud<pcl::PointXYZI>::Ptr passthrough_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo,bool is_gro);
void passthrough_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo,bool is_gro);




#endif // LOAD_DATA_HPP
