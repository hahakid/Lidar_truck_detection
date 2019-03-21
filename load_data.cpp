#include "load_data.hpp"
#include <ctime>

using namespace std;
using namespace cv;


imu_origin_t imu_origin;

std::vector<float> normalize_0_255(vector<float> datas,float dMinValue,float dMaxValue)
{
    int length;
//    dMinValue = *min_element(datas.begin(),datas.end());
//    dMaxValue = *max_element(datas.begin(),datas.end());
    length = datas.size();
    float ymax = 255; //归一化数据范围
    float ymin = 0;
    vector<float> features;
    float tmp;
    for (int d = 0; d < length; ++d)
    {
        tmp = (ymax-ymin)*(datas[d]-dMinValue)/(dMaxValue-dMinValue+1e-8)+ymin;
        //cout<<"> "<<tmp<<"\n"<<endl;
        features.push_back(tmp);
    }
    return features;
}
//数组拼接
float *SortArry(float *StrA,int lenA, float *StrB, int lenB)
{
    if (StrA == NULL || StrB == NULL)
        return NULL;

    float *StrC = new float[lenA + lenB+1];
    int i, j, k;
    i = j = k = 0;
//    while (i < lenA && j < lenB)
//    {
//        if (StrA[i] < StrB[j]) StrC[k++] = StrA[i++];
//        else StrC[k++] = StrB[j++];
//    }

    while (i<lenA)
    {
        StrC[k++] = StrA[i++];
    }

    while (j<lenB)
    {
        StrC[k++] = StrB[j++];
    }

    return StrC;
}


//pcl::PointCloud<pcl::PointXYZI>::Ptr passthrough_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo,bool is_gro)
//{
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
//    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZI>);
//
//    pcl::PassThrough<pcl::PointXYZI> pass;
//
//    pass.setInputCloud(point_cloud_velo);
//    pass.setFilterFieldName("x");
//    pass.setFilterLimits(MINX,MAXX);
//    pass.filter(*cloud_filtered_x);
//
//    //pass.setFilterLimitsNegative (true);
//    pass.setInputCloud(cloud_filtered_x);
//    pass.setFilterFieldName("y");
//    pass.setFilterLimits(MINY,MAXY);
//    pass.filter(*cloud_filtered_y);
//
//    if (is_gro == true)
//    {
//        pass.setInputCloud(cloud_filtered_y);
//        pass.setFilterFieldName("z");
//        pass.setFilterLimits(GROUND_LIMIT_MIN,GROUND_LIMIT_MAX);
//        pass.filter(*cloud_filtered);
//        return cloud_filtered;
//
//    }else{
//        return cloud_filtered_y;
//    }
//}

void passthrough_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo,bool is_gro)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::PassThrough<pcl::PointXYZI> pass;

    pass.setInputCloud(point_cloud_velo);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(MINX,MAXX);
    pass.filter(*cloud_filtered_x);

    //pass.setFilterLimitsNegative (true);
    pass.setInputCloud(cloud_filtered_x);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(MINY,MAXY);
    pass.filter(*cloud_filtered_y);

    if (is_gro == true)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        pass.setInputCloud(cloud_filtered_y);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(GROUND_LIMIT_MIN,GROUND_LIMIT_MAX);
        pass.filter(*cloud_filtered);

        pcl::copyPointCloud(*cloud_filtered,*point_cloud_velo);
        //return cloud_filtered;

    }else{
        pcl::copyPointCloud(*cloud_filtered_y,*point_cloud_velo);
        //return cloud_filtered_y;
    }
}



velo_data_t read_velo_data(string velo_filename160,string velo_filename161)
{
    int arr_counts[1] = {0};
    int counts160 = 0, counts161 = 0, counts = 0;

    velo_data_t points_velo;
    points_velo.counts = 0;

    ifstream fin160(velo_filename160, ios::binary);
	ifstream fin161(velo_filename161, ios::binary);
	if(!fin160 && !fin161)
	{
		cout << "read velo error\n" <<endl;
		return points_velo;
	}else
	{
        fin160.read((char*)arr_counts,sizeof(int));
        if (arr_counts[0] != 0)
            counts160 = arr_counts[0];
        cout<<"points counts160 = "<<counts160<<endl;
        fin161.read((char*)arr_counts,sizeof(int));
        if (arr_counts[0] != 0)
            counts161 = arr_counts[0];
        cout<<"points counts161 = "<<counts161<<endl;

        counts = counts160+counts161;
        points_velo.counts = counts;
        points_velo.x = new float[counts];
        points_velo.y = new float[counts];
        points_velo.z = new float[counts];
        points_velo.r = new int[counts];

        float xx160[counts160],xx161[counts161];
        float yy160[counts160],yy161[counts161];
        float zz160[counts160],zz161[counts161];
        vector<int> rr_tmp;
        char rr_tmp_char;
        int rr_tmp_int;

        //int *rr=new int[counts];

        fin160.read((char*)xx160,counts160*sizeof(float));
        fin160.read((char*)yy160,counts160*sizeof(float));
        fin160.read((char*)zz160,counts160*sizeof(float));
        fin161.read((char*)xx161,counts161*sizeof(float));
        fin161.read((char*)yy161,counts161*sizeof(float));
        fin161.read((char*)zz161,counts161*sizeof(float));
        int k=0;
        for(int i=0;i<counts160;i++)
        {
            fin160.read((char*)&rr_tmp_char,sizeof(char));
            rr_tmp_int = rr_tmp_char*1;
            points_velo.r[k++] = rr_tmp_int;
        }
        for (int i=0;i<counts161;i++)
        {
            fin161.read((char*)&rr_tmp_char,sizeof(char));
            rr_tmp_int = rr_tmp_char*1;
            points_velo.r[k++] = rr_tmp_int;
        }
        fin160.close();fin161.close();
        points_velo.x = SortArry(xx160,counts160,xx161,counts161);
        points_velo.y = SortArry(yy160,counts160,yy161,counts161);
        points_velo.z = SortArry(zz160,counts160,zz161,counts161);

        return points_velo;
	}

    fin160.close();
    fin161.close();
}

imu_data_t read_imu_data(string imu_filename)
{
    ifstream fin(imu_filename, ios::in);
    double lat,lon,direction,v,rtk;
    imu_data_t imu_datas;
    while(!fin.eof())
    {
        fin>>lat>>lon>>direction>>v>>rtk;
        imu_datas.lat = lat;
        imu_datas.lon = lon;
        imu_datas.direction = direction;
    }
    fin.close();
    return imu_datas;
}
//void get_img(cv::Mat& img_src,velo_data_t velo_points)
//{
//    int user_data = 0;
//
////	string velo_filename160 = "/home/kid/min/Annotations/LiDar/anno2/veloseq/1/VLP160/324.bin";
////	string velo_filename161 = "/home/kid/min/Annotations/LiDar/anno2/veloseq/1/VLP161/324.bin";
//
//    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZI>);
//
//    //point_cloud_velo->width    = 5182;
//    point_cloud_velo->width    = velo_points.counts;
//    point_cloud_velo->height   = 1;
//    point_cloud_velo->is_dense = false;  //不是稠密型的
//    point_cloud_velo->points.resize(point_cloud_velo->width*point_cloud_velo->height);
//
//    for (int i=0;i < velo_points.counts;i++ )
//    {
//        point_cloud_velo->points[i].x = velo_points.y[i];
//        point_cloud_velo->points[i].y = velo_points.x[i];
////            point_cloud_velo->points[i].z = zz[i]-MINZ;
//        point_cloud_velo->points[i].z = velo_points.z[i];
//        point_cloud_velo->points[i].intensity = velo_points.r[i];
//    }
//
//
////    for (int i = 0; i < point_cloud_velo->points.size(); ++i)
////    {
////        std::cerr << ">>>>>" << xx[i]<<"<->"<<point_cloud_velo->points[i].x << "," <<yy[i]<<"<->"<< point_cloud_velo->points[i].y << "," << xx[i]<<"<->"<< point_cloud_velo->points[i].z << std::endl;
////    }
////    for (int i = 0; i < point_cloud_velo->points.size(); ++i)
////    {
////        std::cerr << ">>>>>" << rr[i]<<"<->"<<point_cloud_velo->points[i].intensity<<std::endl;
////    }
//
//    // ********************
//
//
//
//    //地面滤出＋边界滤出
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
//    cloud_filtered = passthrough_filter(point_cloud_velo);
//
//
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_d(new pcl::PointCloud<pcl::PointXYZ>);
//
////    pcl::copyPointCloud(*cloud_d, *cloud_filtered);
//    cout<<"remain points = "<<cloud_filtered->points.size()<<endl;
//
//    vector<float>densityMap,nol_densityMap;
//    int M = cloud_filtered->points.size();
//    for (int i = 0;i <M;i++)
//    {
//        pcl::PointXYZ p;
//        p.x = cloud_filtered->points[i].x*100.0;
//        p.y = cloud_filtered->points[i].y*100.0;
//        p.z = cloud_filtered->points[i].z*100.0;
//        cloud_d->points.push_back(p);
//    }
//    cloud_d->width = 1;
//    cloud_d->height = M;
//    //cout<<">>>"<<cloud_d->points.size()<<endl;
//    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //创建一个快速k近邻查询,查询的时候若该点在点云中，则第一个近邻点是其本身
//    kdtree.setInputCloud(cloud_d);
//    float everagedistance =0;
//    for (int i =0; i < cloud_d->size();i++)
//    {
//        //cout<<">>>>> i = "<<i<<endl;
//        vector<int> nnh ;
//        vector<float> squaredistance;
//        pcl::PointXYZ p;
//        p = cloud_d->points[i];
//        kdtree.radiusSearch(p,1,nnh,squaredistance);
//        int tmp = nnh.size() ;
//        //if (tmp > 1) cout<<"i = "<<i<<", density = "<<tmp<<"\n"<<endl;
//
//        if (tmp > 1)
//        {
//            float tmp_ = min(1.0,log((float)tmp)/log(64));
//            densityMap.push_back(tmp_);
//        }
//        else densityMap.push_back(0.0);
//    }
//
//    //everagedistance = everagedistance/(cloud->size()/2);
//
//
//    //*********************
//    //# 转换为像素位置的值 - 基于分辨率
//    int res = 480;
//    float xp = (MAXY-MINY) / (float)res;
//    float yp = (MAXX-MINX) / (float)res;
//    cout<<"xp = "<<xp<<" , yp = "<<yp<<"\n"<<endl;
//    int size_cloud_filtered = cloud_filtered->points.size();
//    int ximg[size_cloud_filtered] = {0};
//    int yimg[size_cloud_filtered] = {0};
//    vector<float> z_points(size_cloud_filtered),nol_z_points;
//    vector<float> intensityMap(size_cloud_filtered),nol_intensityMap;
//    //vector<float> z_points(cloud_filtered->points.z);
//
//
//    int px,py;//用来指向图像位置
//    for(int j=0;j<size_cloud_filtered;j++)
//    {
//        ximg[j] = (int)((cloud_filtered->points[j].y - MINY)/ yp);
//        yimg[j] = res-1-(int)((cloud_filtered->points[j].x - MINX)/ xp);
//        px = ximg[j];
//        py = yimg[j];
//        if(px>=res||py>=res||px<0||py<0)
//        {
//            cout<<">> cloud_filtered->points[j].y="<<cloud_filtered->points[j].y<<", cloud_filtered->points[j].x="<<cloud_filtered->points[j].x<<"\n"<<endl;
//            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
//        }
////        ximg[j] = (int)((cloud_filtered->points[j].y)/ yp);
////        yimg[j] = (int)((cloud_filtered->points[j].x)/ xp);
//        z_points.push_back(cloud_filtered->points[j].z);
//        intensityMap.push_back(cloud_filtered->points[j].intensity);
//    }
//    nol_z_points = normalize_0_255(z_points,MINZ,MAXZ);
//    float d_min = *min_element(intensityMap.begin(), intensityMap.end());
//    float d_max = *max_element(intensityMap.begin(), intensityMap.end());
//    nol_intensityMap = normalize_0_255(intensityMap,0,d_max);
//    d_min = *min_element(densityMap.begin(), densityMap.end());
//    d_max = *max_element(densityMap.begin(), densityMap.end());
//    nol_densityMap = normalize_0_255(densityMap,0,d_max);
//    //float density_max = *max_element(densityMap.begin(), densityMap.end());
//    Mat img(res, res, CV_8UC3,Scalar(0,0,0));  //
//    //img(res, res, CV_8UC3,Scalar(0,0,0));
//    for (int i=0;i<size_cloud_filtered;i++)
//    {
//        px = ximg[i];
//        py = yimg[i];
//        if(px>=res||py>=res||px<0||py<0)
//        {
//            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
//        }else //BGR
//        {
//            //img.at<Vec3b>(py,px)[0] = nol_densityMap[i];
//            img.at<Vec3b>(py,px)[0] = 0;
//            img.at<Vec3b>(py,px)[1] = nol_z_points[i];
//            //img.at<Vec3b>(py,px)[2] = nol_intensityMap[i];
//            img.at<Vec3b>(py,px)[2] = 0;
//            //img.at<uchar>(py,px) = nol_z_points[i];
//        }
//
//    }
//
//
//
//
////    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
////	viewer->setBackgroundColor(0, 0, 0);
////	viewer->addPointCloud<pcl::PointXYZI>(point_cloud_velo, "sample cloud");
////	//viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "sample cloud");
////	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
////	viewer->addCoordinateSystem(1.0);
////	viewer->initCameraParameters();
////
////	while (!viewer->wasStopped())
////	{
////		viewer->spinOnce(100);
////		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
////	}
//
//    img_src = img;
//        //return (1);
//
//}
//
void get_img(cv::Mat& img_src,pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo)
{

    int size_cloud = point_cloud_velo->points.size();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_d(new pcl::PointCloud<pcl::PointXYZ>);

    vector<float>densityMap,nol_densityMap;

//    for (int i = 0;i < size_cloud;i++)
//    {
//        pcl::PointXYZ p;
//        p.x = point_cloud_velo->points[i].x*100.0;
//        p.y = point_cloud_velo->points[i].y*100.0;
//        p.z = point_cloud_velo->points[i].z*100.0;
//        cloud_d->points.push_back(p);
//    }
//    cloud_d->width = 1;
//    cloud_d->height = size_cloud;
    pcl::copyPointCloud(*point_cloud_velo,*cloud_d);
    //cout<<">>>"<<cloud_d->points.size()<<endl;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //创建一个快速k近邻查询,查询的时候若该点在点云中，则第一个近邻点是其本身
    kdtree.setInputCloud(cloud_d);
    float everagedistance =0;
    for (int i =0; i < cloud_d->size();i++)
    {
        //cout<<">>>>> i = "<<i<<endl;
        vector<int> nnh ;
        vector<float> squaredistance;
        pcl::PointXYZ p;
        p = cloud_d->points[i];
        kdtree.radiusSearch(p,0.1,nnh,squaredistance);
        int tmp = nnh.size() ;
        //if (tmp > 1) cout<<"i = "<<i<<", density = "<<tmp<<"\n"<<endl;

        if (tmp > 1)
        {
            float tmp_ = min(1.0,log((float)tmp)/log(64));
            densityMap.push_back(tmp_);
        }
        else densityMap.push_back(0.0);
    }


    //*********************
    //# 转换为像素位置的值 - 基于分辨率
    int res = 480;
    float xp = (MAXY-MINY) / (float)res;
    float yp = (MAXX-MINX) / (float)res;
    cout<<"xp = "<<xp<<" , yp = "<<yp<<"\n"<<endl;

    int ximg[size_cloud] = {0};
    int yimg[size_cloud] = {0};
    vector<float> z_points(size_cloud),nol_z_points;
    vector<float> intensityMap(size_cloud),nol_intensityMap;
    //vector<float> z_points(cloud_filtered->points.z);


    int px,py;//用来指向图像位置
    for(int j=0;j<size_cloud;j++)
    {
        ximg[j] = (int)((point_cloud_velo->points[j].y - MINY)/ yp);
        yimg[j] = res-1-(int)((point_cloud_velo->points[j].x - MINX)/ xp);
//        px = ximg[j];
//        py = yimg[j];
//        if(px>=res||py>=res||px<0||py<0)
//        {
//            cout<<">> point_cloud_velo->points[j].y="<<point_cloud_velo->points[j].y<<", cloud_filtered->points[j].x="<<point_cloud_velo->points[j].x<<"\n"<<endl;
//            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
//        }
        z_points.push_back(point_cloud_velo->points[j].z);
        intensityMap.push_back(point_cloud_velo->points[j].intensity);
    }
    nol_z_points = normalize_0_255(z_points,MINZ,MAXZ);
    //float d_min = *min_element(intensityMap.begin(), intensityMap.end());
    float d_max = *max_element(intensityMap.begin(), intensityMap.end());
    nol_intensityMap = normalize_0_255(intensityMap,0,d_max);
    //d_min = *min_element(densityMap.begin(), densityMap.end());
    d_max = *max_element(densityMap.begin(), densityMap.end());
    nol_densityMap = normalize_0_255(densityMap,0,d_max);
    //float density_max = *max_element(densityMap.begin(), densityMap.end());


    cv::Mat img(res, res, CV_8UC3,Scalar(0,0,0));  //

    //img(res, res, CV_8UC3,Scalar(0,0,0));
    for (int i=0;i<size_cloud;i++)
    {
        px = ximg[i];
        py = yimg[i];
        img.at<Vec3b>(py,px)[0] = nol_densityMap[i];
        //img.at<Vec3b>(py,px)[0] = 0;
        img.at<Vec3b>(py,px)[1] = nol_z_points[i];
        img.at<Vec3b>(py,px)[2] = nol_intensityMap[i];
        //img.at<Vec3b>(py,px)[2] = 0;
        //img.at<uchar>(py,px) = nol_z_points[i];


//        if(px>=res||py>=res||px<0||py<0)
//        {
//            cout<<">> px="<<px<<", py="<<py<<"\n"<<endl;
//        }else //BGR
//        {
//            img.at<Vec3b>(py,px)[0] = nol_densityMap[i];
//            //img.at<Vec3b>(py,px)[0] = 0;
//            img.at<Vec3b>(py,px)[1] = nol_z_points[i];
//            img.at<Vec3b>(py,px)[2] = nol_intensityMap[i];
//            //img.at<Vec3b>(py,px)[2] = 0;
//            //img.at<uchar>(py,px) = nol_z_points[i];
//        }



    }
    img_src = img;
}


//int LOAD_LIDAR_DATA::lidar_process(vector<velo_data_t> velo_points)
pcl::PointCloud<pcl::PointXYZI>::Ptr process_merged(vector<velo_data_t> velo_points,vector<imu_data_t> imu_data)
{
    clock_t startTime,endTime;

    double origin_x;
    double origin_y;
    double xy_scale_factor = 1;//0.8505

    pcl::PointCloud<pcl::PointXYZI>::Ptr src(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr Final(new pcl::PointCloud<pcl::PointXYZI>);
    for(int k= 0;k<OVERLOP_NUM;k++)
    {
        //*************** 获取点
        startTime = clock();
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_velo(new pcl::PointCloud<pcl::PointXYZI>);
        //point_cloud_velo->width    = 5182;
        point_cloud_velo->width    = velo_points[k].counts;
        point_cloud_velo->height   = 1;
        point_cloud_velo->is_dense = false;  //不是稠密型的
        point_cloud_velo->points.resize(point_cloud_velo->width*point_cloud_velo->height);
        for (int i=0;i < velo_points[k].counts;i++ )
        {
            point_cloud_velo->points[i].x = velo_points[k].y[i];
            point_cloud_velo->points[i].y = velo_points[k].x[i];
            //            point_cloud_velo->points[i].z = zz[i]-MINZ;
            point_cloud_velo->points[i].z = velo_points[k].z[i];
            point_cloud_velo->points[i].intensity = velo_points[k].r[i];
        }
        endTime = clock();
        cout << "getpoints run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//        for (int i=0;i < velo_points[k].counts;i++ )
//            cout<<">>> points: x="<<point_cloud_velo->points[i].x<<"<----->"<<velo_points[k].y[i]<<endl;
        //************** 地面滤出＋边界滤出
        //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        //cloud_filtered = passthrough_filter(point_cloud_velo,true);
        passthrough_filter(point_cloud_velo,true);
        //pcl::copyPointCloud(*point_cloud_velo,*cloud_filtered);

        //***********处理并转换imu数据
        startTime = clock();
        imu_data_t merc = trans_imu_data(imu_data[k].lat,imu_data[k].lon,imu_data[k].direction);
        if (k == 0)
        {
            origin_x = merc.lat;origin_y = merc.lon;
            pcl::copyPointCloud(*point_cloud_velo, *tgt);
            continue;
        }
        float d_x = (merc.lat - origin_x)*xy_scale_factor;
        float d_y = (merc.lon - origin_y)*xy_scale_factor;
        //cout<<"d_x="<<d_x<<"d_y"<<d_y<<endl;
//        endTime = clock();
        cout << "trans_imu run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
        //************平移velo
        startTime = clock();
//        int M = point_cloud_velo->points.size();
//        for (int i=0;i < M;i++ )
//        {
//            pcl::PointXYZI p;
//            float x = point_cloud_velo->points[i].x + d_x;
//            float y = point_cloud_velo->points[i].y + d_y;
//            p.x = x;
//            p.y = y;
//            p.z = point_cloud_velo->points[i].z;
//            p.intensity = point_cloud_velo->points[i].intensity;
//            src->points.push_back(p);
//        }
//        src->width = 1;
//        src->height = M;
        Eigen::Matrix4f transform_xy = Eigen::Matrix4f::Identity();
        transform_xy(0,3) = d_x;
        transform_xy(1,3) = d_y;
        pcl::transformPointCloud(*point_cloud_velo, *src, transform_xy);

        endTime = clock();
        cout << "velo_move and trans_imu run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
        //*************配准
        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;   //创建IterativeClosestPoint的对象
        icp.setInputCloud(src);                 //cloud_in设置为点云的源点
        icp.setInputTarget(tgt);               //cloud_out设置为与cloud_in对应的匹配目标
        icp.align(*Final);                             //打印配准相关输入信息

        //# 合并配准后的数据
        startTime = clock();
//        int S = Final->points.size();
//        int L = tgt->points.size();
//        tgt->points.resize(S+L);
//        for (int i=0;i < S;i++ ) //合并配准后的数据 tgt = tgt+final
//        {
//            pcl::PointXYZI p_tmp;
//            p_tmp.x = Final->points[i].x;
//            p_tmp.y = Final->points[i].y;
//            p_tmp.z = Final->points[i].z;
//            p_tmp.intensity = Final->points[i].intensity;
//            tgt->points.push_back(p_tmp);
//        }
        *tgt = *tgt+*Final;
        endTime = clock();
        cout << "match run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//
//        std::cout << "has converged:" << icp.hasConverged() << " score: " <<
//        icp.getFitnessScore() << std::endl;
//        std::cout << icp.getFinalTransformation() << std::endl;
    }

    //************** 地面滤出＋边界滤出
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_second(new pcl::PointCloud<pcl::PointXYZI>);
//    cloud_filtered_second = passthrough_filter(tgt,true);
    startTime = clock();
    passthrough_filter(tgt,true);
    endTime = clock();
    cout << "passthrough_filter run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

    return tgt;
//    return cloud_filtered_second;
}
//int LOAD_LIDAR_DATA::process_merged()

//void move_velo()
//{
//    Eigen::Matrix4f transform_xy = Eigen::Matrix4f::Identity();
//    transform_xy(0,3) = 1;
//    transform_xy(1,3) = 1;
//}




//int LOAD_LIDAR_DATA::process_imu_data(double lat, double lon, double direction)
imu_data_t trans_imu_data(double lat, double lon, double direction)
{
    // lat 经度　lon 维度
    imu_data_t imu_data;
    projPJ pj_merc, pj_latlong;

    // +proj=tmerc +lat_0=0 +lon_0=120 +k=1 +x_0=500000 +y_0=0 +a=6378140 +b=6356755.288157528 +units=m +no_defs  epsg:2385 东经118.30 - 121.30
    // +proj=tmerc +lat_0=0 +lon_0=123 +k=1 +x_0=500000 +y_0=0 +a=6378140 +b=6356755.288157528 +units=m +no_defs  epsg:2386 东经121.30 - 124.30
	if (!(pj_merc = pj_init_plus("+proj=tmerc +lat_0=0 +lon_0=123 +k=1 +x_0=500000 +y_0=0 +a=6378140 +b=6356755.288157528 +units=m +no_defs")))
		exit(1);
	if (!(pj_latlong = pj_init_plus("+proj=longlat +datum=WGS84 +no_defs"))) //WGS84这个GPS所用的系统
		exit(1);


	pj_transform(pj_latlong, pj_merc, 1, 1, &lat, &lon, NULL);

    imu_data.lat = lat;
    imu_data.lon = lon;
    imu_data.direction = direction;

	//std::cout << "(" << lat << " , " << lon << ")" << std::endl;
	return imu_data;
}
