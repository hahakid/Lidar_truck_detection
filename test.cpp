#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <Eigen/Core>
using namespace std;

int ___main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 随机点云生成
    cloud->width=1000;                                 //此处为点云数量
    cloud->height=1;                                   //此处表示点云为无序点云
    cloud->points.resize(cloud->width*cloud->height);
    for(size_t i=0;i<cloud->points.size();++i)      //循环填充点云数据
    {
        cloud->points[i].x=1024.0f*rand()/(RAND_MAX+1.0f);
        cloud->points[i].y=1024.0f*rand()/(RAND_MAX+1.0f);
        cloud->points[i].z=1024.0f*rand()/(RAND_MAX+1.0f);
    }


    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //创建一个快速k近邻查询,查询的时候若该点在点云中，则第一个近邻点是其本身
    kdtree.setInputCloud(cloud);
    int k =2;
    float everagedistance =0;
    for (int i =0; i < cloud->size()/2;i++)
    {
            vector<int> nnh ;
            vector<float> squaredistance;
            //  pcl::PointXYZ p;
            //   p = cloud->points[i];
            kdtree.nearestKSearch(cloud->points[i],k,nnh,squaredistance);
            everagedistance += sqrt(squaredistance[1]);
            //   cout<<everagedistance<<endl;
    }

    everagedistance = everagedistance/(cloud->size()/2);
    cout<<"everage distance is : "<<everagedistance<<endl;
}

