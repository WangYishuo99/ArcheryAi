#include "../include/utils.hpp"
#include "../include/common.hpp"
int DrawRegion(cv::Mat& src_img,std::vector<cv::Point> roi_polygons,float ratio)
{
    if(src_img.empty())
    {
        std::cout<<RED<<"Source image is empty!"<<RESET<<std::endl;
        return -1;
    }

    int poly_size = roi_polygons.size();
    int next_point_pos = 0;
    
    int img_width = src_img.cols;
    int img_height = src_img.rows;
    std::vector<cv::Point> reg;
    int thickness = 2;
    if(ratio != 1.)
    {
        for(int i =0;i < poly_size ; i++)
        {
            cv::Point pt = roi_polygons[i];
            pt.x = (pt.x-image_show_x) / ratio;
            pt.y = (pt.y-image_show_y) / ratio;
            reg.push_back(pt);
        }
    }
    else{
        roi_polygons.swap(reg);
        thickness =1;
    }
    //TODO 目前只支持画第一个区域
    std::vector<int> hull;		//存储一个凸包的边的一维数组	
    cv::convexHull(cv::Mat(reg),hull,true);
    int count = 4;
    for (int i = 0;i < count;i++)
    {
        circle(src_img,reg[i],3, cv::Scalar(255,0,0),cv::FILLED,cv::LINE_AA);
    }

    int hullcount = (int)hull.size();		            //凸包的边数(因为只有一个凸包，而凸包是由边构成的序列，所以返回序列长度，应该返回的是边的个数)
    cv::Point point0 = reg[hull[hullcount - 1]];		//连接凸包边的坐标点		最后一条边的坐标点

    //绘制凸包的边
    for (int i = 0;i < hullcount;i++)
    {
        cv::Point point = reg[hull[i]];		//points[hull[i]]表示构成凸包边的某点（因为凸包是一个点集合最外面的点连接起来的区域）
        line(src_img, point0, point, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        point0 = point;
    }

    // for(int i =0;i < poly_size ; i++)
    // {
    //     cv::Point corner_p;
    //     next_point_pos = i+1;
    //     if(poly_size == next_point_pos)
    //     {
    //         next_point_pos = 0;
    //     }
    //     cv::line(src_img,reg[i],reg[next_point_pos],cv::Scalar(0,255,0),thickness,cv::LINE_AA);
    // }

    return 1;
}

int DrawPoint(cv::Mat& src_img,std::vector<cv::Point> roi_polygons,float ratio)
{
    if(src_img.empty())
    {
        std::cout<<RED<<"Source image is empty!"<<RESET<<std::endl;
        return -1;
    }
   
    int poly_size = roi_polygons.size();
    int radius = 5;
    int img_width = src_img.cols;
    int img_height = src_img.rows;
    std::vector<cv::Point> reg;
    if(ratio != 1.)
    {
        for(int i =0;i < poly_size ; i++)
        {
            cv::Point pt = roi_polygons[i];
            pt.x = (pt.x-image_show_x) / ratio;
            pt.y = (pt.y-image_show_y) / ratio;
            reg.push_back(pt);
        }
    }
    else{
        roi_polygons.swap(reg);
        radius = 5;
    }

    for(int i =0;i < poly_size ; i++)
    {
       cv::circle(src_img, reg[i], radius, cv::Scalar(255,0,0), -1);
    }
    return 1;
}