#ifndef POSE_UTILS_HPP
#define POSE_UTILS_HPP
#include "opencv2/opencv.hpp"
#define image_show_x 250
#define image_show_y 300
#define text_show_x  248

int DrawRegion(cv::Mat& src_img,std::vector<cv::Point> roi_polygons,float ratio=1.0);
int DrawPoint(cv::Mat& src_img,std::vector<cv::Point> roi_polygons,float ratio=1.0);



#endif