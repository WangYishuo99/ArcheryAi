#ifndef _STARPOSERECG_H_
#define _STARPOSERECG_H_
#include <iostream>
#include <math.h>
#include "opencv2/opencv.hpp"
#include <chrono>
#include "common.hpp"
#include <fstream>

using namespace std::chrono;
using namespace pose;


enum ActiontYPE
{
    PREPARE=0,
    UPBOW,
    DRAWBOW,
    FREE,
    COMPLETE
};

struct data_package_
{
    std::string action_type_text;
    float action_score;
    int action_times;
    float action_up_score;
    float action_pull_score;  
    int shooting_score ;
};
using data_package = struct data_package_;


template <typename T>
struct Point{
    T x;
    T y;
};

template <typename T>
class MathVector{
public:
    T vx;
    T vy;
    Point<T> pt1;
    Point<T> pt2;
public:
    MathVector(Point<T> P1,Point<T> P2)
    {
        this->pt1=P1;
        this->pt2=P2;
        this->vx=P2.x-P1.x;
        this->vy=P2.y-P1.y;
    }
    
    T dot(MathVector& vec)
    {
        return  vx*vec.vx+vy*vec.vy;
    }

    T norm()
    {
       return sqrt(vx*vx+vy*vy); 
    }
    
}; 

class ActionRecognition{
private:
    int action_counter;
    int num_in_cycle;
    int lost_frame_counter;
    std::pair<int,int> action_duration_record;
    std::ofstream outputFile;
    //prepare 
    ActiontYPE last_action_type; 
    std::vector<float> features;

    float score_upbow_history;
    float score_drawbow_history;

    //times
    int up_bow_times;
    int draw_bow_times;
    int condition_num ;
    std::vector<float> up_score_history;
    std::vector<float> pull_score_history;

    std::vector<cv::Point> roi_polygon_vec;
    cv::Rect roi_rects_vec;

    float up_score;
    float pull_score;
    float score_record;
    
    int s_score ;//shooting score
private:
    double Angle(MathVector<float>& vec1,MathVector<float>& vec2); 
    /************************************ 
    * @brief 计算两个向量的夹角
    * @param vec1
    * @param vec2     
    * @return 返回说明    夹角角度[0-180]
     **********************************/
    double vetorial_angle(MathVector<float>& vec1,MathVector<float>& vec2);
    
    /************************************ 
    * @brief 计算上臂和身体的夹角
    * @param vec1
    * @param vec2     
    * @return 返回说明    夹角角度[0-180]
     **********************************/
    double arm_bodyside_angle(MathVector<float>& vec1,MathVector<float>& vec2);
    
    /************************************ 
    * @brief 计算左右胳膊的相对位置
    * @param keypoint 
    * @return 返回说明    
     **********************************/
    int arm_location(Object& keypoint);
    int ActionRecognition::arm_location_relative(Object& keypoint);

    /************************************ 
    * @brief举弓动作识别
    * @param keypoint 
    * @return 返回说明 1 举弓，0其他
     **********************************/
    ActiontYPE action_recognize_interface(Object& keypoint,cv::Mat& src_img);

    /************************************ 
    * @brief 扭头判断
    * @param keypoint 
    * @return 返回说明 正面 0 ，左边 1 右边 2
     **********************************/
    int head_location(Object& obj,cv::Mat& src_img);
    int hand_location(Object& obj,cv::Mat& src_img);
    /************************************ 
    * @brief 计算两点之间的距离
    * @param keypoint 
    * @return 返回说明 像素级距离
     **********************************/
    float points_distance(Point<float> p1, Point<float> p2);

    /************************************ 
    * @brief 余弦相似度
    * @param  
    * @return
     **********************************/
    float cos_distance(const std::vector<float>& vec1, const std::vector<float>& vec2);
    
    /************************************ 
    * @brief 简单过滤掉非目标框
    * @param  
    * @return
     **********************************/
    int filter_other(std::vector<Object>& objs);
    
     /************************************ 
    * @brief 欧式距离
    * @param  
    * @return
     **********************************/
    float euc_distance(std::vector<float>& _f1,std::vector<float>& _f2);
    bool is_in_poly(cv::Rect& roi_rects, std::vector<cv::Point> &roi_polygon, cv::Point &po);

    int dist_map(cv::Mat& d_mat,std::vector<float>& score);
public:
    int Init(std::vector<cv::Point> &roi_polygon);
    ActionRecognition();
    ~ActionRecognition();
    float AnalysisInterface(std::vector<Object>& obj_keypoints,cv::Mat& src_img,data_package& package);
    

private:
    // std::vector<POSE_ANGLE_INFO> angleTrace;
    
};

#endif