#include <chrono>
#include "opencv2/opencv.hpp"
#include "../include/yolov8-pose.hpp"
#include "../include/action_recognition.hpp"
#include "../include/utils.hpp"

#define CVUI_IMPLEMENTATION
#include "../include/cvui.h"

#define WINDOW_NAME_VIDEO "stream"
#include <iostream>
#include <opencv2/opencv.hpp>


#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>

#include "../include/mac.h"
// using namespace cv;

#define VIDEO_WIDTH 1920
#define VIDEO_HEIGHT 1080
float anglePI = 180.0 * CV_PI / 180;
int transfer_of_axes(std::vector<float>& sample1,std::vector<float>& sample2)
{
    std::vector<float> transfer_x;
    std::vector<float> transfer_y;
    int sample_size =  sample1.size();
    for (int i =0;i< sample_size;i++)
    {
        int x = (sample1[i]-VIDEO_WIDTH/2)*cosf(anglePI)+(sample2[i]-VIDEO_HEIGHT/2)*sinf(anglePI)+VIDEO_WIDTH/2;
        int y = (sample2[i]-VIDEO_HEIGHT/2)*cosf(anglePI)-(sample1[i]-VIDEO_WIDTH/2)*sinf(anglePI)+VIDEO_HEIGHT/2;
        transfer_x.push_back(x);
        transfer_y.push_back(y);
        std::cout<<"x :"<<x<<","<<"y :"<<y<<"\n";
    }
    return 0;
}

int covariance_variable(std::vector<float>& sample1,std::vector<float>& sample2)
{
    if(sample1.size() != sample2.size())
    {
        return -1;
    }
    int sample_size =  sample1.size();
    float sample1_mean=0,sample2_mean=0;
    for (int i =0;i< sample_size;i++)
    {
        sample1_mean+=sample1[i];
        sample2_mean+=sample2[i];
    }
    sample1_mean/=sample_size;
    sample2_mean/=sample_size;

    float var1=0,var2=0;
    for (int j = 0; j< sample_size;j++)
    {
        var1 += std::pow(sample1[j]-sample1_mean,2);
        var2 += std::pow(sample2[j]-sample2_mean,2);
    }
    
    float stdv1 = std::sqrt(var1/sample_size);
    float stdv2 = std::sqrt(var2/sample_size);

    float  cov = 0; 
    for(int k = 0; k < sample_size ; k++)
    {
        cov += (sample1[k]-sample1_mean)*(sample2[k]-sample2_mean);
    }

    cov /= sample_size;

    float coe = cov/(stdv1*stdv2);
    std::cout<<"coe :"<<coe<<"\n";
    return 0;
}

int main()
{
    std::vector<float> var_x_1 = {0 , 1 , 2 , 3 , 4  ,5 , 6 , 7 , 8 , 9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17, 18 ,19,20, 21 ,22, 23,
                                  24 ,25 ,26, 27 ,28 ,29 ,30, 31, 32 ,33, 34, 35 ,36, 37, 38 ,39 ,40, 41, 42 ,43, 44, 45 ,46 ,47,
                                  48, 49};
    std::vector<float> var_y_1 = {2.7, 4.7, 6.7, 8.7, 10.7, 12.7, 14.7, 16.7, 18.7, 20.7, 22.7, 24.7, 26.7, 28.7, 30.7, 32.7,
                                34.7, 36.7, 38.7, 40.7, 42.7, 44.7, 46.7, 48.7, 50.7, 52.7, 54.7, 56.7, 58.7, 60.7, 62.7, 64.7,
                                 66.7, 68.7, 70.7, 72.7, 74.7, 76.7, 78.7, 80.7, 82.7, 84.7, 86.7, 88.7, 90.7, 92.7, 94.7, 96.7, 98.7, 100.7};


    
    std::vector<float> var_x_2 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
    std::vector<float> var_y_2 = {17.3, 19.3, 21.3, 23.3, 25.3, 27.3, 29.3, 31.3, 33.3, 35.3, 37.3, 39.3, 41.3, 43.3, 45.3, 47.3, 49.3, 51.3, 53.3, 55.3, 57.3, 59.3, 61.3, 63.3, 65.3, 67.3, 69.3, 71.3, 73.3, 75.3, -0.6669380616522619, -0.9873392775238264, -0.39998531498835127, 0.5551133015206257, 0.9998433086476912, 0.5253219888177297, -0.4321779448847783, -0.9923354691509287, -0.6401443394691997, 0.3005925437436371, 0.9649660284921133, 0.7421541968137826, -0.16299078079570548, -0.9182827862121189, -0.8293098328631502, 0.022126756261955736, 0.853220107722584, 0.8998668269691937, 0.11918013544881928, -0.7710802229758452};
    covariance_variable(var_x_1,var_x_2);
    covariance_variable(var_y_1,var_y_2);
    
    std::vector<float> var_a1={965,0};
    std::vector<float> var_a2={545,0};
    transfer_of_axes(var_a1,var_a2);
    
}

#include <WinSock2.h>
#include <Iphlpapi.h>
#include <iostream>
using namespace std;
#pragma comment(lib,"Iphlpapi.lib") 

int maina1(int argc, char* argv[])
{
   
    PIP_ADAPTER_INFO pIpAdapterInfo = new IP_ADAPTER_INFO();
  
    unsigned long stSize = sizeof(IP_ADAPTER_INFO);
  
    int nRel = GetAdaptersInfo(pIpAdapterInfo,&stSize);
 
    int netCardNum = 0;
    std::string mac_address = "";
    int IPnumPerNetCard = 0;
    if (ERROR_BUFFER_OVERFLOW == nRel)
    {
      
        delete pIpAdapterInfo;
      
        pIpAdapterInfo = (PIP_ADAPTER_INFO)new BYTE[stSize];
       
        nRel=GetAdaptersInfo(pIpAdapterInfo,&stSize);    
    }
    if (ERROR_SUCCESS == nRel)
    {
        bool find_ethernet_flag = false;
       
       
        while (pIpAdapterInfo)
        {
            switch(pIpAdapterInfo->Type)
            {
            case MIB_IF_TYPE_OTHER:
               
                break;
            case MIB_IF_TYPE_ETHERNET:
                find_ethernet_flag = true;
                break;
            case MIB_IF_TYPE_TOKENRING:
               
                break;
            case MIB_IF_TYPE_FDDI:
               
                break;
            case MIB_IF_TYPE_PPP:
              
                break;
            case MIB_IF_TYPE_LOOPBACK:
                
                break;
            case MIB_IF_TYPE_SLIP:
               
                break;
            default:

                break;
            }
             
            if(find_ethernet_flag)
            {
                for (DWORD i = 0; i < pIpAdapterInfo->AddressLength; i++)
                {
                    // if (i < pIpAdapterInfo->AddressLength-1)
                    // {
                    //     printf("%02X-",pIpAdapterInfo->Address[i]);
                    //     char ch[3];
                    //     memset(ch,'\0',3);
                    //     sprintf(ch," %02X-",pIpAdapterInfo->Address[i]);
                    //     mac_address = mac_address+std::string(ch);
                    // }
                    // else
                    // {
                    //     printf("%02X\n", i,pIpAdapterInfo->Address[i]);
                    // }
                
                    IP_ADDR_STRING *pIpAddrString =&(pIpAdapterInfo->IpAddressList);
                    do 
                    {
                       
                        pIpAddrString=pIpAddrString->Next;
                    } while (pIpAddrString);
                    
                    pIpAdapterInfo = pIpAdapterInfo->Next;
                    
                }
                 printf("---------------------------------------hello\n");
            }
           
            
        }
     
    }
    
    //释放内存空间
    if (pIpAdapterInfo)
    {
       
        delete pIpAdapterInfo;
    }
    
   
    return 0;
}


int check_usb_camera()
{
    int index = -1;
    for(int i = 0;i<3;i++)
    {
        cv::VideoCapture cap(i);
        if (!cap.isOpened()) {
            std::cout<<"camera "<<i<<" can't open the usb camera!"<<std::endl;
            cap.release();
            continue;
        }
        index = i;
        cap.release();
    }
    return index;
}

int statistic_score_bin(std::vector<float>& score,std::vector<float>& statistic_bin)
{
     if(score.size() ==0 )
    {
        return 0;
    }
    
    std::sort(score.begin(), score.end());
    std::vector<std::vector<float>> bins;
    int pos = -1;

    statistic_bin.push_back(score[0]);
    float sum = 0;
    for(int v = 0;v<score.size();v++)
    {
        sum += score[v];
        std::cout<<"Current score @:"<<score[v]<<"\n";
    }
    float average = sum / score.size();
    statistic_bin.push_back(average);
    statistic_bin.push_back(score[score.size()-1]);
    return 0;
}


int mainad(int argc,char* argv[])
{
    cudaSetDevice(0);
    const std::string  engine_file_path =  "./engine/yolov8s-pose.engine";
    std::cout<<"Current version V07.25.1.1.1"<<"\n";
    std::shared_ptr<ActionRecognition> ar_ptr = std::make_shared<ActionRecognition>();/*  */
    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    cv::Mat  src_img;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.45f;
    std::vector<Object> objs;
    cv::namedWindow(WINDOW_NAME_VIDEO, cv::WINDOW_AUTOSIZE);
    cvui::init(WINDOW_NAME_VIDEO);
    std::vector<cv::Point> calibrate_region={{400,100},{1520,100},{1520,980},{400,980}};
    ar_ptr->Init(calibrate_region);
    int u_id = check_usb_camera();
    if(u_id == -1)
    {
        std::cout<<"Current return :"<<u_id<<"\n";
        system("pause");
        return 0;
    }

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);

    if (!cap.isOpened()) {
        std::cout<<"Can't open the usb camera!"<<std::endl;
        system("pause");
        return -1;
    }
    
    cv::Mat res_mat;
    while (cap.read(src_img)) {
        if(src_img.empty())
        {
            std::cout<<"Read image failed!"<<"\n";
            continue;
        }

        cv::Mat infer_mat = src_img.clone();
        data_package package;
        package.action_type_text="";
        package.action_score = 0;
        package.action_times = 0;
        package.action_up_score = 0;
        package.action_pull_score = 0;
        package.shooting_score =0;

        objs.clear();
        yolov8_pose->copy_from_Mat(infer_mat, size);
        auto start = std::chrono::system_clock::now();
        yolov8_pose->infer();
        auto end = std::chrono::system_clock::now();
        yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
        yolov8_pose->draw_objects(src_img, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
        ar_ptr->AnalysisInterface(objs,src_img,package);
       
        DrawRegion(src_img,calibrate_region);
        // std::cout<<"Image size : ["<<src_img.cols<<","<<src_img.rows<<"]\n";
        // cv::imshow(WINDOW_NAME_VIDEO, src_img);
        if(src_img.empty())
        {
            continue;
        }
        cv::resize(src_img,res_mat,cv::Size(1080,720));
        cvui::imshow(WINDOW_NAME_VIDEO, res_mat);
        cvui::update();
        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    cap.release();
    return 0;
}


int main_nb(int argc,char* argv[])
{
    cudaSetDevice(0);
    const std::string  engine_file_path =  "./engine/yolov8s-pose.engine";
    // std::string video_path = "E:/action_recgnition/build/videos/nstand/t1.mp4";
    // std::string video_path = std::string(argv[1]);
    // const std::string  engine_file_path =  "E:/action_recgnition/3rdparty/TensorRT-8.2.1.8/bin/yolov8s-pose.engine";
    std::cout<<"Current version V07.25.1.1.1"<<"\n";
    std::shared_ptr<ActionRecognition> ar_ptr = std::make_shared<ActionRecognition>();
    std::vector<cv::String> imageFiles;
    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    cv::Mat  src_img;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.45f;
    std::vector<Object> objs;
    cv::namedWindow(WINDOW_NAME_VIDEO);
    cvui::init(WINDOW_NAME_VIDEO);

    bool calibrate_flag = false;
    bool start_run_flag = false;
    std::vector<cv::Point> calibrate_region;


    int u_id = check_usb_camera();
    if(u_id == -1)
    {
        std::cout<<"Current return :"<<u_id<<"\n";
        system("pause");
        return 0;
    }

    cv::VideoCapture cap(u_id);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);

    if (!cap.isOpened()) {
        std::cout<<"Can't open the usb camera!"<<std::endl;
        system("pause");
        return -1;
    }
    

    while (cap.read(src_img)) {
        if(src_img.empty())
        {
            std::cout<<"Read image failed!"<<"\n";
            continue;
        }

        cv::Mat infer_mat = src_img.clone();
        data_package package;
        package.action_type_text="";
        package.action_score = 0;
        package.action_times = 0;
        package.action_up_score = 0;
        package.action_pull_score = 0;
        package.shooting_score =0;

        if (cvui::button(src_img, 10, 10, "calibrate region")) {
            calibrate_flag = true;
            start_run_flag =false;
            calibrate_region.clear();
        }

        if(calibrate_flag)
        {
            if(cvui::mouse(cvui::LEFT_BUTTON,cvui::DOWN))
            {
                cv::Point cursor = cvui::mouse();
                calibrate_region.push_back(cursor);
                if(calibrate_region.size() == 4)
                {
                    calibrate_flag = false;
                    start_run_flag = true ;
                    ar_ptr->Init(calibrate_region);/*  */
                }
            }

            if(cvui::mouse(cvui::RIGHT_BUTTON,cvui::DOWN))
            {
                if(calibrate_region.size()>0)
                    calibrate_region.pop_back(); 
            }
        }
 
        if(start_run_flag )
        {
            objs.clear();
            yolov8_pose->copy_from_Mat(infer_mat, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            yolov8_pose->draw_objects(src_img, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            ar_ptr->AnalysisInterface(objs,src_img,package);
        }

        if(calibrate_region.size() == 4)
        {
            DrawRegion(src_img,calibrate_region);
        }

        DrawPoint(src_img,calibrate_region);
        cvui::imshow(WINDOW_NAME_VIDEO, src_img);
        cvui::update();
        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    cap.release();
    return 0;
}

#if 0
int mainui(int argc,char* argv)
{
    cudaSetDevice(0);
    const std::string  engine_file_path =  "./engine/yolov8s-pose.engine";
    // const std::string  engine_file_path =  "E:/action_recgnition/3rdparty/TensorRT-8.2.1.8/bin/yolov8s-pose.engine";

    std::shared_ptr<ActionRecognition> ar_ptr = std::make_shared<ActionRecognition>();
    std::vector<cv::String> imageFiles;
    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    cv::Mat  res, src_img;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.45f;
    std::vector<Object> objs;
    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);

    bool calibrate_flag = false;
    bool start_run_flag = false;
    std::vector<cv::Point> calibrate_region;

    cv::Mat show_ui_temp = cv::Mat(720, 1080, CV_8UC3);
    show_ui_temp = cv::Scalar(35, 33, 31);
    std::cout<<"Current version :V23.07.1.1.0"<<"\n";
    std::string infor_text = "information bar";
    int u_id = check_usb_camera();
    if(u_id == -1)
    {
        std::cout<<"Current return :"<<u_id<<"\n";
        system("pause");
        return 0;
    }

    cv::VideoCapture cap(u_id);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);

    if (!cap.isOpened()) {
        std::cout<<RED<<"Can't open the usb camera!"<<RESET<<std::endl;
        return -1;
    }
    
    std::vector<float> statistics_score;
    
    std::vector<cv::Mat> ui_res_mat;
    cv::Mat statistic_mat = cv::Mat(cv::Size(340,220),CV_8UC3);
    statistic_mat = cv::Scalar(255,255,255);

    int athlete_num = 0;
    float athlete_score_up =0.;
    float athlete_score_pull =0.;

    while (cap.read(src_img)) {
        data_package package;
        package.action_type_text="";
        package.action_score = 0;
        package.action_times = 0;
        package.action_up_score = 0;
        package.action_pull_score = 0;
        package.shooting_score =0;
        
        cv::Mat ui_mat = show_ui_temp.clone();
        cv::Mat ui_realtime_mat;
        cv::Mat imfer_mat = src_img.clone();
         
        cv::resize(src_img,ui_realtime_mat,cv::Size(720,405));
        float ratio = ui_realtime_mat.cols* 1.0 / src_img.cols;
        cvui::rect(ui_mat,  text_show_x, 50, (ui_realtime_mat.cols-10)/2, 240, 0x00ff00);
        cvui::rect(ui_mat,  text_show_x+ui_realtime_mat.cols/2, 50, (ui_realtime_mat.cols+3)/2, 240, 0x00ff00);
        cvui::rect(ui_mat,  image_show_x-2, image_show_y-2, ui_realtime_mat.cols+3, ui_realtime_mat.rows+3, 0x00ff00);
        cvui::image(ui_mat, image_show_x, image_show_y, ui_realtime_mat);

        if (cvui::button(ui_mat, 10, 10, "calibrate region")) {
            calibrate_flag = true;
            calibrate_region.clear();
        }

        if (cvui::button(ui_mat, 10, 55, "     start      ")) {
            infor_text = "usb camera has open!";
            start_run_flag = true;

        }

        if (cvui::button(ui_mat, 10, 100,"     stop      ")) {
            infor_text = "close comera";
            if( start_run_flag == true)
                cv::destroyWindow(WINDOW_NAME_VIDEO);
            
            start_run_flag = false;

        }

        if (cvui::button(ui_mat, 10, 145,"     reset     ")) {

            infor_text = "reset";
            calibrate_region.clear();
            ui_res_mat.clear();
            if( start_run_flag == true)
                cv::destroyWindow(WINDOW_NAME_VIDEO);
                
            start_run_flag = false;
            calibrate_flag =false;
        }

        if(cvui::button(ui_mat, 10, 190,"    analysis    "))
        {
            std::vector<float> statistic_bin_up,statistic_bin_pull;
            statistic_score_bin(statistics_score,statistic_bin_up);
            statistic_score_bin(statistics_score,statistic_bin_pull);

            
           
            for (int i = 0; i < statistic_bin_up.size() ;i++)
            {
                int value = int(statistic_bin_up[i]);
                cv::rectangle(statistic_mat, cv::Rect(10+i * 30, 120+(100-value),20,100), Scalar(255,0,0), -1);
            }

            for (int i = 0; i < statistic_bin_pull.size() ;i++)
            {
                int value = int(statistic_bin_pull[i]);
                cv::rectangle(statistic_mat, cv::Rect(120+i * 30, 120+(100-value),20,100), Scalar(0,255,255), -1);
            }

        }

        if (cvui::button(ui_mat, 10, 235, "      exit      ")) {
            infor_text = "exit";
            calibrate_region.clear();
            ui_res_mat.clear();
            if( start_run_flag == true)
                cv::destroyAllWindows();
                
            start_run_flag = false;
            calibrate_flag =false;
            return 0;
        }

        if(calibrate_flag)
        {
            if(cvui::mouse(cvui::LEFT_BUTTON,cvui::DOWN))
            {
                cv::Point cursor = cvui::mouse();
                if(cursor.x>=image_show_x && cursor.y>= image_show_y)
                {
                    calibrate_region.push_back(cursor);
                }
                
                if(calibrate_region.size() == 4)
                {
                    calibrate_flag = false;
                    int poly_size = calibrate_region.size();
                    std::vector<cv::Point> vec_regs;
                    for(int i =0;i < poly_size ; i++)
                    {
                        cv::Point pt = calibrate_region[i];
                        pt.x = (pt.x-image_show_x) / ratio;
                        pt.y = (pt.y-image_show_y) / ratio;
                        vec_regs.push_back(pt);
                    }
                    ar_ptr->Init(vec_regs);
                }
            }

            if(cvui::mouse(cvui::RIGHT_BUTTON,cvui::DOWN))
            {
                if(calibrate_region.size()>0)
                    calibrate_region.pop_back(); 
            }
        }
 
        if(start_run_flag)
        {
            if(calibrate_region.size()==4)
            {
                objs.clear();
                yolov8_pose->copy_from_Mat(imfer_mat, size);
                auto start = std::chrono::system_clock::now();
                yolov8_pose->infer();
                auto end = std::chrono::system_clock::now();
                yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
                yolov8_pose->draw_objects(src_img, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
                ar_ptr->AnalysisInterface(objs,res,package);
                cvui::imshow(WINDOW_NAME_VIDEO, res);
                // if(ui_res_mat.size()==2)
                // {
                    
                //     for(int i = 0 ;i < ui_res_mat.size();i++)
                //     {
                //         if(ui_res_mat[i].empty())
                //         {
                //             continue;
                //         } 
                //         cv::Mat img ;
                //         int step = 20;
                //         cv::resize(ui_res_mat[i].clone(),img,cv::Size(150,200));
                //         cvui::image(ui_mat, text_show_x+ui_realtime_mat.cols/2+i*150+(i+1)*step, 70, img);
                //     }
                // }  
            }
            else
            {
                cvui::printf(ui_mat, text_show_x+70, 10, 0.7, 0xff0000, " Please calibrate the detection area!");
            }
        }

        if(calibrate_region.size() == 4)
        {
            DrawRegion(ui_mat,calibrate_region);
            // DrawRegion(src_img,calibrate_region);
        }
        DrawPoint(ui_mat,calibrate_region);
        // DrawPoint(src_img,calibrate_region,ratio);
        cvui::image(ui_mat, text_show_x+ui_realtime_mat.cols/2+10, 60, statistic_mat);
        cvui::printf(ui_mat, text_show_x+5, 60, 0.7, 0xfffff, " action type: %s", package.action_type_text.c_str());
        cvui::printf(ui_mat, text_show_x+5, 100, 0.7, 0xfffff, " real-time score: %.3f", package.action_score);
        cvui::printf(ui_mat, text_show_x+5, 140,0.7, 0xfffff, " action times: %d", package.action_times);
        cvui::printf(ui_mat, text_show_x+5, 180,0.7, 0xff000, " action up: %.3f", package.action_up_score);
        cvui::printf(ui_mat, text_show_x+5, 220,0.7, 0xff000, " action pull: %.3f", package.action_pull_score);
        cvui::printf(ui_mat, text_show_x+5, 260,0.7, 0xff000, " shooting score: %d", package.shooting_score);
        
        if( package.action_up_score != athlete_score_up && package.action_up_score != 0)
        {
            statistics_score.push_back(package.action_up_score);
            athlete_score_up = package.action_up_score;
        }

        if( package.action_pull_score != athlete_score_pull  && package.action_pull_score != 0)
        {
            statistics_score.push_back(package.action_pull_score);
            athlete_score_pull = package.action_pull_score;
        }
        
        cvui::imshow(WINDOW_NAME, ui_mat);
        cvui::update();
        if (cv::waitKey(20) == 27) {
            break;
        }
    }
    cap.release();
    return 0;
}
#endif


int mainbv(int argc, char** argv)
{
    cudaSetDevice(0);
    
    std::string video_name = std::string(argv[1]);
    // const std::string engine_file_path="E:/action_recgnition/build/engine/yolov8s-pose.engine";
    const std::string  engine_file_path=  "E:/action_recgnition/3rdparty/TensorRT-8.2.1.8/bin/yolov8m-pose.engine";
    // const std::string path= "E:/action_recgnition/build123/videos/nstand/"+video_name;
    const std::string path= "C:/Users/binli/Desktop/action/stand/"+video_name;
     
    std::shared_ptr<ActionRecognition> ar_ptr = std::make_shared<ActionRecognition>();
    std::string imageFolder = "E:/action_recgnition/build123/test/draw_bow/*.jpg";

    std::vector<cv::String> imageFiles;
    cv::glob(imageFolder, imageFiles);
    std::vector<cv::Point> calibrate_region={{400,100},{1520,100},{1520,980},{400,980}};
    ar_ptr->Init(calibrate_region);

    bool                     isVideo{true};
    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);
    

    cv::Mat  res, image;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.45f;

    std::vector<Object> objs;
     
    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
   
    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
             
            return -1;
        }
        std::cout<<"infer :"<<"\n";
    //  return 0;
        cv::VideoWriter writer;    
        int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        std::string filename = video_name;
        writer.open(filename, codec, 30, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)), true);

        while (cap.read(image)) {
            objs.clear();
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            
            data_package package;

            ar_ptr->AnalysisInterface(objs,image,package);
            yolov8_pose->draw_objects(image, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
             DrawRegion(image,calibrate_region);
            cv::imshow("result", image);
            // cv::waitKey(0);
            writer.write(image);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
        writer.release();
    }
    else {
        for (auto& path : imageFiles) {
            objs.clear();
            image = cv::imread(path);
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
             data_package package;
            ar_ptr->AnalysisInterface(objs,image,package);
            yolov8_pose->draw_objects(image, objs, SKELETON, KPS_COLORS, LIMB_COLORS);

            DrawRegion(res,calibrate_region);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            
            // cv::imwrite("debug_mat.jpg",res);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
            
        }
    }
    cv::destroyAllWindows();
    delete yolov8_pose;
    return 0;
}
