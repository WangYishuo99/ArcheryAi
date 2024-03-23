#include "../include/action_recognition.hpp"
#include <cstdlib>

#define PI 3.1415926
#define STRAIGHT_ANGLE 180.0
#define IMAGE_WIDTH  1920.0
#define IMAGE_HEIGHT 1080.0

#define INFER_SHOW_X 20
#define INFER_SHOW_Y 900

std::string file = "up.txt";
//constant variable
#define  DEBUG 1
#define  LOST_FRAME_NUM (20)

//--------------params------------
//Up bow,angle threadhold between arm and body
float upbow_left_bodyside  = 70.0; //73
float upbow_right_bodyside = 70.0; //80.0 

//Up bow,angle threadhold between up arm and forearm
float upbow_left_arm_bend  = 113.0;
float upbow_right_arm_bend = 101.0;

//Draw bow,angle threadhold between up arm and forearm
// float drawbow_right_arm_bend  = 76.5;//62
float drawbow_right_arm_bend  = 86.5;//62


//arm up 
float stand_arm_v = 25.0;

//Face deflection angle
float turn_to_v_var = 18.5;
float turn_to_h_var = 17.0;

//threshold
int upbow_times_threshold = 10;
int draw_times_threshold = 10;
int prepare_times_threshold = 15;

//features
// upbow_feat_k1 = {0.5057494444444445, 0.005904054642857142, 0.011717648648648648, 0.046320757446808504, 0.00900112862745098, 0.8076376999999999, 0.720570314814815, 0.5971666730769231, 0.8304589130434784};
// upbow_feat_k2 = {0.29348446575342463, 0.004267899651162792, 0.0043770184375000005, 0.011813088666666666, 0.02259189090909091, 0.9111647777777778, 0.9725297142857143, 0.5451845, 0.661158387755102};
// upbow_feat_k3 = {0.1584628372093023, 0.007040554230769231, 0.00855361057142857, 0.030204487096774188, 0.040895282191780825, 0.9614146440677966, 0.8905640196078431, 0.46943692000000004, 0.5264136438356164};

// drawbow_feat_k1 = {0.08172434509803923, 0.0044464213513513515, 0.0036136609999999998, 0.02876341538461538, 0.0405764164556962, 0.9748597701149426, 0.1627029387755102, 0.55103, 0.7288097349397591};
// drawbow_feat_k2 = {0.1335369739130435, 0.003493431791044776, 0.004651424909090909, 0.015693721739130435, 0.029598788372093027, 0.946101492063492, 0.19736439080459772, 0.5256026391752577, 0.6656686222222221};
// drawbow_feat_k3 = {0.06489121587301588, 0.0065399353600000005, 0.009124384677419354, 0.006290985279661018, 0.017846562616822426, 0.9896793797468355, 0.26139884946236563, 0.5837069066666667, 0.8154174653465348};

std::vector<float> upbow_feat_k1 = {0.5057494444444445, 0.007040554230769231, 0.012234692592592591, 0.007400649565217391, 0.05154428235294117, 0.8076376999999999, 0.809196, 0.4664549347826087, 0.6511572765957447};
std::vector<float> upbow_feat_k2 = {0.29348446575342463, 0.005904054642857142, 0.008946503777777778, 0.04376452950819672, 0.03526578108108109, 0.9614146440677966, 0.693330611111111, 0.53330352, 0.8236050800000001};
std::vector<float> upbow_feat_k3 = {0.1584628372093023, 0.004267899651162792, 0.0043770184375000005, 0.01912338524590164, 0.012370594285714286, 0.9111647777777778, 0.9474475999999998, 0.5886748194444444, 0.5245270563380281};

std::vector<float> drawbow_feat_k1 = {0.08172434509803923, 0.0044464213513513515, 0.0036136609999999998, 0.016183441304347825, 0.028117192857142852, 0.946101492063492, 0.3099756315789474, 0.5505922542372882, 0.6656686222222221};
std::vector<float> drawbow_feat_k2 = {0.06489121587301588, 0.003493431791044776, 0.004651424909090909, 0.028966998412698407, 0.0405764164556962, 0.9896793797468355, 0.24505983720930227, 0.5837069066666667, 0.7288097349397591};
std::vector<float> drawbow_feat_k3 = {0.1335369739130435, 0.0065399353600000005, 0.009124384677419354, 0.006370918858333334, 0.01710390744680851, 0.9748597701149426, 0.1813593225806452, 0.5253391894736843, 0.8154174653465348};


ActionRecognition::ActionRecognition()
{
    action_counter = 0;
}

ActionRecognition::~ActionRecognition()
{
    // outputFile.close();
    
}

int ActionRecognition::Init(std::vector<cv::Point> &roi_polygon)
{
    roi_polygon_vec.clear();
    lost_frame_counter=0;
    action_duration_record =std::make_pair(0,0);
    num_in_cycle = 0;
    last_action_type = PREPARE;
    score_upbow_history = 0.6;
    score_drawbow_history = 0.6;
    condition_num = 0;
    action_counter = 0;
    up_score_history.clear();
    pull_score_history.clear();

    up_score = 0;
    pull_score = 0;
    s_score = 0;

    score_record=0;

    if(roi_polygon.size() == 4)
    {
        for(int i = 0;i<roi_polygon.size();i++)
        {
             roi_polygon_vec.push_back(roi_polygon[i]);
        }
       
        roi_rects_vec = cv::boundingRect(roi_polygon);
    }
    // outputFile.open(file);

    return 0;
}

float ActionRecognition::euc_distance(std::vector<float>& _f1,std::vector<float>& _f2)
{
    assert(_f1.size() == _f1.size());
	float c = 0;
	float temp = 0.0;
    float ptemp1[256] = {0.0};
	float ptemp2[256] = {0.0};
    int feat_size = _f1.size();
    float norm1 =  0.0;
	float norm2 =  0.0;
    for (auto i = 0; i < feat_size; i++) 
	{
		norm1 += _f1[i]  *  _f1[i];
		norm2 += _f2[i]  *  _f2[i];
	}
	norm1  = sqrt(norm1);
	norm2  = sqrt(norm2);

    for (auto i = 0; i < feat_size; i++) 
	{
		ptemp1[i]= _f1[i] / norm1;
		ptemp2[i]= _f2[i] / norm2;
	} 

    for (auto i = 0; i < feat_size; i++) 
	{
		temp += (ptemp1[i] - ptemp2[i]) * (ptemp1[i] - ptemp2[i]);
	}
    c = sqrt(temp);
    
	return ( 1/(1 + c) ); 
}

inline float find_best_score(std::vector<float>& score)
{
    if(score.size() ==0 )
    {
        return 0;
    }
    
    std::sort(score.begin(), score.end());
    std::vector<std::vector<float>> bins;
    int pos = -1;
    //0-10 10-20 90-100
    for(int i =0 ;i < 10 ;i++)
    {
        std::vector<float> bin;
        for(int v = pos+1;v<score.size();v++)
        {
            if(score[v]<=(i+1)*10 && score[v] > i*10)
            {
                bin.push_back(score[v]);
                pos = v;
                continue;
            }
            break;
        }
        bins.push_back(bin);
    }
    
    int max_elem = 0;
    int best_idx = -1;
    for(int i = 0 ;i<bins.size();i++)
    {
        if(bins[i].size()>max_elem)
        {
            best_idx = i;
            max_elem = bins[i].size();
        }
    }
    
    float bin_score_sum = 0.0;
    if (best_idx != -1)
    {
        for(int i = 0 ;i< max_elem;i++)
        {
            bin_score_sum+= bins[best_idx][i];
        }
    }
    bin_score_sum /= max_elem;
    return bin_score_sum;
    
}
inline float getMold(const std::vector<float>& vec)
{
	int n = vec.size();
	float sum = 0.0;
	for (int i = 0; i < n; ++i)
		sum += vec[i] * vec[i];
	return sqrt(sum);
}
float ActionRecognition::cos_distance(const std::vector<float>& vec1, const std::vector<float>& vec2)
{
	int n = vec1.size();
	float tmp = 0.0;
	for (int i = 0; i < n; ++i)
		tmp += vec1[i] * vec2[i];
	float simility =  tmp / (getMold(vec1)*getMold(vec2));
	return simility;
}
 

double ActionRecognition::Angle(MathVector<float>& vec1,MathVector<float>& vec2)
{
    float norm_vec1=0.,norm_vec2=0.;
    double cosx=0.,arccosx=0.,ang=0.;
    norm_vec1=vec1.norm();
    norm_vec2=vec2.norm();
    if(fabs(norm_vec1-norm_vec2)<1e-6)
    {
        norm_vec1=norm_vec1;
    }
    cosx=vec1.dot(vec2)/(norm_vec1*norm_vec2);
    if(cosx > 1.0)
    {
        cosx = 1.0;
    }
    else if(cosx<-1.0)
    {
        cosx = -1.0;
    }
    
    arccosx=acosf(cosx);
    ang=180.-arccosx*180./PI;
    return ang;
}

double ActionRecognition::arm_bodyside_angle(MathVector<float>& vec1,MathVector<float>& vec2)
{
    double ang=180.0 -Angle(vec1,vec2);
    return ang;
}

int ActionRecognition::filter_other(std::vector<Object>& objs)
{
    float prob = 0.65;
    int best_match = -1;
    for(int i = 0;i < objs.size(); i++)
    {
        if (objs[i].prob > prob)
        {
            best_match = i;
            prob = objs[i].prob;
        }
    }
    if(best_match == -1)
    {
        return -1;
    }
    Object tmp_obj = objs[best_match];
    objs.clear();
    objs.push_back(tmp_obj);
    return 0;
}

double ActionRecognition::vetorial_angle(MathVector<float>& vec1,MathVector<float>& vec2)
{
    double ang=Angle(vec1,vec2);
    return ang;
}

float ActionRecognition::points_distance(Point<float> p1, Point<float> p2)
{
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

int ActionRecognition::arm_location_relative(Object& keypoint)
{
    Point<float> p6,p8,p10,p5,p7,p9;
    p6.x = keypoint.kps[6 * 3];
    p6.y = keypoint.kps[6 * 3 + 1];

    p8.x = keypoint.kps[8 * 3];
    p8.y = keypoint.kps[8 * 3 +1];

    p10.x = keypoint.kps[10 * 3];
    p10.y = keypoint.kps[10 * 3 +1];

    p5.x = keypoint.kps[5 * 3];
    p5.y = keypoint.kps[5 * 3 + 1];

    p7.x = keypoint.kps[7 * 3];
    p7.y = keypoint.kps[7 * 3 +1];

    p9.x = keypoint.kps[9 * 3];
    p9.y = keypoint.kps[9 * 3 + 1];

    float right_center_x = (p6.x+p8.x+p10.x)/3;
    float right_center_y = (p6.y+p8.y+p10.y)/3;

    float left_center_x = (p5.x+p7.x+p9.x)/3;
    float left_center_y = (p5.y+p7.y+p9.y)/3;

    bool left_arm_direction_h =  p9.x > p7.x;
    bool right_arm_direction_h = p10.x > p8.x;
    bool arm_direction = left_arm_direction_h &  right_arm_direction_h;
    return right_center_x < left_center_x && right_center_y < left_center_y && arm_direction;
}
int ActionRecognition::arm_location(Object& keypoint)
{  
    Point<float> p6,p8,p10,p5,p7,p9;
    p6.x = keypoint.kps[6 * 3];
    p6.y = keypoint.kps[6 * 3 + 1];

    p8.x = keypoint.kps[8 * 3];
    p8.y = keypoint.kps[8 * 3 +1];

    p10.x = keypoint.kps[10 * 3];
    p10.y = keypoint.kps[10 * 3 +1];

    p5.x = keypoint.kps[5 * 3];
    p5.y = keypoint.kps[5 * 3 + 1];

    p7.x = keypoint.kps[7 * 3];
    p7.y = keypoint.kps[7 * 3 +1];

    p9.x = keypoint.kps[9 * 3];
    p9.y = keypoint.kps[9 * 3 + 1];

     //right arm var 
    float right_arm_v_mean = (p6.y+p8.y+p10.y)/3;
    float right_arm_h_mean = (p6.x+p6.x+p10.x)/3;
    // std::cout<<"Right Arm h mean:"<<right_arm_h_mean<<",v mean :"<<right_arm_v_mean<<"\n";

    float right_var_arm_h = (std::pow((p6.x-right_arm_h_mean),2)+std::pow((p8.x-right_arm_h_mean),2)+std::pow((p10.x-right_arm_h_mean),2))/3;
    float right_var_arm_v = (std::pow((p6.y-right_arm_v_mean),2)+std::pow((p8.y-right_arm_v_mean),2)+std::pow((p10.y-right_arm_v_mean),2))/3;
    
    // std::cout<<"Right Arm h var:"<<right_var_arm_h<<",v var :"<<right_var_arm_v<<"\n";

    float right_stand_dev_h = std::sqrt(right_var_arm_h);
    float right_stand_dev_v = std::sqrt(right_var_arm_v);
    // std::cout<<"Right Arm h stand dev:"<<right_stand_dev_h<<",v stand dev :"<<right_stand_dev_v<<"\n";
    
    //left
    float left_arm_v_mean = (p5.y+p7.y+p9.y)/3;
    float left_arm_h_mean = (p5.x+p7.x+p9.x)/3;
    // std::cout<<"Left Arm h mean:"<<left_arm_h_mean<<",v mean :"<<left_arm_v_mean<<"\n";

    float left_var_arm_h = (std::pow((p5.x-left_arm_h_mean),2)+std::pow((p7.x-left_arm_h_mean),2)+std::pow((p9.x-left_arm_h_mean),2))/3;
    float left_var_arm_v = (std::pow((p5.y-left_arm_v_mean),2)+std::pow((p7.y-left_arm_v_mean),2)+std::pow((p9.y-left_arm_v_mean),2))/3;
    
    // std::cout<<"Left Arm h var:"<<left_var_arm_h<<",v var :"<<left_var_arm_v<<"\n";

    float left_stand_dev_h = std::sqrt(left_var_arm_h);
    float left_stand_dev_v = std::sqrt(left_var_arm_v);
    // std::cout<<"Left Arm h stand dev:"<<left_stand_dev_h<<",v stand dev :"<<left_stand_dev_v<<"\n";

   
    bool left_arm_direction_h =  p9.x > p7.x;
    bool right_arm_direction_h = p10.x > p8.x;
    bool arm_direction = left_arm_direction_h &  right_arm_direction_h;
    int ret = arm_direction && (fabs(right_stand_dev_v-right_stand_dev_v) < stand_arm_v) && (right_stand_dev_v < right_stand_dev_h) && left_stand_dev_v < left_stand_dev_h;
    
    left_stand_dev_v /= IMAGE_HEIGHT;
    right_stand_dev_v /= IMAGE_HEIGHT;
    features.push_back(left_stand_dev_v);
    features.push_back(right_stand_dev_v);

    // outputFile<<left_stand_dev_v<<"\t"<<right_stand_dev_v<<"\t";

    // std::cout<<GREEN<<"4-5 arm standard : h:"<<left_stand_dev_v<<", v:"<<right_stand_dev_v<<RESET<<"\n";
    return ret ;
}

int ActionRecognition::head_location(Object& obj,cv::Mat& src_img)
{
    Point<float> p0,p1,p2,p3,p4,points_mean;
    p0.x = obj.kps[0 * 3];
    p0.y = obj.kps[0 * 3 + 1];

    p1.x = obj.kps[1 * 3];
    p1.y = obj.kps[1 * 3 +1];

    p2.x = obj.kps[2 * 3];
    p2.y = obj.kps[2 * 3 +1];

    p3.x = obj.kps[3 * 3];
    p3.y = obj.kps[3 * 3 + 1];

    p4.x = obj.kps[4 * 3];
    p4.y = obj.kps[4 * 3 + 1];
    
    points_mean.x = (p0.x + p1.x + p2.x )/3;
    points_mean.y = (p0.y + p1.y + p2.y )/3;
    cv::circle(src_img, cv::Point(points_mean.x,points_mean.y), 7, cv::Scalar(255,0,0), -1);

    float left_dist = points_distance(points_mean,p3);
    float right_dist = points_distance(points_mean,p4);
   
    if( right_dist > 2*left_dist)
    {
        float var_x = 0.25*(pow((p0.x - points_mean.x),2) + pow((p1.x - points_mean.x),2) + pow((p2.x - points_mean.x),2) + pow((p3.x - points_mean.x),2)  );
        float var_y = 0.25*(pow((p0.y - points_mean.y),2) + pow((p1.y - points_mean.y),2) + pow((p2.y - points_mean.y),2) + pow((p3.y - points_mean.y),2)  );
        var_x = std::sqrtf(var_x);
        var_y = std::sqrtf(var_y);
        var_x /= IMAGE_WIDTH;
        var_y /= IMAGE_HEIGHT;

        // outputFile<<var_x<<"\t"<<var_y<<"\t";
        features.push_back(var_x);
        features.push_back(var_y);
        // std::cout<<GREEN<<"2 3  head standard : h :"<<var_x<<", v :"<< var_y<<RESET<<"\n";

        if(var_x <= turn_to_h_var && var_y <= turn_to_h_var)
        {
            return 1; 
        }
    }
    else if ( left_dist > 2*right_dist)
    {
        float var_x = 0.25*(pow((p0.x - points_mean.x),2) + pow((p1.x - points_mean.x),2) + pow((p2.x - points_mean.x),2) + pow((p4.x - points_mean.x),2));
        float var_y = 0.25*(pow((p0.y - points_mean.y),2) + pow((p1.y - points_mean.y),2) + pow((p2.y - points_mean.y),2) + pow((p4.y - points_mean.y),2));
        var_x = std::sqrtf(var_x);
        var_y = std::sqrtf(var_y);
        if(var_x <= turn_to_h_var && var_y <= turn_to_h_var)
        {
            return 2;
        }
    }
    else 
    {
        return 0;
    }
    return 0;
}

int ActionRecognition::hand_location(Object& obj,cv::Mat& src_img)
{
    Point<float> p0,p1,p2,p6,p5,p10,points_mean;
    Point<float> vec ;
    
    p0.x = obj.kps[0 * 3];
    p0.y = obj.kps[0 * 3 + 1];

    p1.x = obj.kps[1 * 3];
    p1.y = obj.kps[1 * 3 +1];

    p2.x = obj.kps[2 * 3];
    p2.y = obj.kps[2 * 3 +1];

    p5.x = obj.kps[5 * 3];
    p5.y = obj.kps[5 * 3 +1];

    p6.x = obj.kps[6 * 3];
    p6.y = obj.kps[6 * 3 +1];

    p10.x = obj.kps[10 * 3];
    p10.y = obj.kps[10 * 3 +1];

    points_mean.x = (p0.x + p1.x + p2.x )/3;
    points_mean.y = (p0.y + p1.y + p2.y )/3;

    MathVector<float> vec_p10_p =  MathVector<float>(p10,points_mean);
    MathVector<float> vec_p10_p5 = MathVector<float>(p10,p5);
    MathVector<float> vec_p10_p6 = MathVector<float>(p10,p6);

    vec.x = vec_p10_p.vx + vec_p10_p5.vx + vec_p10_p6.vx;
    vec.y = vec_p10_p.vy + vec_p10_p5.vy + vec_p10_p6.vy;
    
    float norm = sqrt(pow(vec.x,2)+pow(vec.y,2))/IMAGE_WIDTH;
    features.push_back(norm);
    // outputFile<<norm<<"\t";
    // std::cout<<GREEN<<"1  p10-p6-p5 : vec :["<<vec.x<<","<<vec.y<<"] ,value :"<<norm <<RESET<<"\n";
    return 0;
}

bool ActionRecognition::is_in_poly(cv::Rect& roi_rects, std::vector<cv::Point> &roi_polygon, cv::Point &po)
{
	/* 判断点是否落在外接矩形内，如果在矩形内才有继续判断的必要 */

    if((po.x < roi_rects.x) || (po.x > (roi_rects.x+roi_rects.width)) || \
        (po.y < roi_rects.y) || (po.y > (roi_rects.y+roi_rects.height))){
        return false;
    }
    // try
    // {
    //     if(cv::pointPolygonTest(roi_polygon, po, false) >= 0){
    //         return true;
    // }
    // }
    // catch(const std::exception& e)
    // {
    //     std::cerr << e.what() << "point ["<<po.x<<","<<po.y<<"]\n";
    // }
    
	return true;
}



// bool ActionRecognition::is_in_poly(cv::Rect& roi_rects, std::vector<cv::Point> &roi_polygon, cv::Point &po)
// {
// 	int nCross = 0;  
//     int nCount = roi_polygon.size();
// 	for (int i = 0; i < nCount; i++)   
// 	{  
// 		cv::Point p1 = roi_polygon[i];  
// 		cv::Point p2 = roi_polygon[(i + 1) % nCount]; 
 
// 		if ( p1.y == p2.y )  
// 			continue;  
// 		if ( po.y < std::min(p1.y, p2.y) )  
// 			continue;  
// 		if ( po.y >= std::max(p1.y, p2.y) )  
// 			continue;  
 
// 		double x = (double)(po.y - p1.y) * (double)(p2.x - p1.x) / (double)(p2.y - p1.y) + p1.x;  

// 		if ( x > po.x )  
// 		{  
// 			nCross++;  
// 		}  
// 	}  
    
// 	if ((nCross % 2) == 1)
// 	{
// 		return 0;
// 	}
// 	else
// 	{
// 		return 1;
// 	}
// }

 


ActiontYPE ActionRecognition::action_recognize_interface(Object& obj,cv::Mat& src_img)
{
    features.clear();
    ActiontYPE action_type = PREPARE;
    Point<float> p6,p8,p10;
    Point<float> p5,p7,p9;
    Point<float> p12,p11;
    
    p6.x = obj.kps[6 * 3];
    p6.y = obj.kps[6 * 3 + 1];

    p8.x = obj.kps[8 * 3];
    p8.y = obj.kps[8 * 3 +1];

    p10.x = obj.kps[10 * 3];
    p10.y = obj.kps[10 * 3 +1];

    p5.x = obj.kps[5 * 3];
    p5.y = obj.kps[5 * 3 + 1];

    p7.x = obj.kps[7 * 3];
    p7.y = obj.kps[7 * 3 +1];

    p9.x = obj.kps[9 * 3];
    p9.y = obj.kps[9 * 3 + 1];

    p11.x = obj.kps[11 * 3];
    p11.y = obj.kps[11 * 3 + 1];

    p12.x = obj.kps[12 * 3];
    p12.y = obj.kps[12 * 3 + 1];
     
    MathVector<float> left_upper_arm_vector =  MathVector<float>(p5,p7);
    MathVector<float> left_forearm_vector   =  MathVector<float>(p7,p9);

    MathVector<float> right_upper_arm_vector = MathVector<float>(p6,p8);
    MathVector<float> right_forearm_vector   = MathVector<float>(p8,p10);

    MathVector<float> left_side_body_vector  = MathVector<float>(p5,p11);
    MathVector<float> right_side_body_vector = MathVector<float>(p6,p12);
    //1 write p10-p6/p5/pq
    hand_location(obj,src_img);

    
    double angle_left_arm  =  vetorial_angle(left_upper_arm_vector,left_forearm_vector);
    double angle_right_arm =  vetorial_angle(right_upper_arm_vector,right_forearm_vector);

    
    double angle_left_side  =  arm_bodyside_angle(left_upper_arm_vector,left_side_body_vector);
    double angle_right_side =  arm_bodyside_angle(right_upper_arm_vector,right_side_body_vector);

    //2 write p0,p1,p2 horizontal and vertical standard deviation
    int head_status =  head_location(obj,src_img);
    
    int arm_status = arm_location(obj);
    int arm_relative = arm_location_relative(obj);
    arm_status = arm_status || arm_relative;
    

    //action1
    int up_bow_status = arm_status && (angle_left_side > upbow_left_bodyside)  && (angle_left_arm > upbow_left_arm_bend) && (angle_right_arm > upbow_right_arm_bend);
    if(up_bow_status)
    {
        action_type = UPBOW;
    }

    //action 2
    int draw_bow_status = arm_status && (angle_left_side > upbow_left_bodyside) && (angle_left_arm > upbow_left_arm_bend) && (angle_right_arm < drawbow_right_arm_bend);

    if(draw_bow_status)
    {
        action_type = DRAWBOW;
    }

    // std::cout<<"Current action :"<<action_type<<"\n";
    if(action_type == UPBOW)
    {
        condition_num = (int)(angle_left_side > upbow_left_bodyside) + (int)(angle_right_side > upbow_right_bodyside) + (int)(angle_left_arm > upbow_left_arm_bend) + (int)(angle_right_arm > upbow_right_arm_bend) \
                        + (int)(arm_status == 1)+ (int)(head_status == 1);
    }
    else if(action_type == DRAWBOW)
    {
        condition_num = (int)(arm_status == 1) + (int)(angle_left_side > upbow_left_bodyside) + (int)(angle_right_side > upbow_right_bodyside) + (int)(angle_left_arm > upbow_left_arm_bend)\
                        + (int)(angle_right_arm < drawbow_right_arm_bend)+(int)(head_status == 1);
    }
    else
    {
        condition_num = 0;
    }

    // std::cout<<"---------\n";
    // std::cout<<RED<<"angle_left_side > upbow_left_bodyside : "<<(angle_left_side > upbow_left_bodyside)<<
    //                " ,angle_left_arm > upbow_left_arm_bend : "<<(angle_left_arm > upbow_left_arm_bend)<<
    //                " ,angle_right_arm > upbow_right_arm_bend : "<<(angle_right_arm > upbow_right_arm_bend)<<
    //                " ,arm_location == 1 : "<<(arm_status == 1)<<RESET<<"\n";

    // std::cout<<"---------\n";

    // std::cout<<GREEN<<"-> left arm: "<<angle_left_arm<<" ,angle right: "<<angle_right_arm<<"\n"<<
    //                             "   left side: "<<angle_left_side<<",right side :"<<angle_right_side<<"\n"<<
    //                             "   arm location : "<<arm_status<<"\n"
    //                             <<RESET<<std::endl;

   
    //4 5 write Left and right arm bend;
    angle_left_arm /= STRAIGHT_ANGLE;
    angle_right_arm /= STRAIGHT_ANGLE;
    features.push_back(angle_left_arm);
    features.push_back(angle_right_arm);
    // outputFile<<angle_left_arm<<"\t"<<angle_right_arm<<"\t";
    // std::cout<<GREEN<<"6 7  left arm bend :"<<angle_left_arm<<", right arm bend :"<< angle_right_arm<<RESET<<"\n";
    
    //6 7 arm and body
    angle_left_side /= STRAIGHT_ANGLE;
    angle_right_side /= STRAIGHT_ANGLE;
    features.push_back(angle_left_side);
    features.push_back(angle_right_side);
    // outputFile<<angle_left_side<<"\t"<<angle_right_side<<"\n";
    // std::cout<<GREEN<<"8  9 left arm->body :"<<angle_left_side<<", right arm->body :"<< angle_right_side<<RESET<<"\n";


   
#if DEBUG1
    char text[32],ltext[32];
    char left_side[32],right_size[32];
    memset(text,'\0',32);
    memset(ltext,'\0',32);

    memset(left_side,'\0',32);
    memset(right_size,'\0',32);
    sprintf(left_side,"l %.2f",angle_left_side);
    sprintf(right_size,"r %.2f",angle_right_side);
    sprintf(text,"r %.2f",angle_right_arm);
    sprintf(ltext," l %.2f",angle_left_arm); 

    int font_face = cv::FONT_HERSHEY_COMPLEX; 
    double font_scale = 0.5;
    int thickness = 1;
    int baseline;
    //获取文本框的长宽
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
    cv::Point origin,lorigin; 
    origin.x = p8.x;
    origin.y =  p8.y;
    
    lorigin.x=p7.x;
    lorigin.y=p7.y;
    cv::putText(src_img, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 1, 0);
    cv::putText(src_img, ltext, lorigin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 1, 0);
    

    cv::putText(src_img, right_size, cv::Point(p6.x,p6.y), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 1, 0);
    cv::putText(src_img, left_side, cv::Point(p5.x,p5.y), font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 1, 0);

    cv::rectangle(src_img,obj.rect,cv::Scalar(0,255,0),1,1,0);
    cv::line(src_img,cv::Point(p6.x,p6.y),cv::Point(p8.x,p8.y),cv::Scalar(0,255,0));
    cv::line(src_img,cv::Point(p8.x,p8.y),cv::Point(p10.x,p10.y),cv::Scalar(0,255,0));
    cv::line(src_img,cv::Point(p5.x,p5.y),cv::Point(p7.x,p7.y),cv::Scalar(0,255,0));
    cv::line(src_img,cv::Point(p7.x,p7.y),cv::Point(p9.x,p9.y),cv::Scalar(0,255,0));
#endif
    return action_type;
}


float ActionRecognition::AnalysisInterface(std::vector<Object>& obj_keypoints,cv::Mat& src_img,data_package& package)
{   
    filter_other(obj_keypoints);
    condition_num = 0;
    // if (obj_keypoints.size()==0)
    // {
    //     lost_frame_counter++;
    //     if (lost_frame_counter>LOST_FRAME_NUM)
    //         Init(roi_polygon_vec); 
    // }
    
    for (auto &obj : obj_keypoints)
    {
        cv::Point pt ;
        pt.x = obj.rect.x+obj.rect.width/2;
        pt.y = obj.rect.y+obj.rect.height;
        
        bool incnt_01 = is_in_poly(roi_rects_vec,roi_polygon_vec,pt);

        pt.x = obj.rect.x+obj.rect.width/2;
        pt.y = obj.rect.y+obj.rect.height/2;
        bool incnt_02 = is_in_poly(roi_rects_vec,roi_polygon_vec,pt);

        if(!incnt_01 && !incnt_02)
        {
            continue;
        }

        ActiontYPE current_action =  action_recognize_interface(obj,src_img);   
        
        if((last_action_type == PREPARE && current_action == UPBOW) || (last_action_type == DRAWBOW && current_action == UPBOW))
        {
            action_duration_record.second++;
            // std::cout<<"From ready to up"<<action_duration_record.second<<"\n";
            if(action_duration_record.second - action_duration_record.first >= upbow_times_threshold)
            {
                action_duration_record.first = 0;
                action_duration_record.second = 0;
                if(last_action_type == DRAWBOW)
                {
                    if(num_in_cycle == 1)
                    {
                        num_in_cycle ++;
                    }
                    std::cout<<"action :from pull turn to up"<<"\n";
                    last_action_type = FREE;
                }else
                {
                     num_in_cycle =1;
                    std::cout<<"action :from ready turn to up"<<"\n";
                    last_action_type = UPBOW;
                }
               
            }
        }
        else if(last_action_type == UPBOW && current_action == DRAWBOW )
        {
            action_duration_record.second++;
            // std::cout<<"From draw to up"<<action_duration_record.second<<"\n";
            if(action_duration_record.second - action_duration_record.first >= draw_times_threshold)
            {
                action_duration_record.first = 0;
                action_duration_record.second = 0;
                last_action_type = DRAWBOW;
                std::cout<<"action :from up turn to pull"<<"\n";
            }
        }

        else if(current_action == PREPARE && last_action_type == FREE || current_action == PREPARE && last_action_type == UPBOW || condition_num == 0)
        {
            action_duration_record.second ++;
            if(action_duration_record.second - action_duration_record.first >= prepare_times_threshold)
            {
                action_duration_record.first = 0;
                action_duration_record.second = 0;
                last_action_type = PREPARE;
            }
        }
        else
        {
            action_duration_record.second--;
            if(action_duration_record.second <= 0)
            {
                action_duration_record.second = 0;
            }
        }
       
        float score = 0;
         
        if(last_action_type == UPBOW || last_action_type == FREE)
        {
            float sim1 = euc_distance(features,upbow_feat_k1);
            float sim2 = euc_distance(features,upbow_feat_k2);
            float sim3 = euc_distance(features,upbow_feat_k3);
           
            // std::cout<<BLUE<<"upbow euc distance :["<<sim1<<","<<sim2<<","<<sim3<<"]---> :"<<condition_num<<RESET<<"\n";
            
            // score = std::max(std::max(sim1,sim2),sim3)*condition_num/6.0;
            // score = (0.1*sim1+0.3*sim2+0.6*sim3)*condition_num/6.0;
            score = 0.3333*(sim1+sim2+sim3)*condition_num/6.0;
            score = score_upbow_history*0.1+0.9*score;
            score_upbow_history = score;
             
            if(last_action_type == UPBOW)
            {
                up_score_history.push_back(score*100);
            }
                
        }
        else if(last_action_type == DRAWBOW)
        {
            float sim1 = euc_distance(features,drawbow_feat_k1);
            float sim2 = euc_distance(features,drawbow_feat_k2);
            float sim3 = euc_distance(features,drawbow_feat_k3);
            // std::cout<<"draw euc distance :["<<sim1<<","<<sim2<<","<<sim3<<"]---> :"<<condition_num<<"\n";
            // score = std::max(std::max(sim1,sim2),sim3)*condition_num/6.0;
            // score = (0.1*sim1+0.3*sim2+0.6*sim3)*condition_num/6.0;
            score = 0.3333*(sim1+sim2+sim3)*condition_num/6.0;
            score = score_drawbow_history*0.1+0.9*score;
            score_drawbow_history = score;
            pull_score_history.push_back(score*100);
        }
        
        if(last_action_type == PREPARE)
        {
            score_drawbow_history =0.6;
            score_upbow_history =0.6;

            if(up_score_history.size() != 0)
            {
                up_score =    find_best_score(up_score_history);
                up_score_history.clear();
            }

            if(pull_score_history.size() != 0)
            {
                pull_score =  find_best_score(pull_score_history);
                pull_score_history.clear();
                s_score = (int)(pull_score/10);
            } 
        }
        else if (last_action_type != PREPARE)
        {
            up_score = 0.;
            pull_score = 0.;
        }
        

        if(num_in_cycle == 2)
        {
            num_in_cycle = 0;
            action_counter++;
            // last_action_type = PREPARE;
        }
        
        package.action_score = score;
        package.action_times = action_counter;
        package.action_up_score = up_score;
        package.action_pull_score = pull_score;
        package.shooting_score = s_score;
        score_record = score;

        // char action_status_text[32];
        // memset(action_status_text,'\0',32);
        // if(last_action_type == UPBOW)
        // {
        //     package.action_type_text = "up";
        //     sprintf(action_status_text," action type : up");
        // }
        // else if (last_action_type == DRAWBOW)
        // {
        //     package.action_type_text = "pull";

        //     sprintf(action_status_text,"action type : pull");
        // }
        // else if(last_action_type == FREE)
        // {
        //     package.action_type_text = "free";
        //     sprintf(action_status_text," action type : free");
        // }
        // else if(last_action_type == PREPARE)
        // {
        //     package.action_type_text = "ready";
        //     sprintf(action_status_text," action type : ready");
        // }
        
        // int font_face = cv::FONT_HERSHEY_COMPLEX; 
        // double font_scale = 1;
        // int thickness = 2;
        // int baseline;
        // cv::Size text_size = cv::getTextSize(action_status_text, font_face, font_scale, thickness, &baseline);
        // cv::Point origin,lorigin; 
        // origin.x =   20;
        // origin.y =   30;
        // cv::putText(src_img, action_status_text, origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 1, 0);
        

        // char action_score_text[32];
        // memset(action_score_text,'\0',32);
        // sprintf(action_score_text," action score : %.3f",score);

        // font_face = cv::FONT_HERSHEY_COMPLEX; 
        // font_scale = 1;
        // thickness = 2;
        // origin.y =   origin.y+text_size.height+5;

        // text_size = cv::getTextSize(action_score_text, font_face, font_scale, thickness, &baseline);
      
        // origin.x =   20;
        // cv::putText(src_img, action_score_text, origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 1, 0);
        
      
        // char action_times_text[32];
        // memset(action_times_text,'\0',32);
        // sprintf(action_times_text," action times : %d",action_counter);

        // font_face = cv::FONT_HERSHEY_COMPLEX; 
        // font_scale = 1;
        // thickness = 2;
        // origin.y =   origin.y+text_size.height+5;

        // text_size = cv::getTextSize(action_times_text, font_face, font_scale, thickness, &baseline);
      
        // origin.x =   20;
        // cv::putText(src_img, action_times_text, origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 1, 0);


        // //best score
        // char action_up_score[32];
        // memset(action_up_score,'\0',32);
        // sprintf(action_up_score," action up : %.3f",up_score);

        // font_face = cv::FONT_HERSHEY_COMPLEX; 
        // font_scale = 1;
        // thickness = 2;
        // origin.y =   origin.y+text_size.height+5;

        // text_size = cv::getTextSize(action_up_score, font_face, font_scale, thickness, &baseline);
      
        // origin.x =   20;
        // cv::putText(src_img, action_up_score, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 1, 0);


        // char action_pull_score[32];
        // memset(action_pull_score,'\0',32);
        // sprintf(action_pull_score," action pull : %.3f",pull_score);

        // font_face = cv::FONT_HERSHEY_COMPLEX; 
        // font_scale = 1;
        // thickness = 2;
        // origin.y =   origin.y+text_size.height+5;

        // text_size = cv::getTextSize(action_pull_score, font_face, font_scale, thickness, &baseline);
      
        // origin.x =   20;
        // cv::putText(src_img, action_pull_score, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 1, 0);

        // cv::Rect rect;
        // rect.x= 20;
        // rect.y = 5;
        // rect.width = 380;
        // rect.height = 150;
        // cv::rectangle(src_img,rect,cv::Scalar(255,0,0),2,1,0);

    }

     char action_status_text[32];
        memset(action_status_text,'\0',32);
        if(last_action_type == UPBOW)
        {
            package.action_type_text = "up";
            sprintf(action_status_text," action type : up");
        }
        else if (last_action_type == DRAWBOW)
        {
            package.action_type_text = "pull";

            sprintf(action_status_text,"action type : pull");
        }
        else if(last_action_type == FREE)
        {
            package.action_type_text = "free";
            sprintf(action_status_text," action type : free");
        }
        else if(last_action_type == PREPARE)
        {
            package.action_type_text = "ready";
            sprintf(action_status_text," action type : ready");
        }
        
        int font_face = cv::FONT_HERSHEY_COMPLEX; 
        double font_scale = 1;
        int thickness = 2;
        int baseline;
        cv::Size text_size = cv::getTextSize(action_status_text, font_face, font_scale, thickness, &baseline);
        cv::Point origin,lorigin; 
        origin.x =   INFER_SHOW_X;
        origin.y =   INFER_SHOW_Y+text_size.height/2+10;
        cv::putText(src_img, action_status_text, origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 1, 0);
        

        char action_score_text[32];
        memset(action_score_text,'\0',32);
        sprintf(action_score_text," action score : %.3f",package.action_score);

        font_face = cv::FONT_HERSHEY_COMPLEX; 
        font_scale = 1;
        thickness = 2;

        text_size = cv::getTextSize(action_score_text, font_face, font_scale, thickness, &baseline);
        origin.y =   origin.y+text_size.height+5;
        origin.x =   20;
        cv::putText(src_img, action_score_text, origin, font_face, font_scale, cv::Scalar(0,0, 255), thickness, 1, 0);
        
      
        char action_times_text[32];
        memset(action_times_text,'\0',32);
        sprintf(action_times_text," action times : %d",action_counter);

        font_face = cv::FONT_HERSHEY_COMPLEX; 
        font_scale = 1;
        thickness = 2;

        text_size = cv::getTextSize(action_times_text, font_face, font_scale, thickness, &baseline);
        origin.y =   origin.y+text_size.height+5;
        origin.x =   20;
        cv::putText(src_img, action_times_text, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 1, 0);


        //best score
        char action_up_score[32];
        memset(action_up_score,'\0',32);
        sprintf(action_up_score," action up : %.3f",up_score);

        font_face = cv::FONT_HERSHEY_COMPLEX; 
        font_scale = 1;
        thickness = 2;
        

        text_size = cv::getTextSize(action_up_score, font_face, font_scale, thickness, &baseline);
        origin.y =   origin.y+text_size.height+5;
        origin.x =   20;
        cv::putText(src_img, action_up_score, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 1, 0);


        char action_pull_score[32];
        memset(action_pull_score,'\0',32);
        sprintf(action_pull_score," action pull : %.3f",pull_score);

        font_face = cv::FONT_HERSHEY_COMPLEX; 
        font_scale = 1;
        thickness = 2;
        text_size = cv::getTextSize(action_pull_score, font_face, font_scale, thickness, &baseline);
        origin.y =   origin.y+text_size.height+5;
        origin.x =   20;
        cv::putText(src_img, action_pull_score, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 1, 0);

        char action_shooting_score[32];
        memset(action_shooting_score,'\0',32);
        sprintf(action_shooting_score," shooting score : %d",int(pull_score/10));

        font_face = cv::FONT_HERSHEY_COMPLEX; 
        font_scale = 1;
        thickness = 2;
       
        text_size = cv::getTextSize(action_shooting_score, font_face, font_scale, thickness, &baseline);
        origin.y =   origin.y+text_size.height+5;
        origin.x =   20;
        
        cv::putText(src_img, action_shooting_score, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 1, 0);

        cv::Rect rect;
        rect.x= 20;
        rect.y = 900;
        rect.width = 380;
        rect.height = 170;
        cv::rectangle(src_img,rect,cv::Scalar(255,0,0),2,1,0);

    return 0.0;
}


        



    