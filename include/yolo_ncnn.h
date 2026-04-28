#pragma once

#include <opencv2/opencv.hpp>
//#include <ncnn/net.h>   这个会报错
#include <net.h>
#include <string>
#include <vector>

//std::vector是C++ 标准库 里的容器，本质就是可自动扩容的数组
//std::vector<Detection>表示"一个存很多 Detection 类型对象的动态数组",其中Detection是自定义的类型，通常是struct或class
//std::vector<某种类型>
struct Detection
{
    cv::Rect_<float> box;
    float score;
    int label;
};

class YoloNcnn
{
public:
    YoloNcnn();

    bool load(const std::string& param_path, const std::string& bin_path);

    bool detect(const cv::Mat& bgr,
                std::vector<Detection>& detections,
                float conf_threshold = 0.50f,
                float nms_threshold = 0.45f);

private:
    static float iou(const Detection& a, const Detection& b);
    static void nms(const std::vector<Detection>& input,
                    std::vector<Detection>& output,
                    float nms_threshold);

private:
    ncnn::Net net_;
    bool loaded_ = false;
    bool printed_shape_ = false;

    // 你的导出模型是 imgsz=640
    int target_size_ = 640;

    // 目前你的模型是单类 UAV-Thermal
    int num_classes_ = 1;
};