#include "yolo_ncnn.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& det : dets)
    {
        cv::rectangle(frame, det.box, cv::Scalar(255, 0, 0), 2);

        std::ostringstream oss;
        oss << "UAV-Thermal " << std::fixed << std::setprecision(2) << det.score;
        std::string label = oss.str();

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label,
                                              cv::FONT_HERSHEY_SIMPLEX,
                                              0.7, 2, &baseLine);

        int x = static_cast<int>(det.box.x);
        int y = static_cast<int>(det.box.y) - 6;
        if (y < label_size.height) y = static_cast<int>(det.box.y) + label_size.height + 6;

        cv::rectangle(frame,
                      cv::Rect(x, y - label_size.height - 4,
                               label_size.width + 6, label_size.height + 6),
                      cv::Scalar(255, 0, 0), cv::FILLED);

        cv::putText(frame, label,
                    cv::Point(x + 3, y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(255, 255, 255), 2);
    }
}

int main()
{
    const std::string param_path = "../models/fz00_ncnn_model/model.ncnn.param";
    const std::string bin_path   = "../models/fz00_ncnn_model/model.ncnn.bin";

    YoloNcnn detector;
    if (!detector.load(param_path, bin_path))
    {
        std::cerr << "Failed to load NCNN model." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open /dev/video0" << std::endl;
        return -1;
    }

    // 浣犲凡缁忕‘璁? video0 鑳藉嚭鐑?鎴愬儚锛屽厛鎸? 640x512 璇?
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 512);

    std::cout << "Camera opened." << std::endl;
    std::cout << "Width  = " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Height = " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "FPS    = " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    float conf_threshold = 0.50f;
    float nms_threshold  = 0.45f;

    cv::Mat frame;
    auto t_last = std::chrono::steady_clock::now();
    float fps = 0.0f;

    // while (true)
    // {
    //     cap >> frame;
    //     if (frame.empty())
    //     {
    //         std::cerr << "Empty frame from camera." << std::endl;
    //         break;
    //     }

    //     std::vector<Detection> detections;
    //     bool ok = detector.detect(frame, detections, conf_threshold, nms_threshold);
    //     if (!ok)
    //     {
    //         std::cerr << "detect() failed." << std::endl;
    //     }

    //     //
    //     cv::Mat show = frame.clone();
    //     draw_detections(show, detections);

    //     auto t_now = std::chrono::steady_clock::now();
    //     float dt = std::chrono::duration<float>(t_now - t_last).count();
    //     t_last = t_now;
    //     if (dt > 0.0f) fps = 1.0f / dt;

    //     //dets是最终输出结果的数量（比如最终只检测到了一个无人机）
    //     std::ostringstream info;
    //     info << "FPS: " << std::fixed << std::setprecision(1) << fps
    //          << "  conf: " << std::setprecision(2) << conf_threshold
    //          << "  dets: " << detections.size();

    //     //实时显示画面输出的pfs打印在终端上
    //     std::cout <<"[INFO_2] Frame_PFS: " << fps <<std::endl;

    //     cv::putText(show, info.str(), cv::Point(15, 30),
    //                 cv::FONT_HERSHEY_SIMPLEX, 0.8,
    //                 cv::Scalar(0, 255, 0), 2);

    //     cv::putText(show, "q: quit  [/: conf -/+  s: save frame", cv::Point(15, 60),
    //                 cv::FONT_HERSHEY_SIMPLEX, 0.6,
    //                 cv::Scalar(0, 255, 255), 2);

    //     cv::imshow("NCNN Real-time Detect", show);
    //     char key = static_cast<char>(cv::waitKey(1));

    //     if (key == 'q')
    //     {
    //         break;
    //     }
    //     else if (key == '[')
    //     {
    //         conf_threshold = std::max(0.05f, conf_threshold - 0.05f);
    //     }
    //     else if (key == ']')
    //     {
    //         conf_threshold = std::min(0.95f, conf_threshold + 0.05f);
    //     }
    //     else if (key == 's')
    //     {
    //         static int idx = 0;
    //         std::ostringstream save_path;
    //         save_path << "/home/cat/models/Image_detected/debug_frame_" << idx++ << ".jpg";
    //         cv::imwrite(save_path.str(), show);
    //         std::cout << "Saved: " << save_path.str() << std::endl;
    //     }
    // }


    while (true)
{
    auto t0 = std::chrono::steady_clock::now();     //开始计时

    cap >> frame;

    auto t1 = std::chrono::steady_clock::now();

    if (frame.empty())
    {
        std::cerr << "Empty frame from camera." << std::endl;
        break;
    }

    std::vector<Detection> detections;
    bool ok = detector.detect(frame, detections, conf_threshold, nms_threshold);

    auto t2 = std::chrono::steady_clock::now();

    if (!ok)
    {
        std::cerr << "detect() failed." << std::endl;
    }

    cv::Mat show = frame.clone();
    draw_detections(show, detections);

    auto t3 = std::chrono::steady_clock::now();

    auto t_now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(t_now - t_last).count();
    t_last = t_now;
    if (dt > 0.0f) fps = 1.0f / dt;

    std::ostringstream info;
    info << "FPS: " << std::fixed << std::setprecision(1) << fps
         << "  conf: " << std::setprecision(2) << conf_threshold
         << "  dets: " << detections.size();

    cv::putText(show, info.str(), cv::Point(15, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);

    cv::putText(show, "q: quit  [/: conf -/+  s: save frame", cv::Point(15, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);

    cv::imshow("NCNN Real-time Detect", show);

    auto t4 = std::chrono::steady_clock::now();
    
    char key = static_cast<char>(cv::waitKey(1));

    

    auto ms_read = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ms_detect = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto ms_draw = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto ms_show = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t0).count();

    std::cout << "[TIME] read=" << ms_read
              << " ms, detect=" << ms_detect
              << " ms, draw=" << ms_draw
              << " ms, show=" << ms_show
              << " ms, total=" << ms_total
              << " ms" << std::endl;

    if (key == 'q')
    {
        break;
    }
    else if (key == '[')
    {
        conf_threshold = std::max(0.05f, conf_threshold - 0.05f);
    }
    else if (key == ']')
    {
        conf_threshold = std::min(0.95f, conf_threshold + 0.05f);
    }
    else if (key == 's')
    {
        static int idx = 0;
        std::ostringstream save_path;
        save_path << "/home/cat/models/Image_detected/debug_frame_" << idx++ << ".jpg";
        cv::imwrite(save_path.str(), show);
        std::cout << "Saved: " << save_path.str() << std::endl;
    }
}

    cap.release();
    cv::destroyAllWindows();
    return 0;
}