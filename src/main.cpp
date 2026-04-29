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

        //无人机主检测框，cv::FILLED表示填空矩形
        /**
         * frame：画在哪张图上，
         * cv::Rect：这个小底板的位置和大小（左上角，右下角，宽，高）
         * cv::Scalar(0, 0, 255)：颜色通道为BGR
         * cv::FILLED：填充，表示画实心矩形
         */
        cv::rectangle(frame,
                      cv::Rect(x, y - label_size.height - 4,
                               label_size.width + 6, label_size.height + 6),
                      cv::Scalar(255, 0, 0), cv::FILLED);

        /**
        frame：画到哪张图上
        label：要显示的文字内容，比如 UAV-Thermal 0.87
        cv::Point(x + 3, y)：文字起点位置
        cv::FONT_HERSHEY_PLAIN：字体
        0.7：字体缩放倍数
        cv::Scalar(255, 255, 255)：白色文字
        2：文字线宽
         */
        cv::putText(frame, label,
                    cv::Point(x + 3, y),
                    cv::FONT_HERSHEY_PLAIN,
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

    // 尽量让摄像头缓存队列变小，避免程序处理旧帧。
    // 注意：这个设置在部分 V4L2 摄像头上可能不完全生效，但可以先加上。
    cap.set(cv::CAP_PROP_BUFFERSIZE,1);

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

    // ================= 新增：YOLO 跳帧检测参数 =================

    // 每隔多少帧运行一次 YOLO。
    // 例如 DETECT_INTERVAL = 5：
    // 第 0、5、10、15... 帧运行 YOLO，其余帧复用上一帧检测框。
    const int DETECT_INTERVAL = 3;

    // 当前处理到第几帧，用来判断这一帧是否需要运行 YOLO。
    long long frame_count = 0;

    // 保存最近一次 YOLO 的检测结果。
    // 中间不跑 YOLO 的帧，就直接复用这个检测结果。
    std::vector<Detection> last_detections;

    // 记录 YOLO 单次检测 FPS。
    // 注意：这个是 YOLO 检测速度，不是显示 FPS。
    float yolo_fps = 0.0f;

    bool frame_wh = true;
    
    while (true)
    {
        cap >> frame;

        if(frame_wh){
            std::cout << "w= " << frame.cols << ",  h= " << frame.rows << std::endl;
            frame_wh = false;
        }

        if (frame.empty())
        {
            std::cerr << "Empty frame from camera." << std::endl;
            break;
        }

        // std::vector<Detection> detections;
        // bool ok = detector.detect(frame, detections, conf_threshold, nms_threshold);
        // if (!ok)
        // {
        //     std::cerr << "detect() failed." << std::endl;
        // }

        // //
        // cv::Mat show = frame.clone();
        // draw_detections(show, detections);

        // 判断当前这一帧是否需要运行 YOLO。
        // frame_count % DETECT_INTERVAL == 0 表示每隔 DETECT_INTERVAL 帧检测一次。
        bool run_yolo_this_frame = (frame_count % DETECT_INTERVAL == 0);


        if (run_yolo_this_frame)
        {
            // 只有在指定帧才真正运行 YOLO。
            // 这一步最耗时，你前面测出来大约是 150 ms。
            std::vector<Detection> detections;

            auto t_yolo_start = std::chrono::steady_clock::now();

            bool ok = detector.detect(frame, detections, conf_threshold, nms_threshold);

            auto t_yolo_end = std::chrono::steady_clock::now();

            if (!ok)
            {
                std::cerr << "detect() failed." << std::endl;
            }
            else
            {
                // YOLO 成功后，把这一次检测结果保存下来。
                // 后面跳过 YOLO 的帧，会复用 last_detections。
                last_detections = detections;
            }

            // 计算 YOLO 单次检测速度。
            // 这个值大概率还是 6~7 FPS，因为 YOLO 本身没有变快。
            //std::chrono::duration是一个类，表示时间间隔
            float yolo_dt = std::chrono::duration<float>(t_yolo_end - t_yolo_start).count();
            if (yolo_dt > 0.0f)
            {
                yolo_fps = 1.0f / yolo_dt;
            }
        }

        // 复制当前摄像头画面用于显示。
        // frame 是当前最新画面，show 是要画框和文字的显示画面。
        cv::Mat show = frame.clone();

        // 不管这一帧有没有运行 YOLO，都画最近一次检测结果。
        // 如果这一帧没跑 YOLO，就复用 last_detections。
        draw_detections(show, last_detections);

        /********************
         * std::chrono::steady_clock::now()->它不是给你“现在是几点几分几秒”，而是给你一个适合做计时的“时间点”
         * std::chrono::duration<float>(t_now - t_last)是一个float类型的时间长度对象，不是纯数字时间。.count()就是把这个时间长度对象转换成数字
         * ******************************** */
        auto t_now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(t_now - t_last).count();
        t_last = t_now;
        if (dt > 0.0f) fps = 1.0f / dt;

        
        /*Display FPS：当前显示循环帧率
        YOLO FPS：YOLO 自己的检测速度
        conf：置信度阈值
        dets：最近一次 YOLO 检测出来的目标数量
        mode: detect：这一帧运行了 YOLO
        mode: reuse：这一帧没有运行 YOLO，复用上一帧检测框
        
        std::fixed << std::setprecision(1)这两个一起用就是固定显示成小数点后2位
        std::fixed ->让浮点数用“固定小数形式”显示。
        std::setprecision(1) ->设置精度的格式控制符,保留1位小数
        
        */
        //
        std::ostringstream info;    //info就是字符缓存。
        info << "Display FPS: " << std::fixed << std::setprecision(1) << fps
        << "  YOLO FPS: " << std::setprecision(1) << yolo_fps
        << "  conf: " << std::setprecision(2) << conf_threshold
        << "  dets: " << last_detections.size()
        << "  mode: " << (run_yolo_this_frame ? "detect" : "reuse");

        //终端上打印显示画面的帧率
        // std::cout <<"[INFO_2] Frame_PFS: " << fps <<std::endl;

        //cv::putText才把info中的字符打印到画面中，info.str()转换成可用的字符串
        cv::putText(show, info.str(), cv::Point(15, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);

        cv::putText(show, "q: quit  [/: conf -/+  s: save frame", cv::Point(15, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);

        cv::imshow("NCNN Real-time Detect", show);
        char key = static_cast<char>(cv::waitKey(1));

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

        // 当前帧处理完成，帧计数加 1。
        // 下一轮循环会根据 frame_count 判断是否运行 YOLO。
        frame_count++;
    }
}

