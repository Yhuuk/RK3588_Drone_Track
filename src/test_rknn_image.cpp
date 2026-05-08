#include <opencv2/opencv.hpp>

#include <rknn_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ======================= 模型输入输出参数 =======================
//
// 你的 RKNN 模型输入是：images [1, 3, 512, 640]
// 这里的意思是：
// batch = 1
// channel = 3
// height = 512
// width = 640
//
// OpenCV 图像一般是 HWC 格式：高 × 宽 × 通道
// 所以后面我们会把图片 resize 到 640×512，然后 BGR 转 RGB。
static const int MODEL_W = 640;
static const int MODEL_H = 512;

// 你的 RKNN 输出是：output0 [1, 5, 26880]
// 5 表示每个候选框有 5 个数：cx, cy, w, h, score
// 26880 表示候选预测数量。
static const int NUM_CLASSES = 1;

// ======================= 检测框结构体 =======================
//
// 这个结构体和你 yolo_ncnn.h 里的 Detection 思路一样。
// box：检测框
// score：置信度
// label：类别编号。你的模型只有 UAV-Thermal 一个类别，所以固定为 0。
struct Detection
{
    cv::Rect_<float> box;
    float score;
    int label;
};

// ======================= 读取 RKNN 模型文件 =======================
//
// RKNN C API 的 rknn_init 需要传入模型二进制数据。
// 所以这里先把 fz00.rknn 整个文件读到 vector<unsigned char> 里。
static bool read_file(const std::string& path, std::vector<unsigned char>& data)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Failed to open model file: " << path << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    if (size <= 0)
    {
        std::cerr << "Invalid model file size: " << path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));

    if (!file.read(reinterpret_cast<char*>(data.data()), size))
    {
        std::cerr << "Failed to read model file: " << path << std::endl;
        return false;
    }

    return true;
}

// ======================= IOU 计算 =======================
//
// NMS 会用到 IOU。
// IOU 越大，说明两个框重叠越严重。
static float iou(const Detection& a, const Detection& b)
{
    float x1 = std::max(a.box.x, b.box.x);
    float y1 = std::max(a.box.y, b.box.y);
    float x2 = std::min(a.box.x + a.box.width,  b.box.x + b.box.width);
    float y2 = std::min(a.box.y + a.box.height, b.box.y + b.box.height);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.box.width * a.box.height;
    float area_b = b.box.width * b.box.height;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

// ======================= NMS 去重 =======================
//
// 模型会输出很多候选框。
// 同一个无人机附近可能会有多个重叠框。
// NMS 的作用是：保留分数最高的框，删掉重叠严重的框。
static void nms(const std::vector<Detection>& input,
                std::vector<Detection>& output,
                float nms_threshold)
{
    output.clear();
    if (input.empty()) return;

    std::vector<Detection> sorted = input;

    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b)
              {
                  return a.score > b.score;
              });

    std::vector<bool> removed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i)
    {
        if (removed[i]) continue;

        output.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); ++j)
        {
            if (removed[j]) continue;

            if (iou(sorted[i], sorted[j]) > nms_threshold)
            {
                removed[j] = true;
            }
        }
    }
}

// ======================= 画检测框 =======================
static void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& det : dets)
    {
        cv::Rect rect(
            static_cast<int>(det.box.x),
            static_cast<int>(det.box.y),
            static_cast<int>(det.box.width),
            static_cast<int>(det.box.height)
        );

        cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);

        std::ostringstream oss;
        oss << "UAV-Thermal " << std::fixed << std::setprecision(2) << det.score;
        std::string label = oss.str();

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label,
                                              cv::FONT_HERSHEY_SIMPLEX,
                                              0.6, 2, &baseLine);

        int x = rect.x;
        int y = rect.y - 6;
        if (y < label_size.height)
        {
            y = rect.y + label_size.height + 6;
        }

        cv::rectangle(frame,
                      cv::Rect(x, y - label_size.height - 4,
                               label_size.width + 6, label_size.height + 6),
                      cv::Scalar(255, 0, 0),
                      cv::FILLED);

        cv::putText(frame,
                    label,
                    cv::Point(x + 3, y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    2);
    }
}

// ======================= 打印 RKNN 张量信息 =======================
static void print_tensor_attr(const char* tag, const rknn_tensor_attr& attr)
{
    std::cout << tag
              << " index=" << attr.index
              << " name=" << attr.name
              << " n_dims=" << attr.n_dims
              << " dims=[";

    for (int i = 0; i < attr.n_dims; ++i)
    {
        std::cout << attr.dims[i];
        if (i + 1 < attr.n_dims) std::cout << ", ";
    }

    std::cout << "]"
              << " n_elems=" << attr.n_elems
              << " size=" << attr.size
              << " fmt=" << attr.fmt
              << " type=" << attr.type
              << " qnt_type=" << attr.qnt_type
              << " zp=" << attr.zp
              << " scale=" << attr.scale
              << std::endl;
}

// ======================= 解析 output0 [1, 5, 26880] =======================
//
// output0 的逻辑：
// 第 0 行：所有候选框的 cx
// 第 1 行：所有候选框的 cy
// 第 2 行：所有候选框的 w
// 第 3 行：所有候选框的 h
// 第 4 行：所有候选框的 score
//
// 也就是：
// data[0 * N + i] -> 第 i 个框的 cx
// data[1 * N + i] -> 第 i 个框的 cy
// data[2 * N + i] -> 第 i 个框的 w
// data[3 * N + i] -> 第 i 个框的 h
// data[4 * N + i] -> 第 i 个框的 score
static bool decode_output(const float* data,
                          const rknn_tensor_attr& out_attr,
                          int orig_w,
                          int orig_h,
                          float conf_threshold,
                          float nms_threshold,
                          std::vector<Detection>& final_dets)
{
    final_dets.clear();

    int feat_dim = -1;
    int num_preds = -1;
    bool layout_5_N = false;

    // 你的模型是 [1, 5, 26880]
    // 这里写得稍微通用一点：
    // 如果发现 dims 中是 [1, 5, N]，就按 5×N 解析。
    // 如果是 [1, N, 5]，就按 N×5 解析。
    if (out_attr.n_dims == 3)
    {
        if (out_attr.dims[1] == 5)
        {
            feat_dim = 5;
            num_preds = out_attr.dims[2];
            layout_5_N = true;
        }
        else if (out_attr.dims[2] == 5)
        {
            feat_dim = 5;
            num_preds = out_attr.dims[1];
            layout_5_N = false;
        }
    }
    else if (out_attr.n_dims == 2)
    {
        if (out_attr.dims[0] == 5)
        {
            feat_dim = 5;
            num_preds = out_attr.dims[1];
            layout_5_N = true;
        }
        else if (out_attr.dims[1] == 5)
        {
            feat_dim = 5;
            num_preds = out_attr.dims[0];
            layout_5_N = false;
        }
    }

    if (feat_dim != 5 || num_preds <= 0)
    {
        std::cerr << "Unsupported output shape. Need [1,5,N] or [1,N,5]." << std::endl;
        return false;
    }

    std::cout << "[INFO] Decode output: feat_dim=" << feat_dim
              << ", num_preds=" << num_preds
              << ", layout=" << (layout_5_N ? "5xN" : "Nx5")
              << std::endl;

    // 因为模型输入是 640×512。
    // 如果原图也是 640×512，这两个比例就是 1。
    // 如果测试图不是 640×512，前处理会 resize 到 640×512，
    // 所以后处理需要按比例映射回原图尺寸。
    float scale_x = static_cast<float>(orig_w) / static_cast<float>(MODEL_W);
    float scale_y = static_cast<float>(orig_h) / static_cast<float>(MODEL_H);

    std::vector<Detection> proposals;
    proposals.reserve(num_preds);

    for (int i = 0; i < num_preds; ++i)
    {
        float cx, cy, w, h, score;

        if (layout_5_N)
        {
            cx    = data[0 * num_preds + i];
            cy    = data[1 * num_preds + i];
            w     = data[2 * num_preds + i];
            h     = data[3 * num_preds + i];
            score = data[4 * num_preds + i];
        }
        else
        {
            const float* row = data + i * 5;
            cx    = row[0];
            cy    = row[1];
            w     = row[2];
            h     = row[3];
            score = row[4];
        }

        if (score < conf_threshold) continue;

        float x0 = cx - w * 0.5f;
        float y0 = cy - h * 0.5f;
        float x1 = cx + w * 0.5f;
        float y1 = cy + h * 0.5f;

        // 从模型输入坐标系 640×512 映射回原图坐标系
        x0 *= scale_x;
        x1 *= scale_x;
        y0 *= scale_y;
        y1 *= scale_y;

        x0 = std::max(0.0f, std::min(x0, static_cast<float>(orig_w - 1)));
        y0 = std::max(0.0f, std::min(y0, static_cast<float>(orig_h - 1)));
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h - 1)));

        float bw = x1 - x0;
        float bh = y1 - y0;

        if (bw <= 1.0f || bh <= 1.0f) continue;

        Detection det;
        det.box = cv::Rect_<float>(x0, y0, bw, bh);
        det.score = score;
        det.label = 0;

        proposals.push_back(det);
    }

    std::cout << "[INFO] proposals before NMS: " << proposals.size() << std::endl;

    nms(proposals, final_dets, nms_threshold);

    std::cout << "[INFO] final detections after NMS: " << final_dets.size() << std::endl;

    return true;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <model.rknn> <input.jpg> <output.jpg>\n\n"
                  << "Example:\n"
                  << "  " << argv[0]
                  << " /home/cat/models/fz00_rknn_model/fz00.rknn"
                  << " /home/cat/xshell_test/热成像照片/20260416142731_6_279.jpg"
                  << " /home/cat/xshell_test/rknn_test_image/rknn_result.jpg\n";
        return -1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string save_path  = argv[3];

    float conf_threshold = 0.50f;
    float nms_threshold  = 0.45f;

    // 1. 读取 RKNN 模型文件
    std::vector<unsigned char> model_data;
    if (!read_file(model_path, model_data))
    {
        return -1;
    }

    // 2. 初始化 RKNN Runtime
    rknn_context ctx = 0;

    int ret = rknn_init(&ctx,
                        model_data.data(),
                        static_cast<uint32_t>(model_data.size()),
                        0,
                        nullptr);

    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_init failed! ret=" << ret << std::endl;
        return -1;
    }

    std::cout << "[INFO] rknn_init success." << std::endl;



    ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_set_core_mask failed! ret=" << ret << std::endl;
    }
    else
    {
        std::cout << "[INFO] set NPU core mask: RKNN_NPU_CORE_0_1_2" << std::endl;
    }


    // 3. 查询输入输出数量
    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    std::cout << "[INFO] n_input=" << io_num.n_input
              << ", n_output=" << io_num.n_output << std::endl;

    if (io_num.n_input != 1 || io_num.n_output != 1)
    {
        std::cerr << "This demo expects 1 input and 1 output." << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    // 4. 查询输入 tensor 属性
    rknn_tensor_attr input_attr;
    std::memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;

    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_query input attr failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    print_tensor_attr("[INPUT]", input_attr);

    // 5. 查询输出 tensor 属性
    rknn_tensor_attr output_attr;
    std::memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;

    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_query output attr failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    print_tensor_attr("[OUTPUT]", output_attr);

    // 6. 读取图片
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    int orig_w = bgr.cols;
    int orig_h = bgr.rows;

    std::cout << "[INFO] input image size: "
              << orig_w << "x" << orig_h << std::endl;

    // 7. 预处理：resize 到 640×512，然后 BGR 转 RGB
    //
    // 你的模型输入是 [1,3,512,640]。
    // OpenCV 读出来是 BGR 顺序。
    // YOLO 通常训练/导出时使用 RGB 顺序。
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(MODEL_W, MODEL_H));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    if (!rgb.isContinuous())
    {
        rgb = rgb.clone();
    }

    // 8. 设置 RKNN 输入
    //
    // 这里输入 uint8 RGB 图像。
    // 如果你转换 RKNN 时设置了：
    // mean_values=[[0,0,0]]
    // std_values=[[255,255,255]]
    //
    // 那么这里不要手动除以 255。
    // 直接喂 uint8，RKNN 内部会按转换配置处理。
    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = MODEL_W * MODEL_H * 3;
    inputs[0].buf = rgb.data;
    inputs[0].pass_through = 0;

    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_inputs_set failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

// 9. 运行 NPU 推理
//
// 注意：第一次 rknn_run 可能包含额外初始化/预热开销。
// 所以这里先 warmup 几次，再连续跑多次取平均值。

const int WARMUP_RUNS = 5;
const int TEST_RUNS = 30;

// 预热运行，不计入最终耗时
for (int i = 0; i < WARMUP_RUNS; ++i)
{
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_run warmup failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
}

// 正式计时
auto t0 = std::chrono::steady_clock::now();

for (int i = 0; i < TEST_RUNS; ++i)
{
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_run failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
}

auto t1 = std::chrono::steady_clock::now();

float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
float avg_ms = total_ms / TEST_RUNS;
float avg_fps = 1000.0f / avg_ms;

std::cout << "[TIME] rknn_run avg = " << avg_ms
          << " ms, fps = " << avg_fps
          << " over " << TEST_RUNS << " runs" << std::endl;

    // 10. 获取输出
    //
    // want_float=1 表示把输出转成 float，方便我们直接按照 cx,cy,w,h,score 解析。
    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 1;

    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_outputs_get failed! ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }

    const float* out_data = reinterpret_cast<const float*>(outputs[0].buf);

    // 11. 后处理：解析 output0 [1,5,26880]，NMS，得到最终检测框
    std::vector<Detection> detections;

    bool decode_ok = decode_output(out_data,
                                   output_attr,
                                   orig_w,
                                   orig_h,
                                   conf_threshold,
                                   nms_threshold,
                                   detections);

    // 12. 释放 RKNN 输出
    rknn_outputs_release(ctx, 1, outputs);

    if (!decode_ok)
    {
        rknn_destroy(ctx);
        return -1;
    }

    // 13. 画框并保存图片
    cv::Mat result = bgr.clone();
    draw_detections(result, detections);

    cv::imwrite(save_path, result);

    std::cout << "[INFO] result saved to: " << save_path << std::endl;

    // 14. 销毁 RKNN 上下文
    rknn_destroy(ctx);

    return 0;
}