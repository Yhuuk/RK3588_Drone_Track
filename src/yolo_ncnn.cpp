#include "yolo_ncnn.h"

#include <algorithm>
#include <cmath>
#include <iostream>

YoloNcnn::YoloNcnn()
{
    net_.opt.use_vulkan_compute = false;  // 
    net_.opt.num_threads = 4;             //
}

bool YoloNcnn::load(const std::string& param_path, const std::string& bin_path)
{
    if (net_.load_param(param_path.c_str()) != 0)
    {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return false;
    }

    if (net_.load_model(bin_path.c_str()) != 0)
    {
        std::cerr << "Failed to load bin: " << bin_path << std::endl;
        return false;
    }

    loaded_ = true;
    return true;
}

float YoloNcnn::iou(const Detection& a, const Detection& b)
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

//因为output是需要被修改的，所以不带const
void YoloNcnn::nms(const std::vector<Detection>& input,
                   std::vector<Detection>& output,
                   float nms_threshold)
{
    output.clear();     //先把数组output清空，避免旧数据影响
    if (input.empty()) return;

    std::vector<Detection> sorted = input;

    //里面是一个lambda表达式，按照score从大到小排序
    //std::sort(起始位置, 结束位置, 比较规则);  std::sort是C++标准库中的一个排序函数
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b)
              {
                  return a.score > b.score;
              });

    //创建一个和sorted长度一样的数组，并且初始值都给false
    //removed[i] = false：第 i 个框还保留
    //removed[i] = true：第 i 个框已经被压掉
    //这个removed数组只是用给"标记"，而不是直接删除sorted的元素
    std::vector<bool> removed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i)
    {
        if (removed[i]) continue;   //如果第[i]个框已经删除，就直接跳过处理，直接下一次循环
        output.push_back(sorted[i]);    //push_back()的作用是把一个新元素追加到数组的末尾

        for (size_t j = i + 1; j < sorted.size(); ++j)
        {
            if (removed[j]) continue;                       //如果第 j 个框已经被别的更高分框删掉了，那就不用再比较了
            if (iou(sorted[i], sorted[j]) > nms_threshold)      //i后面的框如果大于这个阈值，就直接删掉
            {
                removed[j] = true;  
            }
        }
    }
}


//cv::Mat 是 OpenCV 的图像/矩阵类
//detections是最终要输出的检测结果列表，conf_threshold是低分阈值，nms_threshold删重叠框的阈值
bool YoloNcnn::detect(const cv::Mat& bgr,
                      std::vector<Detection>& detections,
                      float conf_threshold,
                      float nms_threshold)
{
    detections.clear();

    //loaded_是YoloNcnn的成员变量
    if (!loaded_)
    {
        std::cerr << "Model is not loaded." << std::endl;
        return false;
    }

    //调用cv::Mat::empty(),
    if (bgr.empty())
    {
        std::cerr << "Input image is empty." << std::endl;
        return false;
    }

    //取原图的宽高
    const int img_w = bgr.cols;     //图像列数
    const int img_h = bgr.rows;     //图像行数

    // -------- 1) letterbox 到 640x640 --------
    /************************************************************************
     * static_cast<float>()是C++的显示类型转换写法，把target_size转换成Float类型
     * std::min()函数返回较小值
     * std::round()四舍五入
     ***********************************************************************/
    float scale = std::min(static_cast<float>(target_size_) / img_w,
                           static_cast<float>(target_size_) / img_h);

    int resized_w = static_cast<int>(std::round(img_w * scale));
    int resized_h = static_cast<int>(std::round(img_h * scale));

    /************************************************************************
     * bgr.data：OpenCV 图像底层像素数据指针。OpenCV 文档明确说明 Mat::data 是用户数据指针
     * ncnn::Mat::PIXEL_BGR2RGB：输入像素原来是 BGR 顺序，但模型想要 RGB，所以在构造输入时顺便做通道顺序转换。
     * img_w, img_h：图片原始宽高
     * resized_w, resized_h：映射完后的图片宽高
     * 
     ***********************************************************************/
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img_w, img_h,
        resized_w, resized_h
    );

    //宽和高还差多少能补到640
    //了解padding这个结果？
    int pad_w = target_size_ - resized_w;
    int pad_h = target_size_ - resized_h;

    //把宽和高的尺寸差分别补到左右和上下
    //chatgpt：这就是典型的居中 letterbox。
    int left   = pad_w / 2;
    int right  = pad_w - left;
    int top    = pad_h / 2;
    int bottom = pad_h - top;

    //border_constant
    /**********
     *   in：原来的缩放图
     *   in_pad：输出结果，补边后的图
     *   top, bottom, left, right：上下左右各补多少
     *   ncnn::BORDER_CONSTANT：用常量值补边
     *   114.f：常量值就是 114
     * 函数的整体意思:  在缩放后的图外面，补一圈值为 114 的边，把图凑成 640×640。
     * 
     */
    ncnn::Mat in_pad;
    ncnn::copy_make_border(
        in, in_pad,
        top, bottom, left, right,
        ncnn::BORDER_CONSTANT, 114.f
    );

    /**********
     * ncnn 的 substract_mean_normalize(mean_vals, norm_vals) 在 mat.h 里的注释写得很直白：先按通道减去均值，再乘以归一化系数；传 0 表示跳过该步骤。
     * 函数的实际做法是：不减均值，只把像素值乘以 1/255，也就是把 0~255 缩放到 0~1。
     * 不是很理解，需要去明白这个归一化是怎么做的
     */
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // -------- 2) 推理 --------
    /***********
     * ncnn::Net net_   这是.h文件中定义过的
     * 
     * 
     * ********* */
    //ncnn::Extractor的实际意思就是：从已经加载好的网络 net_ 创建一个本次推理用的执行器 ex。
    ncnn::Extractor ex = net_.create_extractor();
    // ex.set_num_threads(4);


    // "in0"：模型输入 blob 名字，in_pad：预处理后的 640×640 输入
    //你代码里用 != 0 判失败，这是 ncnn C++ 接口里很常见的写法风格；你的代码逻辑就是“非 0 就认为出错”。
    if (ex.input("in0", in_pad) != 0)
    {
        std::cerr << "Failed to set input blob: in0" << std::endl;
        return false;
    }


    //"out0"：输出 blob 名，out：接收输出张量
    //这里得到的 out 不是最终“框对象数组”，而只是模型原始输出张量。
    //ex.extract("out0", out)：真正的神经网络前向推理
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0)       
    {
        std::cerr << "Failed to extract output blob: out0" << std::endl;
        return false;
    }

    //这个打印的作用：程序第一次跑到这里时，把输出张量形状打出来，方便确认模型输出到底是什么布局
    //在 mat.h 里都能看到。dims 表示维度数，w/h/c 是该 Mat 的形状信息成员。w/h/c分别是什么，特别是c
    //out.c表示没有第三维通道，所以c默认为1
    if (!printed_shape_)
    {
        std::cout << "[INFO] out.dims = " << out.dims
                  << ", out.w = " << out.w
                  << ", out.h = " << out.h
                  << ", out.c = " << out.c << std::endl;
        printed_shape_ = true;
    }

    
    // A: out.h == 5, out.w == N
    // B: out.w == 5, out.h == N
    //feat_dim：每个预测项有多少个数，num_preds：一共有多少个候选预测，layout_hw：记录到底是哪种排布方式

    int feat_dim = -1;// 
    int num_preds = -1;
    bool layout_hw = false; // true 代表 out.h=5, out.w=N

    //如果输出根本不是你当前这套解析逻辑支持的格式，就直接退出// @link@:images/f30b5af6-ad68-4ecf-8047-6983320904d6.png
    //这里有两种格式：5 x N或者N x 5
    //现在确认的是out.h = 5，out.w = 33600。需要完全弄清楚33600代表了什么
    if (out.dims == 2)  //二维// @link@:images/f30b5af6-ad68-4ecf-8047-6983320904d6.png
    {
        if (out.h == 5)
        {
            feat_dim = out.h;
            num_preds = out.w;
            layout_hw = true;
        }
        else if (out.w == 5)
        {
            feat_dim = out.w;
            num_preds = out.h;
            layout_hw = false;
        }
    }

    //如果输出不合规，不是当前这个解析逻辑，就输出提示语后直接退出
    if (feat_dim != 5 || num_preds <= 0)
    {
        std::cerr << "Unexpected output shape. "
                  << "Need single-class output shaped like 5xN or Nx5." << std::endl;
        return false;
    }

    //proposals是临时候选框列表
    std::vector<Detection> proposals;
    proposals.reserve(num_preds);       //先给这个 vector 预留足够空间，等会儿装候选框更省事，num_preds =-1是什么意思

    for (int i = 0; i < num_preds; ++i)
    {
        //cx：框中心x，cy：框中心 y，w：框宽，h：框高，score：分数
        float cx, cy, w, h, score;

        if (layout_hw)
        {
            // out.h = 5, out.w = N
            //这些都是指针类型
            const float* row0 = out.row(0);
            const float* row1 = out.row(1);
            const float* row2 = out.row(2);
            const float* row3 = out.row(3);
            const float* row4 = out.row(4);

            //第 i 个预测的 5 个字段，分别从 5 行的第 i 列取出来
            cx    = row0[i];
            cy    = row1[i];
            w     = row2[i];
            h     = row3[i];
            score = row4[i];
        }
        else
        {
            // out.h = N, out.w = 5
            const float* row = out.row(i);
            cx    = row[0];
            cy    = row[1];
            w     = row[2];
            h     = row[3];
            score = row[4];
        }

        //根据分数阈值先过滤，如果低于阈值，直接进下一轮循环
        if (score < conf_threshold) continue;

        //按中心坐标(cx,cy)并根据框的大小计算出左上角坐标(x0,y0)和右上角坐标(x1,y1)
        float x0 = cx - w * 0.5f;
        float y0 = cy - h * 0.5f;
        float x1 = cx + w * 0.5f;
        float y1 = cy + h * 0.5f;

        // 把letterbox坐标系中的框，还原到原图坐标系
        //因为推理不是直接用的原图，所以输出想要回到原图，就要按着比例反向映射回去
        x0 = (x0 - left)  / scale;
        y0 = (y0 - top)   / scale;
        x1 = (x1 - left)  / scale;
        y1 = (y1 - top)   / scale;

        //img_w是原图的宽
        // clip，裁剪到图像边界内
        /*****
         *  1- 先保证坐标不超过右边/下边界
         *  2- 再保证坐标不小于 0
         *  这个裁剪的实际作用：也就是把坐标强行压进合法范围内。
         * 
         * ***************** */
        x0 = std::max(0.0f, std::min(x0, static_cast<float>(img_w - 1)));
        y0 = std::max(0.0f, std::min(y0, static_cast<float>(img_h - 1)));
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_h - 1)));

        //因为 cv::Rect_<float> 需要的不是右下角坐标。而是左上角，宽和高
        //重新计算框的实际反映射后的实际框的宽和高
        float bw = x1 - x0;
        float bh = y1 - y0;

        //如果这个框宽或高几乎没有意义，就直接丢掉。
        if (bw <= 1.0f || bh <= 1.0f) continue;

        //创建一个Detection对象
        Detection det;
        //创建一个浮点型矩形框：左上角在 (x0, y0)，宽 bw，高 bh
        det.box = cv::Rect_<float>(x0, y0, bw, bh);
        det.score = score;
        det.label = 0; // 单类别检测 UAV-Thermal，ID固定为0

        proposals.push_back(det);
    }

    //proposals只是通过置信度阈值的框，里面还有重复框
    //proposals是原始候选，detections是最终输出
    nms(proposals, detections, nms_threshold);

    //true只表示这次推理流程走完了，不代表一定检测到了目标
    return true;
}