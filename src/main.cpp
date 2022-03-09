//
//  main.cpp
//  machine-learning-deployment
//
//  Created by xiaohao on 2022/3/4.
//

#include <iostream>
#include <onnx/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <array>
#include <algorithm>
#include <unistd.h>


#define constexpr const

using namespace std;

int main(){

    clock_t start_time = clock();
    
    // 创建启动环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX");

    // 优化设置
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(int(sysconf(_SC_NPROCESSORS_ONLN)));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 内存分配
    Ort::AllocatorWithDefaultOptions allocator;

    const char *model_path = "onnx.onnx";

    Ort::Session session(env, model_path, session_options);

    // 输入节点
    const char *modal_input_names[] = {session.GetInputName(0, allocator)};
    // 输出节点
    const char *modal_output_names[] = {session.GetOutputName(0, allocator)};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 模型输入和输出形状
    // 批 × 长 × 宽 × 通道
    std::vector<int64_t> modal_input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> modal_output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    printf("-> 模型输入形态：%ld × %ld × %ld × %ld\n", modal_input_shape[0], modal_input_shape[1], modal_input_shape[2], modal_input_shape[3]);

    char *image_path = "img.jpg";

    // 读取图片
    cv::Mat input_image = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);

    printf("-> 图片大小形态：%d × %d × %d\n", input_image.rows, input_image.cols, input_image.channels());

    cv::resize(input_image, input_image, cv::Size((int)modal_input_shape.at(1), (int)modal_input_shape.at(2)), cv::InterpolationFlags::INTER_CUBIC);

    // OpenCV中颜色空间转换
    cv::cvtColor(input_image, input_image, cv::ColorConversionCodes::COLOR_BGR2RGB);

    // 转换成浮点数，并且归一化
    input_image.convertTo(input_image, CV_32F, 1.0 / 255.0, -1);

    std::vector<int64_t> input_tensor_dims = {1, input_image.rows, input_image.cols, input_image.channels()};

    // 输入张量
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.ptr<float>(), input_image.total() * input_image.channels(), input_tensor_dims.data(), input_tensor_dims.size());

    // 模型推理计算
    vector<Ort::Value> output_value = session.Run(Ort::RunOptions{nullptr}, modal_input_names, &input_tensor, input_tensor_dims.at(0), modal_output_names, input_tensor_dims.at(0));

    for (int i = 0; i < input_tensor_dims.at(0); i++){
        float *out_arr = output_value[i].GetTensorMutableData<float>();
        printf("-> 模型预测标签：%ld\n", max_element(out_arr, out_arr + modal_output_shape.at(1)) - out_arr);
    }

    printf("-> 用时：%.3f秒\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);
    
    return 0;
}
