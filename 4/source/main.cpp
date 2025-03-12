#include <onnxruntime_cxx_api.h>
#include "opencv2/opencv.hpp" // OpenCV for image preprocessing
#include <iostream>
#include <vector>
#include <string>
#include <filesystem> // Ensure correct path handling
#include <memory>

int main() {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ThyroidClassification");

    // Set up ONNX Runtime session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load the ONNX model
    const std::string model_path = "C:\\Users\\USER\\Documents\\MyPythonProjects\\ThyroidClassification\\thyroid_ultrasound_model.onnx";
    std::wstring model_path_w = std::filesystem::path(model_path).wstring();
    
    // Create ONNX Runtime session
    Ort::Session session(env, model_path_w.c_str(), session_options);

    // Allocator for input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    // Use GetInputNameAllocated() instead of GetInputName()
    std::unique_ptr<char, Ort::detail::AllocatedFree> input_name = session.GetInputNameAllocated(0, allocator);
    std::unique_ptr<char, Ort::detail::AllocatedFree> output_name = session.GetOutputNameAllocated(0, allocator);

    std::cout << "Input Tensor Name: " << input_name.get() << std::endl;
    std::cout << "Output Tensor Name: " << output_name.get() << std::endl;

    // Ask the user to input the image path
    std::string image_path;
    std::cout << "Enter the path to the image: ";
    std::getline(std::cin, image_path);

    // Load and preprocess the image using OpenCV
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image at " << image_path << std::endl;
        return -1;
    }

    // Resize the image to 224x224 (required by the model)
    cv::resize(image, image, cv::Size(224, 224));

    // Convert the image to a float tensor (Normalize to [0,1])
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Convert BGR to RGB

    // Convert OpenCV image from HWC format to CHW format
    std::vector<float> input_tensor_values(3 * 224 * 224);
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    for (int i = 0; i < 3; i++) {
        std::memcpy(input_tensor_values.data() + i * 224 * 224,
                    channels[i].data, 224 * 224 * sizeof(float));
    }

    // Create a 4D input tensor [1, 3, 224, 224]
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run the ONNX model
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};
    
    std::cout << "Running inference on the model..." << std::endl;
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );

    std::cout << "Inference completed successfully." << std::endl;

    // Get the output data (predictions)
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    // Debugging: Print raw output tensor values
    std::cout << "Raw output tensor values: ";
    for (int i = 0; i < 3; i++) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    // Process and print the results
    std::vector<std::string> labels = {"normal thyroid", "malignant", "benign"};
    int predicted_class = std::max_element(output_data, output_data + 3) - output_data;
    std::cout << "Predicted Diagnosis: " << labels[predicted_class] << std::endl;

    return 0;
}
