// ====================================================================
//                           INCLUDES
// ====================================================================
#include "crow.h"
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <memory>
#include <filesystem>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using json = nlohmann::json;

// ====================================================================
//                   GLOBAL CONSTANTS & MACROS
// ====================================================================
#define UPLOAD_DIR "./resources/preprocessed/" // Directory to store preprocessed images
#define PROCESSED_DIR "./resources/processed/" // Directory to store processed images

// ====================================================================
//                    ONNX MODEL ANALYSIS (FIXED)
// ====================================================================

int final_results()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ThyroidClassification");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Model path set relative to project root.
    const string model_path = "../resources/model/thyroid_ultrasound_model.onnx";

    // ✅ Fix: Use string instead of wstring
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    unique_ptr<char, Ort::detail::AllocatedFree> input_name = session.GetInputNameAllocated(0, allocator);
    unique_ptr<char, Ort::detail::AllocatedFree> output_name = session.GetOutputNameAllocated(0, allocator);

    cout << "Input Tensor Name: " << input_name.get() << endl;
    cout << "Output Tensor Name: " << output_name.get() << endl;

    // ✅ Fix: Use OpenCV instead of undefined Image class
    string image_path;
    cout << "Enter the path to the image: ";
    getline(cin, image_path);

    Mat image = imread(image_path);
    if (image.empty())
    {
        cerr << "Error: Unable to load image at " << image_path << endl;
        return -1;
    }
    resize(image, image, Size(224, 224));
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    cvtColor(image, image, COLOR_BGR2RGB);

    vector<float> input_tensor_values(3 * 224 * 224);
    vector<Mat> channels(3);
    split(image, channels);
    for (int i = 0; i < 3; i++)
        memcpy(input_tensor_values.data() + i * 224 * 224, channels[i].data, 224 * 224 * sizeof(float));

    vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    vector<const char *> input_names = {input_name.get()};
    vector<const char *> output_names = {output_name.get()};

    cout << "Running inference on the model..." << endl;
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names.data(),
                                      &input_tensor,
                                      1,
                                      output_names.data(),
                                      1);
    cout << "Inference completed successfully." << endl;

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    cout << "Raw output tensor values: ";
    for (int i = 0; i < 3; i++)
        cout << output_data[i] << " ";
    cout << endl;

    vector<string> labels = {"normal thyroid", "malignant", "benign"};
    int predicted_class = max_element(output_data, output_data + 3) - output_data;
    cout << "Predicted Diagnosis: " << labels[predicted_class] << endl;

    return 0;
}

// ====================================================================
//                  IMAGE PROCESSING FUNCTION (FIXED)
// ====================================================================
int processImage(string image_path)
{
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        cerr << "Error: Unable to load image: " << image_path << endl;
        return -1;
    }

    Mat gray_img, blur_img, edges;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    GaussianBlur(gray_img, blur_img, Size(5, 5), 0);
    Canny(blur_img, edges, 50, 150);

    imwrite(string(PROCESSED_DIR) + "gray.png", gray_img);
    imwrite(string(PROCESSED_DIR) + "blur.png", blur_img);
    imwrite(string(PROCESSED_DIR) + "edges.png", edges);

    return 0;
}

// ====================================================================
//                  CROW WEB SERVER & ROUTES
// ====================================================================

int main() {
    crow::SimpleApp app;

    // CORS Middleware: Allow cross-origin requests from the specified origin
    app.before([](const crow::request &req) {
        crow::response res;
        
        // Allow all origins (for development purposes), change this to "http://localhost:3000" in production for security
        res.add_header("Access-Control-Allow-Origin", "*");  // Change "*" to "http://localhost:3000" in production for more security
        
        // Allow all methods that you are using (GET, POST, OPTIONS, etc.)
        res.add_header("Access-Control-Allow-Methods", "*");
        
        // Allow specific headers (you can adjust this based on your needs)
        res.add_header("Access-Control-Allow-Headers", "*");

        // Handle preflight OPTIONS requests (this is necessary for CORS to work)
        if (req.method == crow::HTTPMethod::Options) {
            return crow::response(200); // Respond with 200 OK for OPTIONS preflight requests
        }

        return crow::response(); // Continue with the request processing
    });

    // Upload route
    CROW_ROUTE(app, "/upload").methods(crow::HTTPMethod::Post)([](const crow::request &req) {
        if (!req.has_file("file")) {
            return crow::response(400, "No image uploaded");
        }

        // Get the file from the form data (field name: "file")
        std::string fileName = "uploaded_image.jpg"; // Default filename
        const auto& file = req.get_file("file");
        fileName = file.filename();

        // Define the upload path dynamically using the uploaded file's name
        std::string uploadPath = "uploads/" + fileName;

        // Write the file to the server
        std::ofstream fileStream(uploadPath, std::ios::binary);
        if (!fileStream.is_open()) {
            return crow::response(500, "Failed to save the uploaded file");
        }

        fileStream.write(req.body.c_str(), req.body.size());
        fileStream.close();

        // Call image processing and model inference
        int processStatus = processImage(uploadPath);
        if (processStatus != 0) {
            return crow::response(500, "Image processing failed");
        }

        int finalStatus = final_results();
        if (finalStatus != 0) {
            return crow::response(500, "Model inference failed");
        }

        // Respond with a success message
        crow::json::wvalue response;
        response["message"] = "Image processed and analyzed successfully";
        return crow::response(200, response);
    });

    // Run the server on port 8080
    app.port(8080).multithreaded().run();
}