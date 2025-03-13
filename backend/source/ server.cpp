#include "crow.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <memory>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using json = nlohmann::json;

#define UPLOAD_DIR "./resources/preprocessed/" // Directory to store uploaded images

// Cloudinary credentials
#define CLOUD_NAME "dd0ogn2qg"
#define API_KEY "782955842683427"
#define API_SECRET "YZG1IJY7BJ7InxCD9LZQ7w205gU"

// Callback function to capture response
size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

// Function to upload image
void uploadImage(const string &imagePath)
{
    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl)
    {
        string response;
        string apiUrl = "https://api.cloudinary.com/v1_1/" + string(CLOUD_NAME) + "/image/upload";
        string apiKey = "api_key=" + string(API_KEY);
        string uploadPreset = "upload_preset=unsigned"; // Change if using presets
        string fileField = "file=@" + imagePath;

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: multipart/form-data");

        curl_easy_setopt(curl, CURLOPT_URL, apiUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);

        curl_mime *mime;
        curl_mimepart *part;
        mime = curl_mime_init(curl);

        // File part
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "stored_thyroid.dat");
        curl_mime_filedata(part, imagePath.c_str());

        // API Key part
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "api_key");
        curl_mime_data(part, API_KEY, CURL_ZERO_TERMINATED);

        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            cerr << "Failed to upload: " << curl_easy_strerror(res) << endl;
        }
        else
        {
            cout << "Upload Successful! Response: " << response << endl;
        }

        curl_easy_cleanup(curl);
        curl_mime_free(mime);
        curl_global_cleanup();
    }
}

int upload_to_cloudinary()
{
    string imagePath = "thyroid_ultrasound.jpg";
    uploadImage(imagePath);
    return 0;
}

int final_results()
{
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
    if (image.empty())
    {
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

    for (int i = 0; i < 3; i++)
    {
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
        input_shape.size());

    // Run the ONNX model
    std::vector<const char *> input_names = {input_name.get()};
    std::vector<const char *> output_names = {output_name.get()};

    std::cout << "Running inference on the model..." << std::endl;

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1);

    std::cout << "Inference completed successfully." << std::endl;

    // Get the output data (predictions)
    float *output_data = output_tensors[0].GetTensorMutableData<float>();

    // Debugging: Print raw output tensor values
    std::cout << "Raw output tensor values: ";
    for (int i = 0; i < 3; i++)
    {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    // Process and print the results
    std::vector<std::string> labels = {"normal thyroid", "malignant", "benign"};
    int predicted_class = std::max_element(output_data, output_data + 3) - output_data;
    std::cout << "Predicted Diagnosis: " << labels[predicted_class] << std::endl;

    return 0;
}

// function to save image as binary
void save_processed_image(const Mat &image, const string &filename)
{
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char *>(image.data), image.total() * image.elemSize());
    file.close();
}

// Function to load an image from a binary file
Mat loadImage(const string &filename, int rows, int cols, int type)
{
    ifstream file(filename, ios::binary);
    vector<uchar> buffer(istreambuf_iterator<char>(file), {});
    Mat image(rows, cols, type, buffer.data());
    return image.clone();
}

int image_storage()
{
    // Load ultrasound image
    string imagePath = "thyroid_ultrasound.jpg";
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cerr << "Error: Unable to load image!" << endl;
        return -1;
    }

    // Save image to binary file
    string filename = "stored_thyroid.dat";
    saveImage(image, filename);
    cout << "Image stored successfully as " << filename << endl;

    // Load image from file
    Mat loadedImage = loadImage(filename, image.rows, image.cols, image.type());
    imwrite("restored_thyroid.jpg", loadedImage);

    cout << "Image loaded and restored successfully!" << endl;

    return 0;
}

// const unsigned char AES_USER_KEY[16] = "1234567890abcdef"; // Fixed variable name
// const unsigned char AES_IV[16] = "1234567890abcdef";      // Initialization Vector (IV)
const unsigned char AES_USER_KEY[16] = {
    '1', '2', '3', '4', '5', '6', '7', '8',
    '9', '0', 'a', 'b', 'c', 'd', 'e', 'f'};

const unsigned char AES_IV[16] = {
    'f', 'e', 'd', 'c', 'b', 'a', '0', '9',
    '8', '7', '6', '5', '4', '3', '2', '1'};

// Function to encrypt data
void encryptImage(const std::vector<unsigned char> &inputData, std::vector<unsigned char> &encryptedData)
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len;
    int ciphertext_len;

    encryptedData.resize(inputData.size() + 16); // Extra space for padding

    EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, AES_USER_KEY, AES_IV);
    EVP_EncryptUpdate(ctx, encryptedData.data(), &len, inputData.data(), inputData.size());
    ciphertext_len = len;
    EVP_EncryptFinal_ex(ctx, encryptedData.data() + len, &len);
    ciphertext_len += len;

    encryptedData.resize(ciphertext_len); // Trim excess space
    EVP_CIPHER_CTX_free(ctx);
}

// Function to decrypt data
void decryptImage(const std::vector<unsigned char> &encryptedData, std::vector<unsigned char> &decryptedData)
{
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len;
    int plaintext_len;

    decryptedData.resize(encryptedData.size());

    EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, AES_USER_KEY, AES_IV);
    EVP_DecryptUpdate(ctx, decryptedData.data(), &len, encryptedData.data(), encryptedData.size());
    plaintext_len = len;
    EVP_DecryptFinal_ex(ctx, decryptedData.data() + len, &len);
    plaintext_len += len;

    decryptedData.resize(plaintext_len); // Trim excess space
    EVP_CIPHER_CTX_free(ctx);
}

// Function to read image file
std::vector<unsigned char> readImageFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Function to write encrypted/decrypted file
void writeImageFile(const std::string &filename, const std::vector<unsigned char> &data)
{
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
}

int image_encryption()
{
    std::string inputFilename = "thyroid_ultrasound.jpg";
    std::string encryptedFilename = "encrypted_thyroid.dat";
    std::string decryptedFilename = "decrypted_thyroid.jpg";

    // Read the input image
    std::vector<unsigned char> imageData = readImageFile(inputFilename);
    std::vector<unsigned char> encryptedData, decryptedData;

    // Encrypt and save
    encryptImage(imageData, encryptedData);
    writeImageFile(encryptedFilename, encryptedData);
    std::cout << "Image encrypted and saved as: " << encryptedFilename << std::endl;

    // Decrypt and save
    decryptImage(encryptedData, decryptedData);
    writeImageFile(decryptedFilename, decryptedData);
    std::cout << "Image decrypted and saved as: " << decryptedFilename << std::endl;

    return 0;
}

// Function to compute entropy
double computeEntropy(const Mat &grayImage)
{
    vector<int> histogram(256, 0);
    for (int i = 0; i < grayImage.rows; i++)
    {
        for (int j = 0; j < grayImage.cols; j++)
        {
            histogram[(int)grayImage.at<uchar>(i, j)]++;
        }
    }

    double entropy = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;
    for (int i = 0; i < 256; i++)
    {
        if (histogram[i] > 0)
        {
            double p = (double)histogram[i] / totalPixels;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

// Function to compute contrast
double computeContrast(const Mat &grayImage)
{
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);
    return stddev[0] * stddev[0]; // Contrast is variance
}

// Function to compute correlation
double computeCorrelation(const Mat &grayImage)
{
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    double mean = 0.0, variance = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;

    // Calculate mean
    for (int i = 0; i < 256; i++)
    {
        mean += i * hist.at<float>(i);
    }
    mean /= totalPixels;

    // Calculate variance
    for (int i = 0; i < 256; i++)
    {
        variance += pow(i - mean, 2) * hist.at<float>(i);
    }
    variance /= totalPixels;

    return sqrt(variance) / mean;
}

// Function to compute energy
double computeEnergy(const Mat &grayImage)
{
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    double energy = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;

    for (int i = 0; i < histSize; i++)
    {
        double p = hist.at<float>(i) / totalPixels;
        energy += p * p;
    }
    return energy;
}

// Function to compute sharpness using Laplacian variance
double computeSharpness(const Mat &grayImage)
{
    Mat laplacian;
    Laplacian(grayImage, laplacian, CV_64F);
    Scalar mean, stddev;
    meanStdDev(laplacian, mean, stddev);
    return stddev[0] * stddev[0];
}

// Function to compute noise level
double computeNoiseLevel(const Mat &grayImage)
{
    Mat blurred;
    GaussianBlur(grayImage, blurred, Size(7, 7), 0);
    Mat noise;
    absdiff(grayImage, blurred, noise);
    Scalar mean, stddev;
    meanStdDev(noise, mean, stddev);
    return stddev[0];
}

// Function to compute resolution
Size computeResolution(const Mat &image)
{
    return image.size();
}

// Function for image segmentation
Mat segmentImage(const Mat &grayImage)
{
    Mat binaryImage;
    // Apply Otsu's thresholding
    double thresh = threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Apply additional preprocessing if needed
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, kernel);

    return binaryImage;
}

// Add new function to compute skewness
double computeSkewness(const Mat &grayImage)
{
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);

    double sum = 0.0;
    double pixelCount = grayImage.rows * grayImage.cols;

    for (int i = 0; i < grayImage.rows; i++)
    {
        for (int j = 0; j < grayImage.cols; j++)
        {
            double diff = grayImage.at<uchar>(i, j) - mean[0];
            sum += pow(diff, 3);
        }
    }

    return sum / (pixelCount * pow(stddev[0], 3));
}

// Enhanced object detection with additional metrics
struct ObjectMetrics
{
    double area;
    double perimeter;
    Point2f centroid;
    Rect boundingBox;
};

vector<ObjectMetrics> detectObjectsEnhanced(const Mat &binaryImage)
{
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<ObjectMetrics> objects;

    for (const auto &contour : contours)
    {
        ObjectMetrics metrics;

        // Compute area
        metrics.area = contourArea(contour);

        // Compute perimeter
        metrics.perimeter = arcLength(contour, true);

        // Compute centroid
        Moments m = moments(contour);
        metrics.centroid = Point2f(m.m10 / m.m00, m.m01 / m.m00);

        // Get bounding box
        metrics.boundingBox = boundingRect(contour);

        objects.push_back(metrics);
    }

    return objects;
}

// Function to perform and analyze morphological operations
struct MorphologicalMetrics
{
    Mat dilated;
    Mat eroded;
    Mat opened;
    Mat closed;
    double areaChange;
};

MorphologicalMetrics performMorphologicalAnalysis(const Mat &binaryImage)
{
    MorphologicalMetrics metrics;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    // Perform operations
    dilate(binaryImage, metrics.dilated, kernel);
    erode(binaryImage, metrics.eroded, kernel);

    // Opening (erosion followed by dilation)
    erode(binaryImage, metrics.opened, kernel);
    dilate(metrics.opened, metrics.opened, kernel);

    // Closing (dilation followed by erosion)
    dilate(binaryImage, metrics.closed, kernel);
    erode(metrics.closed, metrics.closed, kernel);

    // Calculate area change after morphological operations
    double originalArea = countNonZero(binaryImage);
    double dilatedArea = countNonZero(metrics.dilated);
    metrics.areaChange = ((dilatedArea - originalArea) / originalArea) * 100;

    return metrics;
}

// Function to compute comprehensive statistical metrics
struct StatisticalMetrics
{
    double mean;
    double variance;
    double stdDev;
    double skewness;
    double kurtosis;
};

StatisticalMetrics computeStatistics(const Mat &grayImage)
{
    StatisticalMetrics stats;
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);

    stats.mean = mean[0];
    stats.stdDev = stddev[0];
    stats.variance = stddev[0] * stddev[0];
    stats.skewness = computeSkewness(grayImage);

    // Compute kurtosis
    double sum = 0.0;
    double pixelCount = grayImage.rows * grayImage.cols;

    for (int i = 0; i < grayImage.rows; i++)
    {
        for (int j = 0; j < grayImage.cols; j++)
        {
            double diff = grayImage.at<uchar>(i, j) - mean[0];
            sum += pow(diff, 4);
        }
    }

    stats.kurtosis = (sum / (pixelCount * pow(stddev[0], 4))) - 3.0;

    return stats;
}

int image_quantification()
{
    // Load the ultrasound image
    Mat image = imread("thyroid_ultrasound.jpg", IMREAD_GRAYSCALE);
    if (image.empty())
    {
        cerr << "Error: Could not load the image!" << endl;
        return -1;
    }

    // Segment the image
    Mat segmentedImage = segmentImage(image);

    // Get enhanced object metrics
    vector<ObjectMetrics> objectMetrics = detectObjectsEnhanced(segmentedImage);

    // Perform morphological analysis
    MorphologicalMetrics morphMetrics = performMorphologicalAnalysis(segmentedImage);

    // Compute statistical metrics
    StatisticalMetrics stats = computeStatistics(image);

    // Original metrics
    double entropy = computeEntropy(image);
    double contrast = computeContrast(image);
    double correlation = computeCorrelation(image);
    double energy = computeEnergy(image);
    double sharpness = computeSharpness(image);
    double noiseLevel = computeNoiseLevel(image);
    Size resolution = computeResolution(image);

    // Display enhanced results
    cout << "===== Enhanced Image Quantification Metrics =====" << endl;
    cout << "\nBasic Metrics:" << endl;
    cout << "Entropy: " << entropy << endl;
    cout << "Contrast: " << contrast << endl;
    cout << "Correlation: " << correlation << endl;
    cout << "Energy: " << energy << endl;

    cout << "\nImage Quality Metrics:" << endl;
    cout << "Sharpness: " << sharpness << endl;
    cout << "Noise Level: " << noiseLevel << endl;
    cout << "Resolution: " << resolution.width << " x " << resolution.height << " pixels" << endl;

    cout << "\nStatistical Metrics:" << endl;
    cout << "Mean: " << stats.mean << endl;
    cout << "Variance: " << stats.variance << endl;
    cout << "Standard Deviation: " << stats.stdDev << endl;
    cout << "Skewness: " << stats.skewness << endl;
    cout << "Kurtosis: " << stats.kurtosis << endl;

    cout << "\nSegmentation Metrics:" << endl;
    cout << "Number of Objects: " << objectMetrics.size() << endl;
    double totalArea = 0, totalPerimeter = 0;
    for (size_t i = 0; i < objectMetrics.size(); i++)
    {
        cout << "Object " << i + 1 << ":" << endl;
        cout << "  Area: " << objectMetrics[i].area << " pixels" << endl;
        cout << "  Perimeter: " << objectMetrics[i].perimeter << " pixels" << endl;
        cout << "  Centroid: (" << objectMetrics[i].centroid.x << ", "
             << objectMetrics[i].centroid.y << ")" << endl;
        totalArea += objectMetrics[i].area;
        totalPerimeter += objectMetrics[i].perimeter;
    }
    cout << "Total Area: " << totalArea << " pixels" << endl;
    cout << "Total Perimeter: " << totalPerimeter << " pixels" << endl;

    cout << "\nMorphological Analysis:" << endl;
    cout << "Area Change After Dilation: " << morphMetrics.areaChange << "%" << endl;

    // Display images
    imshow("Original Image", image);
    imshow("Segmented Image", segmentedImage);
    imshow("Dilated Image", morphMetrics.dilated);
    imshow("Eroded Image", morphMetrics.eroded);
    imshow("Opened Image", morphMetrics.opened);
    imshow("Closed Image", morphMetrics.closed);
    waitKey(0);

    return 0;
}

// Callback function to write API response
size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *userp)
{
    userp->append((char *)contents, size * nmemb);
    return size * nmemb;
}

// Function to encode image to base64
std::string encodeBase64(const std::vector<char> &buffer)
{
    static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string encoded;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    for (size_t idx = 0; idx < buffer.size(); ++idx)
    {
        char_array_3[i++] = buffer[idx];
        if (i == 3)
        {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++)
                encoded += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++)
            encoded += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            encoded += '=';
    }

    return encoded;
}

class ClaudeMedicalImageAnalyzer
{
private:
    std::string api_key;
    std::string model;

public:
    ClaudeMedicalImageAnalyzer(const std::string &key, const std::string &model_name = "claude-3-7-sonnet-20250219")
        : api_key(key), model(model_name) {}

    std::string analyzeMedicalImage(const std::string &imagePath)
    {
        // Read image file as binary
        std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            return "Error: Could not open image file";
        }

        // Get file size and read file
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
        {
            return "Error: Could not read image file";
        }

        // Get image file extension
        std::string extension = imagePath.substr(imagePath.find_last_of(".") + 1);
        std::string mimeType;
        if (extension == "jpg" || extension == "jpeg")
        {
            mimeType = "image/jpeg";
        }
        else if (extension == "png")
        {
            mimeType = "image/png";
        }
        else if (extension == "gif")
        {
            mimeType = "image/gif";
        }
        else if (extension == "dicom" || extension == "dcm")
        {
            mimeType = "application/dicom";
        }
        else
        {
            return "Error: Unsupported image format. Supported formats include JPEG, PNG, GIF, and DICOM.";
        }

        // Encode image to base64
        std::string base64Image = encodeBase64(buffer);

        // Create request payload
        json contentArray = json::array();

        // Add text content
        contentArray.push_back({{"type", "text"},
                                {"text", "Please analyze this image and identify any potential abnormalities or areas of concern as detailed as possible."}});

        // Add image content
        contentArray.push_back({{"type", "image"},
                                {"source", {{"type", "base64"}, {"media_type", mimeType}, {"data", base64Image}}}});

        json message = {
            {"role", "user"},
            {"content", contentArray}};

        json requestBody = {
            {"model", model},
            {"messages", json::array({message})},
            {"max_tokens", 1024}};

        // Convert JSON to string
        std::string requestBodyStr = requestBody.dump();

        // Print request for debugging
        std::cout << "Sending request to Anthropic API..." << std::endl;

        // Initialize CURL
        CURL *curl = curl_easy_init();
        std::string response;

        if (curl)
        {
            // Set up headers
            struct curl_slist *headers = NULL;
            headers = curl_slist_append(headers, ("x-api-key: " + api_key).c_str());
            headers = curl_slist_append(headers, "Content-Type: application/json");
            headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

            // Set CURL options
            curl_easy_setopt(curl, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestBodyStr.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            // Enable verbose output for debugging
            curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

            // Perform request
            CURLcode res = curl_easy_perform(curl);

            // Clean up
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK)
            {
                return "Error: " + std::string(curl_easy_strerror(res));
            }
        }

        // Print raw response for debugging
        std::cout << "Raw response: " << response << std::endl;

        // Parse JSON response
        try
        {
            json responseJson = json::parse(response);
            if (responseJson.contains("content") && responseJson["content"].size() > 0)
            {
                // Extract text from response
                return responseJson["content"][0]["text"];
            }
            else if (responseJson.contains("error"))
            {
                // Handle error response
                return "API Error: " + responseJson["error"]["message"].get<std::string>();
            }
            else
            {
                // Return full response for debugging
                return "Full response: " + responseJson.dump(2);
            }
        }
        catch (json::exception &e)
        {
            return "Error parsing response: " + std::string(e.what()) + "\nRaw response: " + response;
        }
    }
};

std::string getApiKeyFromEnv()
{
    char *key = std::getenv("ANTHROPIC_API_KEY");
    if (key == nullptr)
    {
        std::cerr << "ERROR: ANTHROPIC_API_KEY environment variable is not set!" << std::endl;
        return "";
    }
    return std::string(key);
}

std::map<std::string, std::string> loadEnvFile(const std::string &filePath = "../.env")
{
    std::map<std::string, std::string> envVars;
    std::ifstream envFile(filePath);

    if (envFile.is_open())
    {
        std::string line;
        while (std::getline(envFile, line))
        {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            // Find the equals sign
            size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // Remove any trailing whitespace
                key.erase(key.find_last_not_of(" \t") + 1);

                // Store in our map
                envVars[key] = value;
            }
        }
        envFile.close();
    }
    else
    {
        std::cerr << "Warning: Could not open .env file at " << filePath << std::endl;
    }

    return envVars;
}

int image_analyser()
{
    auto envVars = loadEnvFile();

    // Get API key (with fallback to system environment variable)
    std::string api_key;
    if (envVars.count("ANTHROPIC_API_KEY") > 0)
    {
        api_key = envVars["ANTHROPIC_API_KEY"];
    }
    else
    {
        // Try system environment variable as fallback
        char *key = std::getenv("ANTHROPIC_API_KEY");
        if (key != nullptr)
        {
            api_key = std::string(key);
        }
    }

    if (api_key.empty())
    {
        std::cerr << "Error: ANTHROPIC_API_KEY not found in .env file or environment variables" << std::endl;
        return 1;
    }

    // Create analyzer with the API key
    ClaudeMedicalImageAnalyzer analyzer(api_key);

    // Use the image in the project root directory
    std::string imagePath = "../resources/mri_1.jpeg";

    std::cout << "Analyzing the medical image with Claude 3.7 Sonnet: " << imagePath << std::endl;
    std::string result = analyzer.analyzeMedicalImage(imagePath);
    std::cout << "\nAnalysis Result:\n"
              << result << std::endl;

    return 0;
}

// Function to save uploaded file
bool save_preprocessed_file(const std::string &filename, const std::vector<uint8_t> &data)
{
    std::ofstream file(UPLOAD_DIR + filename, std::ios::binary);
    if (!file)
        return false;
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
    return true;
}

// function to gralum image
int processImage()
{
    Image test(argv[3]);
    Image gray_lum = test;
    gray_lum.grayscale_lum();
    gray_lum.write(argv[2]); // Could also convert using std::string(argv[2]).c_str()

    return 0;
}

int main()
{
    crow::SimpleApp app;

    // Home route
    CROW_ROUTE(app, "/")([]()
                         { return "Hello, Clang Docker!"; });

    // Image upload route
    CROW_ROUTE(app, "/upload").methods(crow::HTTPMethod::Post)([](const crow::request &req)
                                                               {
        auto headers = req.headers;
        
        // Check for file content
        if (req.body.empty()) {
            return crow::response(400, "No file uploaded");
        }

        std::string filePath = "uploaded_image.jpg";

        // Save image to disk
        std::ofstream file(filePath, std::ios::binary);
        file.write(req.body.c_str(), req.body.size());
        file.close();

        // Run image analysis
        std::string analysisResult = processImage(filePath);

        // Return JSON response
        crow::json::wvalue response;
        response["message"] = "Image uploaded and analyzed";
        response["analysis"] = analysisResult;

        return crow::response(200, response); });

    app.port(8080).multithreaded().run();
}
