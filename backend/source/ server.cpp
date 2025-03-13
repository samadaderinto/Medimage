// ====================================================================
//                           INCLUDES
// ====================================================================
#include "crow.h"
#include <fstream>
#include <vector>
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
//                        GLOBAL CONSTANTS & MACROS
// ====================================================================
#define UPLOAD_DIR "./resources/preprocessed/"  // Directory to store uploaded images

// ====================================================================
//                  ENVIRONMENT & UTILITY FUNCTIONS
// ====================================================================

// Loads environment variables from a file (default path: "../.env")
map<string, string> loadEnvFile(const string &filePath = "../.env") {
    map<string, string> envVars;
    ifstream envFile(filePath);
    if (envFile.is_open()) {
        string line;
        while (getline(envFile, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            size_t pos = line.find('=');
            if (pos != string::npos) {
                string key = line.substr(0, pos);
                string value = line.substr(pos + 1);
                key.erase(key.find_last_not_of(" \t") + 1);
                envVars[key] = value;
            }
        }
        envFile.close();
    } else {
        cerr << "Warning: Could not open .env file at " << filePath << endl;
    }
    return envVars;
}

// Callback function used by libcurl to capture responses
size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

// Encodes a binary buffer to a Base64 string
string encodeBase64(const vector<char> &buffer) {
    static const string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    string encoded;
    int i = 0, j = 0;
    unsigned char char_array_3[3], char_array_4[4];

    for (size_t idx = 0; idx < buffer.size(); ++idx) {
        char_array_3[i++] = buffer[idx];
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; i < 4; i++)
                encoded += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
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

// ====================================================================
//                      CLOUDINARY UPLOAD FUNCTIONS
// ====================================================================

// Uploads an image to Cloudinary using libcurl; Cloudinary credentials are loaded from .env.
void uploadImage(const string &imagePath) {
    map<string, string> envVars = loadEnvFile();
    string cloudName = envVars["CLOUDINARY_CLOUD_NAME"];
    string apiKey    = envVars["CLOUDINARY_API_KEY"];
    // string apiSecret = envVars["CLOUDINARY_API_SECRET"]; // Uncomment if needed

    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        string response;
        string apiUrl = "https://api.cloudinary.com/v1_1/" + cloudName + "/image/upload";
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: multipart/form-data");
        curl_easy_setopt(curl, CURLOPT_URL, apiUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);

        curl_mime *mime = curl_mime_init(curl);
        curl_mimepart *part;

        // File part
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "file");
        curl_mime_filedata(part, imagePath.c_str());

        // API Key part
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "api_key");
        curl_mime_data(part, apiKey.c_str(), CURL_ZERO_TERMINATED);

        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            cerr << "Failed to upload: " << curl_easy_strerror(res) << endl;
        else
            cout << "Upload Successful! Response: " << response << endl;

        curl_easy_cleanup(curl);
        curl_mime_free(mime);
        curl_global_cleanup();
    }
}

int upload_to_cloudinary() {
    string imagePath = "thyroid_ultrasound.jpg";
    uploadImage(imagePath);
    return 0;
}

// ====================================================================
//                       ONNX MODEL ANALYSIS
// ====================================================================

int final_results() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ThyroidClassification");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const string model_path = "C:\\Users\\USER\\Documents\\MyPythonProjects\\ThyroidClassification\\thyroid_ultrasound_model.onnx";
    wstring model_path_w = filesystem::path(model_path).wstring();

    Ort::Session session(env, model_path_w.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    unique_ptr<char, Ort::detail::AllocatedFree> input_name = session.GetInputNameAllocated(0, allocator);
    unique_ptr<char, Ort::detail::AllocatedFree> output_name = session.GetOutputNameAllocated(0, allocator);

    cout << "Input Tensor Name: " << input_name.get() << endl;
    cout << "Output Tensor Name: " << output_name.get() << endl;

    string image_path;
    cout << "Enter the path to the image: ";
    getline(cin, image_path);

    Mat image = imread(image_path);
    if (image.empty()) {
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
//             IMAGE STORAGE & ENCRYPTION FUNCTIONS
// ====================================================================

// Save an image as a binary file
void save_processed_image(const Mat &image, const string &filename) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char *>(image.data), image.total() * image.elemSize());
    file.close();
}

// Load an image from a binary file
Mat loadImage(const string &filename, int rows, int cols, int type) {
    ifstream file(filename, ios::binary);
    vector<uchar> buffer((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    Mat image(rows, cols, type, buffer.data());
    return image.clone();
}

// Read an image file into a vector of unsigned char
vector<unsigned char> readImageFile(const string &filename) {
    ifstream file(filename, ios::binary);
    return vector<unsigned char>((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

// Write a vector of unsigned char to an image file
void writeImageFile(const string &filename, const vector<unsigned char> &data) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
}

// OpenSSL AES keys and IV (hardcoded for now)
const unsigned char AES_USER_KEY[16] = {
    '1','2','3','4','5','6','7','8',
    '9','0','a','b','c','d','e','f'
};
const unsigned char AES_IV[16] = {
    'f','e','d','c','b','a','0','9',
    '8','7','6','5','4','3','2','1'
};

// Encrypt data using AES-128-CBC
void encryptImage(const vector<unsigned char> &inputData, vector<unsigned char> &encryptedData) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len, ciphertext_len;
    encryptedData.resize(inputData.size() + 16); // Allocate extra space for padding
    EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, AES_USER_KEY, AES_IV);
    EVP_EncryptUpdate(ctx, encryptedData.data(), &len, inputData.data(), inputData.size());
    ciphertext_len = len;
    EVP_EncryptFinal_ex(ctx, encryptedData.data() + len, &len);
    ciphertext_len += len;
    encryptedData.resize(ciphertext_len);
    EVP_CIPHER_CTX_free(ctx);
}

// Decrypt data using AES-128-CBC
void decryptImage(const vector<unsigned char> &encryptedData, vector<unsigned char> &decryptedData) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len, plaintext_len;
    decryptedData.resize(encryptedData.size());
    EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, AES_USER_KEY, AES_IV);
    EVP_DecryptUpdate(ctx, decryptedData.data(), &len, encryptedData.data(), encryptedData.size());
    plaintext_len = len;
    EVP_DecryptFinal_ex(ctx, decryptedData.data() + len, &len);
    plaintext_len += len;
    decryptedData.resize(plaintext_len);
    EVP_CIPHER_CTX_free(ctx);
}

int image_encryption() {
    string inputFilename = "thyroid_ultrasound.jpg";
    string encryptedFilename = "encrypted_thyroid.dat";
    string decryptedFilename = "decrypted_thyroid.jpg";
    vector<unsigned char> imageData = readImageFile(inputFilename);
    vector<unsigned char> encryptedData, decryptedData;
    encryptImage(imageData, encryptedData);
    writeImageFile(encryptedFilename, encryptedData);
    cout << "Image encrypted and saved as: " << encryptedFilename << endl;
    decryptImage(encryptedData, decryptedData);
    writeImageFile(decryptedFilename, decryptedData);
    cout << "Image decrypted and saved as: " << decryptedFilename << endl;
    return 0;
}

int image_storage() {
    string imagePath = "thyroid_ultrasound.jpg";
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Unable to load image!" << endl;
        return -1;
    }
    string filename = "stored_thyroid.dat";
    save_processed_image(image, filename);
    cout << "Image stored successfully as " << filename << endl;
    Mat loadedImage = loadImage(filename, image.rows, image.cols, image.type());
    imwrite("restored_thyroid.jpg", loadedImage);
    cout << "Image loaded and restored successfully!" << endl;
    return 0;
}

// ====================================================================
//                IMAGE QUANTIFICATION & ANALYSIS
// ====================================================================

// Compute basic entropy of a grayscale image
double computeEntropy(const Mat &grayImage) {
    vector<int> histogram(256, 0);
    for (int i = 0; i < grayImage.rows; i++)
        for (int j = 0; j < grayImage.cols; j++)
            histogram[(int)grayImage.at<uchar>(i, j)]++;
    double entropy = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            double p = (double)histogram[i] / totalPixels;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

double computeContrast(const Mat &grayImage) {
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);
    return stddev[0] * stddev[0];
}

double computeCorrelation(const Mat &grayImage) {
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    double mean = 0.0, variance = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;
    for (int i = 0; i < 256; i++)
        mean += i * hist.at<float>(i);
    mean /= totalPixels;
    for (int i = 0; i < 256; i++)
        variance += pow(i - mean, 2) * hist.at<float>(i);
    variance /= totalPixels;
    return sqrt(variance) / mean;
}

double computeEnergy(const Mat &grayImage) {
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    double energy = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;
    for (int i = 0; i < histSize; i++) {
        double p = hist.at<float>(i) / totalPixels;
        energy += p * p;
    }
    return energy;
}

double computeSharpness(const Mat &grayImage) {
    Mat laplacian;
    Laplacian(grayImage, laplacian, CV_64F);
    Scalar mean, stddev;
    meanStdDev(laplacian, mean, stddev);
    return stddev[0] * stddev[0];
}

double computeNoiseLevel(const Mat &grayImage) {
    Mat blurred;
    GaussianBlur(grayImage, blurred, Size(7, 7), 0);
    Mat noise;
    absdiff(grayImage, blurred, noise);
    Scalar mean, stddev;
    meanStdDev(noise, mean, stddev);
    return stddev[0];
}

Size computeResolution(const Mat &image) {
    return image.size();
}

Mat segmentImage(const Mat &grayImage) {
    Mat binaryImage;
    threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, kernel);
    return binaryImage;
}

double computeSkewness(const Mat &grayImage) {
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);
    double sum = 0.0;
    double pixelCount = grayImage.rows * grayImage.cols;
    for (int i = 0; i < grayImage.rows; i++)
        for (int j = 0; j < grayImage.cols; j++) {
            double diff = grayImage.at<uchar>(i, j) - mean[0];
            sum += pow(diff, 3);
        }
    return sum / (pixelCount * pow(stddev[0], 3));
}

// Structure to hold object metrics from segmentation
struct ObjectMetrics {
    double area;
    double perimeter;
    Point2f centroid;
    Rect boundingBox;
};

vector<ObjectMetrics> detectObjectsEnhanced(const Mat &binaryImage) {
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<ObjectMetrics> objects;
    for (const auto &contour : contours) {
        ObjectMetrics metrics;
        metrics.area = contourArea(contour);
        metrics.perimeter = arcLength(contour, true);
        Moments m = moments(contour);
        metrics.centroid = Point2f(m.m10 / m.m00, m.m01 / m.m00);
        metrics.boundingBox = boundingRect(contour);
        objects.push_back(metrics);
    }
    return objects;
}

// Structure to hold morphological analysis results
struct MorphologicalMetrics {
    Mat dilated;
    Mat eroded;
    Mat opened;
    Mat closed;
    double areaChange;
};

MorphologicalMetrics performMorphologicalAnalysis(const Mat &binaryImage) {
    MorphologicalMetrics metrics;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(binaryImage, metrics.dilated, kernel);
    erode(binaryImage, metrics.eroded, kernel);
    erode(binaryImage, metrics.opened, kernel);
    dilate(metrics.opened, metrics.opened, kernel);
    dilate(binaryImage, metrics.closed, kernel);
    erode(metrics.closed, metrics.closed, kernel);
    double originalArea = countNonZero(binaryImage);
    double dilatedArea = countNonZero(metrics.dilated);
    metrics.areaChange = ((dilatedArea - originalArea) / originalArea) * 100;
    return metrics;
}

// Structure for statistical metrics
struct StatisticalMetrics {
    double mean;
    double variance;
    double stdDev;
    double skewness;
    double kurtosis;
};

StatisticalMetrics computeStatistics(const Mat &grayImage) {
    StatisticalMetrics stats;
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);
    stats.mean = mean[0];
    stats.stdDev = stddev[0];
    stats.variance = stddev[0] * stddev[0];
    stats.skewness = computeSkewness(grayImage);
    double sum = 0.0;
    double pixelCount = grayImage.rows * grayImage.cols;
    for (int i = 0; i < grayImage.rows; i++)
        for (int j = 0; j < grayImage.cols; j++) {
            double diff = grayImage.at<uchar>(i, j) - mean[0];
            sum += pow(diff, 4);
        }
    stats.kurtosis = (sum / (pixelCount * pow(stddev[0], 4))) - 3.0;
    return stats;
}

int image_quantification() {
    Mat image = imread("thyroid_ultrasound.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Could not load the image!" << endl;
        return -1;
    }
    Mat segmentedImage = segmentImage(image);
    vector<ObjectMetrics> objectMetrics = detectObjectsEnhanced(segmentedImage);
    MorphologicalMetrics morphMetrics = performMorphologicalAnalysis(segmentedImage);
    StatisticalMetrics stats = computeStatistics(image);
    double entropy = computeEntropy(image);
    double contrast = computeContrast(image);
    double correlation = computeCorrelation(image);
    double energy = computeEnergy(image);
    double sharpness = computeSharpness(image);
    double noiseLevel = computeNoiseLevel(image);
    Size resolution = computeResolution(image);

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
    for (size_t i = 0; i < objectMetrics.size(); i++) {
        cout << "Object " << i + 1 << ":" << endl;
        cout << "  Area: " << objectMetrics[i].area << " pixels" << endl;
        cout << "  Perimeter: " << objectMetrics[i].perimeter << " pixels" << endl;
        cout << "  Centroid: (" << objectMetrics[i].centroid.x << ", " << objectMetrics[i].centroid.y << ")" << endl;
        totalArea += objectMetrics[i].area;
        totalPerimeter += objectMetrics[i].perimeter;
    }
    cout << "Total Area: " << totalArea << " pixels" << endl;
    cout << "Total Perimeter: " << totalPerimeter << " pixels" << endl;
    cout << "\nMorphological Analysis:" << endl;
    cout << "Area Change After Dilation: " << morphMetrics.areaChange << "%" << endl;

    imshow("Original Image", image);
    imshow("Segmented Image", segmentedImage);
    imshow("Dilated Image", morphMetrics.dilated);
    imshow("Eroded Image", morphMetrics.eroded);
    imshow("Opened Image", morphMetrics.opened);
    imshow("Closed Image", morphMetrics.closed);
    waitKey(0);
    return 0;
}

// ====================================================================
//                IMAGE ANALYSIS (CLAUDE MEDICAL ANALYZER)
// ====================================================================
class ClaudeMedicalImageAnalyzer {
private:
    string api_key;
    string model;
public:
    ClaudeMedicalImageAnalyzer(const string &key, const string &model_name = "claude-3-7-sonnet-20250219")
        : api_key(key), model(model_name) {}
    string analyzeMedicalImage(const string &imagePath) {
        ifstream file(imagePath, ios::binary | ios::ate);
        if (!file.is_open())
            return "Error: Could not open image file";
        streamsize size = file.tellg();
        file.seekg(0, ios::beg);
        vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
            return "Error: Could not read image file";
        string extension = imagePath.substr(imagePath.find_last_of(".") + 1);
        string mimeType;
        if (extension == "jpg" || extension == "jpeg")
            mimeType = "image/jpeg";
        else if (extension == "png")
            mimeType = "image/png";
        else if (extension == "gif")
            mimeType = "image/gif";
        else if (extension == "dicom" || extension == "dcm")
            mimeType = "application/dicom";
        else
            return "Error: Unsupported image format. Supported formats include JPEG, PNG, GIF, and DICOM.";

        string base64Image = encodeBase64(buffer);
        json contentArray = json::array();
        contentArray.push_back({{"type", "text"},
                                {"text", "Please analyze this image and identify any potential abnormalities or areas of concern as detailed as possible."}});
        contentArray.push_back({{"type", "image"},
                                {"source", {{"type", "base64"}, {"media_type", mimeType}, {"data", base64Image}}}});
        json message = {{"role", "user"}, {"content", contentArray}};
        json requestBody = {{"model", model},
                            {"messages", json::array({message})},
                            {"max_tokens", 1024}};
        string requestBodyStr = requestBody.dump();
        cout << "Sending request to Anthropic API..." << endl;
        CURL *curl = curl_easy_init();
        string response;
        if (curl) {
            struct curl_slist *headers = NULL;
            headers = curl_slist_append(headers, (string("x-api-key: ") + api_key).c_str());
            headers = curl_slist_append(headers, "Content-Type: application/json");
            headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
            curl_easy_setopt(curl, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestBodyStr.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
            CURLcode res = curl_easy_perform(curl);
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            if (res != CURLE_OK)
                return "Error: " + string(curl_easy_strerror(res));
        }
        cout << "Raw response: " << response << endl;
        try {
            json responseJson = json::parse(response);
            if (responseJson.contains("content") && responseJson["content"].size() > 0)
                return responseJson["content"][0]["text"];
            else if (responseJson.contains("error"))
                return "API Error: " + responseJson["error"]["message"].get<string>();
            else
                return "Full response: " + responseJson.dump(2);
        } catch (json::exception &e) {
            return "Error parsing response: " + string(e.what()) + "\nRaw response: " + response;
        }
    }
};

int image_analyser() {
    map<string, string> envVars = loadEnvFile();
    string api_key;
    if (envVars.count("ANTHROPIC_API_KEY") > 0)
        api_key = envVars["ANTHROPIC_API_KEY"];
    else {
        char *key = getenv("ANTHROPIC_API_KEY");
        if (key != nullptr)
            api_key = string(key);
    }
    if (api_key.empty()) {
        cerr << "Error: ANTHROPIC_API_KEY not found in .env file or environment variables" << endl;
        return 1;
    }
    ClaudeMedicalImageAnalyzer analyzer(api_key);
    string imagePath = "../resources/mri_1.jpeg";
    cout << "Analyzing the medical image with Claude 3.7 Sonnet: " << imagePath << endl;
    string result = analyzer.analyzeMedicalImage(imagePath);
    cout << "\nAnalysis Result:\n" << result << endl;
    return 0;
}

// ====================================================================
//                    IMAGE PROCESSING FUNCTION
// ====================================================================

// Original processImage function (currently uses argv; should be refactored)
int processImage() {
    // Here, we assume processImage reads from argv[3] and writes to argv[2].
    // For this example, this function is left unchanged.
    Image test(argv[3]);
    Image gray_lum = test;
    gray_lum.grayscale_lum();
    gray_lum.write(argv[2]);
    return 0;
}

// ====================================================================
//                      CROW WEB SERVER & ROUTES
// ====================================================================

int main() {
    crow::SimpleApp app;

    // Home route
    CROW_ROUTE(app, "/")([]() {
        return "Hello, Clang Docker!";
    });

    // Upload route: Executes the full pipeline and returns final results as JSON.
    CROW_ROUTE(app, "/upload").methods(crow::HTTPMethod::Post)([](const crow::request &req) {
        // Step 0: Save the uploaded file.
        if (req.body.empty())
            return crow::response(400, "No file uploaded");
        string uploadPath = "uploaded_image.jpg";
        ofstream file(uploadPath, ios::binary);
        file.write(req.body.c_str(), req.body.size());
        file.close();

        // Step 1: Process the image (using processImage function)
        int processStatus = processImage();

        // Step 2: Save the preprocessed image.
        Mat preprocessed = imread("processed.jpg");  // Assume processImage outputs "processed.jpg"
        bool saveStatus = false;
        if (!preprocessed.empty()) {
            vector<uint8_t> preprocessedData(preprocessed.data,
                                              preprocessed.data + preprocessed.total() * preprocessed.elemSize());
            saveStatus = true;
            save_processed_image(preprocessed, string(UPLOAD_DIR) + "processed_saved.jpg");
        }

        // Step 3: Analyze the image.
        int analysisStatus = image_analyser();

        // Step 4: Quantify the image.
        int quantStatus = image_quantification();

        // Step 5: Encrypt the image.
        int encryptStatus = image_encryption();

        // Step 6: Store the encrypted image locally.
        int storageStatus = image_storage();

        // Step 7: Upload the image to Cloudinary.
        int uploadStatus = upload_to_cloudinary();

        // Step 8: Get final results from model inference.
        int finalStatus = final_results();

        // Combine statuses into a JSON response.
        crow::json::wvalue response;
        response["process_image"] = processStatus;
        response["save_preprocessed"] = saveStatus;
        response["analysis"] = analysisStatus;
        response["quantification"] = quantStatus;
        response["encryption"] = encryptStatus;
        response["storage"] = storageStatus;
        response["upload"] = uploadStatus;
        response["final_results"] = finalStatus;
        response["message"] = "Image processed and analyzed successfully";
        return crow::response(200, response);
    });

    app.port(8080).multithreaded().run();
}
