#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Function to compute entropy
double computeEntropy(const Mat& grayImage) {
    vector<int> histogram(256, 0);
    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols; j++) {
            histogram[(int)grayImage.at<uchar>(i, j)]++;
        }
    }
    
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

// Function to compute contrast
double computeContrast(const Mat& grayImage) {
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);
    return stddev[0] * stddev[0]; // Contrast is variance
}

// Function to compute correlation
double computeCorrelation(const Mat& grayImage) {
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    double mean = 0.0, variance = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;
    
    // Calculate mean
    for (int i = 0; i < 256; i++) {
        mean += i * hist.at<float>(i);
    }
    mean /= totalPixels;
    
    // Calculate variance
    for (int i = 0; i < 256; i++) {
        variance += pow(i - mean, 2) * hist.at<float>(i);
    }
    variance /= totalPixels;
    
    return sqrt(variance) / mean;
}

// Function to compute energy
double computeEnergy(const Mat& grayImage) {
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    double energy = 0.0;
    int totalPixels = grayImage.rows * grayImage.cols;
    
    for (int i = 0; i < histSize; i++) {
        double p = hist.at<float>(i) / totalPixels;
        energy += p * p;
    }
    return energy;
}

// Function to compute sharpness using Laplacian variance
double computeSharpness(const Mat& grayImage) {
    Mat laplacian;
    Laplacian(grayImage, laplacian, CV_64F);
    Scalar mean, stddev;
    meanStdDev(laplacian, mean, stddev);
    return stddev[0] * stddev[0];
}

// Function to compute noise level
double computeNoiseLevel(const Mat& grayImage) {
    Mat blurred;
    GaussianBlur(grayImage, blurred, Size(7, 7), 0);
    Mat noise;
    absdiff(grayImage, blurred, noise);
    Scalar mean, stddev;
    meanStdDev(noise, mean, stddev);
    return stddev[0];
}

// Function to compute resolution
Size computeResolution(const Mat& image) {
    return image.size();
}

// Function for image segmentation
Mat segmentImage(const Mat& grayImage) {
    Mat binaryImage;
    // Apply Otsu's thresholding
    double thresh = threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    // Apply additional preprocessing if needed
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, kernel);
    
    return binaryImage;
}

// Add new function to compute skewness
double computeSkewness(const Mat& grayImage) {
    Scalar mean, stddev;
    meanStdDev(grayImage, mean, stddev);
    
    double sum = 0.0;
    double pixelCount = grayImage.rows * grayImage.cols;
    
    for(int i = 0; i < grayImage.rows; i++) {
        for(int j = 0; j < grayImage.cols; j++) {
            double diff = grayImage.at<uchar>(i,j) - mean[0];
            sum += pow(diff, 3);
        }
    }
    
    return sum / (pixelCount * pow(stddev[0], 3));
}

// Enhanced object detection with additional metrics
struct ObjectMetrics {
    double area;
    double perimeter;
    Point2f centroid;
    Rect boundingBox;
};

vector<ObjectMetrics> detectObjectsEnhanced(const Mat& binaryImage) {
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    vector<ObjectMetrics> objects;
    
    for (const auto& contour : contours) {
        ObjectMetrics metrics;
        
        // Compute area
        metrics.area = contourArea(contour);
        
        // Compute perimeter
        metrics.perimeter = arcLength(contour, true);
        
        // Compute centroid
        Moments m = moments(contour);
        metrics.centroid = Point2f(m.m10/m.m00, m.m01/m.m00);
        
        // Get bounding box
        metrics.boundingBox = boundingRect(contour);
        
        objects.push_back(metrics);
    }
    
    return objects;
}

// Function to perform and analyze morphological operations
struct MorphologicalMetrics {
    Mat dilated;
    Mat eroded;
    Mat opened;
    Mat closed;
    double areaChange;
};

MorphologicalMetrics performMorphologicalAnalysis(const Mat& binaryImage) {
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
struct StatisticalMetrics {
    double mean;
    double variance;
    double stdDev;
    double skewness;
    double kurtosis;
};

StatisticalMetrics computeStatistics(const Mat& grayImage) {
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
    
    for(int i = 0; i < grayImage.rows; i++) {
        for(int j = 0; j < grayImage.cols; j++) {
            double diff = grayImage.at<uchar>(i,j) - mean[0];
            sum += pow(diff, 4);
        }
    }
    
    stats.kurtosis = (sum / (pixelCount * pow(stddev[0], 4))) - 3.0;
    
    return stats;
}

int main() {
    // Load the ultrasound image
    Mat image = imread("thyroid_ultrasound.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
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
    for (size_t i = 0; i < objectMetrics.size(); i++) {
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