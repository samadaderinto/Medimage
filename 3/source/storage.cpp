#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Function to save an image as a binary file
void saveImage(const Mat& image, const string& filename) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(image.data), image.total() * image.elemSize());
    file.close();
}

// Function to load an image from a binary file
Mat loadImage(const string& filename, int rows, int cols, int type) {
    ifstream file(filename, ios::binary);
    vector<uchar> buffer(istreambuf_iterator<char>(file), {});
    Mat image(rows, cols, type, buffer.data());
    return image.clone();
}

int main() {
    // Load ultrasound image
    string imagePath = "thyroid_ultrasound.jpg";
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    if (image.empty()) {
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
