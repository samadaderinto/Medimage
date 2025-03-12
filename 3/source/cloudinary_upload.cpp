#include <iostream>
#include <curl/curl.h>
#include <fstream>

using namespace std;

// Cloudinary credentials
#define CLOUD_NAME "dd0ogn2qg"
#define API_KEY "782955842683427"
#define API_SECRET "YZG1IJY7BJ7InxCD9LZQ7w205gU"

// Callback function to capture response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to upload image
void uploadImage(const string& imagePath) {
    CURL* curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        string response;
        string apiUrl = "https://api.cloudinary.com/v1_1/" + string(CLOUD_NAME) + "/image/upload";
        string apiKey = "api_key=" + string(API_KEY);
        string uploadPreset = "upload_preset=unsigned";  // Change if using presets
        string fileField = "file=@" + imagePath;
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: multipart/form-data");

        curl_easy_setopt(curl, CURLOPT_URL, apiUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);

        curl_mime* mime;
        curl_mimepart* part;
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
        if (res != CURLE_OK) {
            cerr << "Failed to upload: " << curl_easy_strerror(res) << endl;
        } else {
            cout << "Upload Successful! Response: " << response << endl;
        }

        curl_easy_cleanup(curl);
        curl_mime_free(mime);
        curl_global_cleanup();
    }
}

int main() {
    string imagePath = "thyroid_ultrasound.jpg";  
    uploadImage(imagePath);
    return 0;
}
