#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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

#include <fstream>
#include <string>
#include <map>

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

int main()
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