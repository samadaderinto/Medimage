#include "crow.h"
#include <filesystem>
#include <algorithm>
#include <string>
#include <fstream>

#include <../include/middleware/headers.h>

int main()
{
    crow::App<HeaderCheckMiddleware> app;

    CROW_ROUTE(app, "/")([](){
        return "Hello world";
    });


    // CROW_ROUTE(app, "/upload")
    //     .methods(crow::HTTPMethod::POST)
    //     ([](const crow::request& req) {
    //         const std::string content_type = req.get_header_value("Content-Type");
    //         if (content_type.find("multipart/form-data") == std::string::npos) {
    //             return crow::response(400, "Bad Request: Expected multipart/form-data");
    //         }

    //         try {
    //             crow::multipart::message msg(req);
    //             std::string fileName;
    //             std::string fileContent;

    //             // Find the file part
    //             for(const auto& part : msg.parts) {
    //                 auto disposition = part.get_header_object("Content-Disposition");
    //                 if (disposition.find("name=\"file\"") != std::string::npos) {
    //                     auto pos = disposition.find("filename=\"");
    //                     if (pos != std::string::npos) {
    //                         pos += 10;  // length of 'filename="'
    //                         auto end = disposition.find('\"', pos);
    //                         if (end != std::string::npos) {
    //                             fileName = disposition.substr(pos, end - pos);
    //                             fileContent = part.body;
    //                             break;
    //                         }
    //                     }
    //                 }
    //             }

    //             if (fileName.empty()) {
    //                 return crow::response(400, "No file uploaded");
    //             }

    //             // Create uploads directory
    //             std::filesystem::create_directories("uploads");
    //             std::string uploadPath = "uploads/" + fileName;

    //             // Save the file
    //             std::ofstream fileStream(uploadPath, std::ios::binary);
    //             if (!fileStream) {
    //                 return crow::response(500, "Failed to save the uploaded file");
    //             }

    //             fileStream.write(fileContent.c_str(), fileContent.size());
    //             fileStream.close();

    //             // TODO: Add image processing
    //             // int processStatus = processImage(uploadPath);
    //             // if (processStatus != 0) {
    //             //     return crow::response(500, "Image processing failed");
    //             // }

    //             // int finalStatus = final_results();
    //             // if (finalStatus != 0) {
    //             //     return crow::response(500, "Model inference failed");
    //             // }

    //             crow::json::wvalue response;
    //             response["message"] = "File uploaded successfully";
    //             response["filename"] = fileName;
    //             return crow::response(200, response);

    //             return crow::response(200, "File uploaded successfully: " + fileName);
    //         }
    //     catch (const std::exception& e)
    //     {
    //         return crow::response(500, "Error processing multipart data: " + std::string(e.what()));
    //     } });

    // Run the server on port 8080
    app.port(8080).multithreaded().run();
}