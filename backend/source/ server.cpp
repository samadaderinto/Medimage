#include "crow.h"
#include <fstream>
#include <vector>

#define UPLOAD_DIR "./uploads/"  // Directory to store uploaded images

// Function to save uploaded file
bool save_file(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream file(UPLOAD_DIR + filename, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

int main() {
    crow::SimpleApp app;

    // Home route
    CROW_ROUTE(app, "/")([]() {
        return "Hello, Clang Docker!";
    });

    // Image upload route
    CROW_ROUTE(app, "/upload").methods(crow::HTTPMethod::Post)([](const crow::request& req) {
        if (req.body.empty()) {
            return crow::response(400, "No image uploaded.");
        }

        std::string filename = "uploaded_image.jpg";  // Default filename (you can generate unique names)
        std::vector<uint8_t> file_data(req.body.begin(), req.body.end());

        if (save_file(filename, file_data)) {
            return crow::response(200, "Image uploaded successfully: " + filename);
        } else {
            return crow::response(500, "Failed to save image.");
        }
    });

    // Create upload directory if not exists
    system("mkdir -p " UPLOAD_DIR);

    app.port(8080).multithreaded().run();
}
