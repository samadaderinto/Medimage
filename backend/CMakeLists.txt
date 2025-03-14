cmake_minimum_required(VERSION 3.10)
project(Medimage)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set vcpkg toolchain if needed
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "/Users/samad/Documents/GitHub/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Find packages from vcpkg
find_package(Crow CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# === ONNX Runtime Setup (Ensure .dylib is used instead of .dll) ===

# Define ONNX Runtime Path (for macOS)
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/external/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Ensure ONNX Runtime Include Directory Exists
if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(STATUS "Found ONNX Runtime headers at: ${ONNXRUNTIME_INCLUDE_DIR}")
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
else()
    message(FATAL_ERROR "ONNX Runtime headers not found in ${ONNXRUNTIME_INCLUDE_DIR}. Please download the macOS version of ONNX Runtime.")
endif()

# Ensure ONNX Runtime Library Directory Exists
if(EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib")
    message(STATUS "Found ONNX Runtime library at: ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib")
    link_directories(${ONNXRUNTIME_LIB_DIR})
else()
    message(FATAL_ERROR "ONNX Runtime library not found. Expected libonnxruntime.dylib in ${ONNXRUNTIME_LIB_DIR}. Please download the macOS version.")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/homebrew/opt/openssl/include)

# Link directories
link_directories(/opt/homebrew/lib)

# Collect all source files
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/source/*.cpp")

# Create the executable
add_executable(Medimage ${SOURCES})

# Link Crow (from vcpkg)
target_link_libraries(Medimage PRIVATE Crow::Crow)

# Link JSON (from vcpkg)
target_link_libraries(Medimage PRIVATE nlohmann_json::nlohmann_json)

# Link OpenCV
target_link_libraries(Medimage PRIVATE ${OpenCV_LIBS})

# Link OpenSSL
target_link_libraries(Medimage PRIVATE OpenSSL::SSL OpenSSL::Crypto)

# Link ONNX Runtime (.dylib)
target_link_libraries(Medimage PRIVATE onnxruntime)

# Ensure necessary ONNX Runtime libraries are copied to the build directory
if(APPLE)
    add_custom_command(TARGET Medimage POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib"
            "$<TARGET_FILE_DIR:Medimage>"
    )
endif()
