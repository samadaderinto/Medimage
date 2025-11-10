cmake_minimum_required(VERSION 3.10)
project(Medimage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add compiler flags
add_compile_definitions(CROW_ENABLE_SSL)
add_compile_definitions(CROW_DISABLE_STATIC_DIR)

# Find dependencies

# Find packages from system or vcpkg if defined
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# nlohmann_json from system package
find_package(nlohmann_json 3 REQUIRED)

# ONNX Runtime setup
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/external/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# Set RPATH for the executable
set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/source/*.cpp")

add_executable(Medimage ${SOURCES})

# Link dependencies
target_link_libraries(Medimage PRIVATE
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    OpenSSL::SSL
    OpenSSL::Crypto
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    pthread
    ssl
    crypto
)

# Copy ONNX runtime library if found
if(APPLE)
    set(ONNX_LIB_NAME libonnxruntime.dylib)
else()
    set(ONNX_LIB_NAME libonnxruntime.so)
endif()

add_custom_command(TARGET Medimage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/${ONNX_LIB_NAME}"
        "$<TARGET_FILE_DIR:Medimage>"
)
