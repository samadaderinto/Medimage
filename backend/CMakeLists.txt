cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directory for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Gather all .cpp files from the 'source' folder
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/source/*.cpp")

# Create the executable from the collected sources
add_executable(MyProject ${SOURCES})

# Post-build step: Copy the 'resources' folder to the output directory
add_custom_command(TARGET MyProject POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:MyProject>/resources
)
