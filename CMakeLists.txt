# cmake needs this line
cmake_minimum_required(VERSION 3.13)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Set target name, change with your favorite name
set(target nc_rd_facemask)
project(${target} CXX)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_CURRENT_LIST_DIR}")

# Declare the executable target built from your sources
add_executable(${target} headpose.cpp)

# Link your application with OpenCV libraries
target_link_libraries(${target} ${OpenCV_LIBS})

# Custom working directory
set_property(TARGET ${target} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake-build-debug")