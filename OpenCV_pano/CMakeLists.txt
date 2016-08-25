cmake_minimum_required(VERSION 2.8)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project( homography )
find_package( OpenCV REQUIRED )

add_executable( homography homography.cpp )
target_link_libraries( homography ${OpenCV_LIBS} )
