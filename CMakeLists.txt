cmake_minimum_required(VERSION 2.8)
project( visual_odometry )
find_package( OpenCV REQUIRED )

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} 
  /home/anoop/eigen/ )

add_subdirectory(five-point-nister)

file(GLOB viso
    "src/featurematch.h"
    "src/main.cpp"
)
add_executable( mvo ${viso} )
set_target_properties(mvo PROPERTIES COMPILE_FLAGS "-std=c++11 -w -g -O3") 
target_link_libraries( mvo five-point-nister ${OpenCV_LIBS} )
