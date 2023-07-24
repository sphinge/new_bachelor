cmake_minimum_required(VERSION 2.8)

project(Neural)

# Find the Shark libraries and includes
# set Shark_DIR to the proper location of Shark
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS serialization filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

# Executable
add_executable(Neural Neural.cpp)
set_property(TARGET Neural PROPERTY CXX_STANDARD 11)

# Link the Shark libraries and Boost libraries to the target
target_link_libraries(Neural ${SHARK_LIBRARIES} ${Boost_LIBRARIES})