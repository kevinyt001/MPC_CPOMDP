cmake_minimum_required (VERSION 3.9) # CMP0069 NEW

project (MPC_POMDP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

##############################
##       Dependencies       ##
##############################

# set(BOOST_VERSION_REQUIRED 1.59)
# set(EIGEN_VERSION_REQUIRED 3.2.92)

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 COMPONENTS program_options REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

if (NOT WIN32)
    add_definitions(
        -Wall
        -Wextra
    )
endif()

#add_definitions(-O3 -std=c++17 -lnlopt -lm -march=native -fopenmp)
add_definitions(-O3 -std=c++17 -march=native -fopenmp)

include_directories("${PROJECT_SOURCE_DIR}/include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}")

find_package(NLOPT REQUIRED)
if ( NOT NLOPT_FOUND)
	message(STATUS "This project requires the NLOPT library, and will not be compiled.")
	return()
endif()

include_directories(SYSTEM ${NLOPT_INCLUDE_DIRS})

#find_package(MKL REQUIRED)
#if ( NOT MKL_FOUND)
#	message(STATUS "This project requires the MKL library, and will not be compiled.")
#	return()
#endif()

#include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
#add_definitions(-DMKL_LP64 -m64)
#add_definitions(-DEIGEN_USE_MKL_ALL)
#add_definitions(-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core)
#add_definitions(-liomp5 -lpthread -lm -ldl)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE SRC_LIB ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/include/*.hpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIB})
target_link_libraries(${PROJECT_NAME} ${NLOPT_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})

file(GLOB files ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(f ${files})
GET_FILENAME_COMPONENT(fname ${f} NAME_WE )
add_executable(${fname} ${f} )
#set_property(TARGET ${fname} PROPERTY CXX_STANDARD 17)
target_link_libraries(${fname} ${PROJECT_NAME} ${NLOPT_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})
endforeach()