cmake_minimum_required(VERSION 3.18)
project(estimate_pi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DFMT_HEADER_ONLY -lpthread")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --std=c++17")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# create a "results" directory at configuration time
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

# Gather source files from src/ and its subdirectories
file(GLOB SRC_CPP "src/*.cpp" "src/mc/*.cpp")
file(GLOB SRC_CU "src/cuda/*.cu")

# Create the executable using both C++ and CUDA source files
add_executable(${PROJECT_NAME} ${SRC_CPP} ${SRC_CU})

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDAToolkit_LIBRARIES})

# Find and link OpenMP if available
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link pthreads explicitly
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
