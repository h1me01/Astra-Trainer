cmake_minimum_required(VERSION 3.17)
project(Astra-Trainer VERSION 1.0 LANGUAGES CUDA CXX)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# More aggressive CPU optimization flags for Linux
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -DNDEBUG")

# Enhanced CUDA optimization flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")

# Additional CUDA optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -Xcompiler=-ffast-math,-march=native,-mtune=native,-funroll-loops \
    --maxrregcount=64 \
    --ptxas-options=-v")

# Enable Link-Time Optimization for better cross-module optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)

if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)
add_executable(astra-trainer ${SRCS})

set_target_properties(astra-trainer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Helps with device linking
)

# Auto-detect best CUDA architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

target_include_directories(astra-trainer PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(astra-trainer PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Threads::Threads
)

# Link with OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(astra-trainer PRIVATE OpenMP::OpenMP_CXX)
endif()

# Strip symbols in release builds for smaller binary (optional)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET astra-trainer POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:astra-trainer>
        COMMENT "Stripping symbols from release binary"
    )
endif()