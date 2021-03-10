cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(deform_conv_v2)

add_definitions(-D WITH_CUDA)

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(Torch_DIR "F:/workspace/code/lib/libtorch/debug/share/cmake/Torch")         #path to libtorch
ELSE ()
    set(Torch_DIR "F:/workspace/code/lib/libtorch/release_1_7_1/share/cmake/Torch")
ENDIF ()

find_package(CUDA)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA Not Found!")
endif(NOT CUDA_FOUND)

find_package(Python3 3.6.10 COMPONENTS Development)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python Not Found!")
endif(NOT Python3_FOUND)
message(STATUS "Python status:")
message(STATUS "    libraries: ${Python3_LIBRARIES}")
message(STATUS "    includes: ${Python3_INCLUDE_DIRS}")

find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

include_directories(${Python3_INCLUDE_DIRS})

set(CUDA_HOST_COMPILATION_CPP ON)
# set_target_properties(PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TORCH_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GF /Z7 /FS")
set(CUDA_NVCC_FLAGS -D__CUDA_NO_HALF_OPERATORS__)


file(GLOB SRC_LIST "src/*.cpp" "src/*.cu" "src/*.cuh")
message(STATUS "    src: ${SRC_LIST}")
cuda_add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} 
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)

# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_target_properties( ${PROJECT_NAME}
  PROPERTIES
  CXX_STANDARD 14
  DEBUG_POSTFIX "_d"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)