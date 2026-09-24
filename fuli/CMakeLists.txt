cmake_minimum_required(VERSION 3.22)
project(VectorOrchestrator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_GPU "Enable Faiss GPU and CUDA support" ON)

include(FetchContent)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenMP REQUIRED)

# This Boost 1.85 install only generated CMake package configs for the
# compiled components (system/thread/context/coroutine/chrono); header-only
# libraries boost_redis links against have no per-component target here.
# Shim them onto the umbrella Boost::boost include target.
foreach(_header_only_boost_lib asio assert core mp11 throw_exception)
    if(NOT TARGET Boost::${_header_only_boost_lib})
        add_library(Boost::${_header_only_boost_lib} INTERFACE IMPORTED)
        set_target_properties(Boost::${_header_only_boost_lib} PROPERTIES
            INTERFACE_LINK_LIBRARIES "Boost::boost;Threads::Threads"
        )
    endif()
endforeach()

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

FetchContent_Declare(
    asio_grpc
    GIT_REPOSITORY https://github.com/Tradias/asio-grpc.git
    GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(asio_grpc)

FetchContent_Declare(
    boost_redis
    GIT_REPOSITORY https://github.com/boostorg/redis.git
    GIT_TAG        boost-1.85.0
)
FetchContent_MakeAvailable(boost_redis)

FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG        v1.0.4
)
FetchContent_MakeAvailable(concurrentqueue)

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    yaml_cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
FetchContent_MakeAvailable(yaml_cpp)

set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(PROTO_FILES "${PROTO_SRC_DIR}/search_service.proto")

set(PROTO_SRCS "${PROTO_GEN_DIR}/search_service.pb.cc")
set(PROTO_HDRS "${PROTO_GEN_DIR}/search_service.pb.h")
set(GRPC_SRCS  "${PROTO_GEN_DIR}/search_service.grpc.pb.cc")
set(GRPC_HDRS  "${PROTO_GEN_DIR}/search_service.grpc.pb.h")


find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND protobuf::protoc
        --proto_path=${PROTO_SRC_DIR}
        --cpp_out=${PROTO_GEN_DIR}
        --grpc_out=${PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
        ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    COMMENT "Compiling Protobuf definitions..."
)

set(COMMON_SOURCES
    src/main.cpp
    src/pipeline/orchestrator.cpp
    src/clients/redis_client.cpp
    src/clients/embedding_client.cpp
    src/http/http_server.cpp
    src/deltaEGO/deltaEGO.cpp
    src/deltaEGO/Ayin.cpp
    src/deltaEGO/Carmen.cpp
    src/health/health_checker.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

if(ENABLE_GPU)
    find_package(CUDAToolkit REQUIRED)
    find_package(faiss REQUIRED)

    add_compile_definitions(USE_GPU_FAISS)
    list(APPEND COMMON_SOURCES src/engine/gpu_faiss_engine.cpp)
else()
    list(APPEND COMMON_SOURCES src/engine/mock_faiss_engine.cpp)
endif()

add_executable(orchestrator_server ${COMMON_SOURCES})

# Ayin's Int16Tensor::search_knn uses AVX-512 intrinsics directly (Ryzen 9 9950x target)
set_source_files_properties(src/deltaEGO/Ayin.cpp PROPERTIES
    COMPILE_OPTIONS "-march=native"
)

target_include_directories(orchestrator_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GEN_DIR}
)

target_link_libraries(orchestrator_server PRIVATE
    asio-grpc::asio-grpc
    Boost::system
    Boost::redis
    concurrentqueue
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    yaml-cpp::yaml-cpp
)

if(ENABLE_GPU)
    target_link_libraries(orchestrator_server PRIVATE
        faiss
        CUDA::cudart
        CUDA::cublas
        OpenMP::OpenMP_CXX
    )
endif()