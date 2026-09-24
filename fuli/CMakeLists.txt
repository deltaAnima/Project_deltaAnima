cmake_minimum_required(VERSION 3.22)
project(VectorOrchestrator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW) # find_package(Boost) uses BoostConfig.cmake
endif()

option(ENABLE_GPU "Enable Faiss GPU and CUDA support" ON)

include(FetchContent)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)

# This Boost install only generated CMake package configs for the
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

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

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

set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(PROTO_FILES "${PROTO_SRC_DIR}/search_service.proto")

set(PROTO_SRCS "${PROTO_GEN_DIR}/search_service.pb.cc")
set(PROTO_HDRS "${PROTO_GEN_DIR}/search_service.pb.h")
set(GRPC_SRCS  "${PROTO_GEN_DIR}/search_service.grpc.pb.cc")
set(GRPC_HDRS  "${PROTO_GEN_DIR}/search_service.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND protobuf::protoc
        --proto_path=${PROTO_SRC_DIR}
        --cpp_out=${PROTO_GEN_DIR}
        --grpc_out=${PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        ${PROTO_FILES}
    DEPENDS ${PROTO_FILES} protobuf::protoc gRPC::grpc_cpp_plugin
    COMMENT "Compiling Protobuf definitions..."
)

set(COMMON_SOURCES
    src/main.cpp
    src/pipeline/orchestrator.cpp
    src/clients/redis_db_client.cpp
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
