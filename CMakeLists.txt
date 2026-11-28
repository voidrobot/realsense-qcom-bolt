# Copyright 2026 voidrobot
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(realsense_qcom_bolt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_WITH_QIRP "Enable Qualcomm QIRP DMA-BUF Zero-Copy transport" OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

if(BUILD_WITH_QIRP)
  find_package(qrb_ros_transport_image_type REQUIRED)
  find_package(qrb_ros_transport_point_cloud2_type REQUIRED)
  find_package(lib_mem_dmabuf REQUIRED)
  message(STATUS "Building realsense_qcom_bolt with Qualcomm QIRP DMA-BUF Zero-Copy support.")
endif()

# Compile GLSL 450 compute shaders -> C++ SPIR-V headers at build time
set(SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders")
file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

set(SHADER_NAMES
  yuyv_to_rgb8
  depth_filter
  align_z_init
  align_z_scatter
  align_z_finalize
  deproject_gather_xyzrgb
  aligned_to_xyzrgb_fused
  deproject_xyz_only
)

set(GENERATED_SHADER_HEADERS)
foreach(SHADER_NAME ${SHADER_NAMES})
  set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.comp")
  set(SHADER_HDR "${SHADER_OUT_DIR}/${SHADER_NAME}.spv.h")
  set(SHADER_VAR "${SHADER_NAME}_spv")
  add_custom_command(
    OUTPUT "${SHADER_HDR}"
    COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3
            -I"${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            --vn "${SHADER_VAR}"
            -o "${SHADER_HDR}"
            "${SHADER_SRC}"
    DEPENDS "${SHADER_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common_math.glsl"
    COMMENT "Compiling GLSL shader ${SHADER_NAME}.comp -> ${SHADER_NAME}.spv.h"
  )
  list(APPEND GENERATED_SHADER_HEADERS "${SHADER_HDR}")
endforeach()

add_library(${PROJECT_NAME} SHARED
  src/config.cpp
  src/vulkan_context.cpp
  src/output_backend.cpp
  src/vulkan_pipeline.cpp
  src/realsense_capture.cpp
  src/bolt_node.cpp
  ${GENERATED_SHADER_HEADERS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SHADER_OUT_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  Vulkan::Vulkan
  realsense2::realsense2
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  std_msgs
  tf2_ros
)

if(BUILD_WITH_QIRP)
  target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_WITH_QIRP)
  ament_target_dependencies(${PROJECT_NAME}
    qrb_ros_transport_image_type
    qrb_ros_transport_point_cloud2_type
    lib_mem_dmabuf
  )
  if(TARGET realsense2::realsense2)
    target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
      $<TARGET_PROPERTY:realsense2::realsense2,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
endif()

rclcpp_components_register_nodes(${PROJECT_NAME} "realsense_qcom_bolt::BoltNode")

add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  enable_testing()

  add_executable(test_config_router test/test_config_router.cpp src/config.cpp)
  target_include_directories(test_config_router PRIVATE include)
  add_test(NAME test_config_router COMMAND test_config_router)

  add_executable(test_dmabuf_pool test/test_dmabuf_pool.cpp)
  target_include_directories(test_dmabuf_pool PRIVATE include)
  target_link_libraries(test_dmabuf_pool pthread)
  add_test(NAME test_dmabuf_pool COMMAND test_dmabuf_pool)

  add_executable(test_math_and_imu test/test_math_and_imu.cpp)
  target_include_directories(test_math_and_imu PRIVATE include)
  add_test(NAME test_math_and_imu COMMAND test_math_and_imu)

  add_executable(test_vulkan_kernels
    test/test_vulkan_kernels.cpp
    src/vulkan_context.cpp
    ${GENERATED_SHADER_HEADERS}
  )
  target_include_directories(test_vulkan_kernels PRIVATE include "${SHADER_OUT_DIR}")
  target_link_libraries(test_vulkan_kernels Vulkan::Vulkan)
  add_test(NAME test_vulkan_kernels COMMAND test_vulkan_kernels)
endif()

ament_package()
