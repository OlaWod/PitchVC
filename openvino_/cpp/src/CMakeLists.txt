set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(EXECNAME "app")

find_package(OpenVINO REQUIRED)
find_package(Torch REQUIRED)

add_executable(${EXECNAME} "main.cpp")

target_link_libraries(${EXECNAME} PRIVATE openvino::runtime "${TORCH_LIBRARIES}")
