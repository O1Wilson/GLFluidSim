cmake_minimum_required(VERSION 3.15)
project(FluidSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(external/glfw)

add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

add_executable(FluidSim
    source/main.cpp
    source/Application.cpp
    source/FluidSolver.cpp
    source/InputHandler.cpp
    source/Renderer.cpp
)

target_include_directories(FluidSim PRIVATE
    source
)

target_link_libraries(FluidSim PRIVATE
    glad
    glfw
)

if (APPLE)
    target_link_libraries(FluidSim PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
elseif (UNIX)
    target_link_libraries(FluidSim PRIVATE dl pthread)
endif()

add_custom_command(TARGET FluidSim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:FluidSim>/shaders
)