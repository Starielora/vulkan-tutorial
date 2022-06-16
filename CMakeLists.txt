cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)

project(vulkan-tutorial)

    add_subdirectory(deps)

    add_executable(vulkan-tutorial
        main.cpp
    )

    target_link_libraries(vulkan-tutorial
        PRIVATE
            glfw
            glm
            ${Vulkan_LIBRARIES}
    )

    target_include_directories(vulkan-tutorial
        PRIVATE
            ${Vulkan_INCLUDE_DIRS}
    )
