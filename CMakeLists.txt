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
            deps
    )

    add_custom_target(compile-vertex-shader
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} shader.vert -o vert.spv
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shaders
        VERBATIM
        COMMENT "Compiling vertex shader"
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.vert
        SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.vert
    )

    add_custom_target(compile-fragment-shader
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} shader.frag -o frag.spv
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shaders
        VERBATIM
        COMMENT "Compiling fragment shader"
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.frag
        SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.frag
    )

    # TODO try to omit compilation if shader files did not change
    # custom targets are always considered out of date therefore they execute each build
    add_dependencies(vulkan-tutorial compile-vertex-shader compile-fragment-shader)

    add_custom_command(
        TARGET vulkan-tutorial
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/shaders/vert.spv ${CMAKE_CURRENT_LIST_DIR}/shaders/frag.spv ${CMAKE_CURRENT_LIST_DIR}/textures/texture.jpg $<TARGET_FILE_DIR:vulkan-tutorial>
        COMMENT "Copying shader files to exe output dir"
    )
