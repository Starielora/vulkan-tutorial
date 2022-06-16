cmake_minimum_required(VERSION 3.22)

# vulkan
find_package(Vulkan REQUIRED)

if(NOT Vulkan_FOUND)
    message(FATAL "Vulkan not found.")
endif()

# glm
add_subdirectory(glm)

# glfw
add_subdirectory(glfw)

target_link_libraries(glfw PUBLIC ${Vulkan_LIBRARIES})
target_include_directories(glfw PUBLIC ${Vulkan_INCLUDE_DIRS})

# # imgui
# add_library(Dear-ImGui STATIC
#     imgui/imconfig.h
#     imgui/imgui_demo.cpp
#     imgui/imgui_draw.cpp
#     imgui/imgui_internal.h
#     imgui/imgui_tables.cpp
#     imgui/imgui_widgets.cpp
#     imgui/imgui.cpp
#     imgui/imgui.h
#     imgui/imstb_rectpack.h
#     imgui/imstb_textedit.h
#     imgui/imstb_truetype.h

#     # backends
#     imgui/backends/imgui_impl_glfw.cpp
#     imgui/backends/imgui_impl_glfw.h
#     imgui/backends/imgui_impl_opengl3_loader.h
#     imgui/backends/imgui_impl_opengl3.cpp
#     imgui/backends/imgui_impl_opengl3.h
#     )

# target_include_directories(Dear-ImGui PUBLIC imgui imgui/backends)
# target_link_libraries(Dear-ImGui PRIVATE glfw)
