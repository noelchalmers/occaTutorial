add_executable(04_Streams
               "main.cpp")
target_link_libraries(04_Streams libocca)
target_include_directories(04_Streams PRIVATE
                           $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

add_custom_target(04_Streams_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kernels.okl kernels.okl)
add_dependencies(04_Streams 04_Streams_okl)
