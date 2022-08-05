add_executable(02_Loops
               "main.cpp")
target_link_libraries(02_Loops libocca)
target_include_directories(02_Loops PRIVATE
                           $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

add_custom_target(02_Loops_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/matrixMultiply.okl matrixMultiply.okl)
add_dependencies(02_Loops 02_Loops_okl)
