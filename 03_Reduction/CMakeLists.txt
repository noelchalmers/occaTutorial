add_executable(03_Reduction
               "main.cpp")
target_link_libraries(03_Reduction libocca)
target_include_directories(03_Reduction PRIVATE
                           $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

add_custom_target(03_Reduction_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/sum.okl sum.okl)
add_dependencies(03_Reduction 03_Reduction_okl)
