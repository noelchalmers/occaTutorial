add_executable(01_Introduction
               "main.cpp")
target_link_libraries(01_Introduction libocca)
target_include_directories(01_Introduction PRIVATE
                           $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

add_custom_target(01_Introduction_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(01_Introduction 01_Introduction_okl)
