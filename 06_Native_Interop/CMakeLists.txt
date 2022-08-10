add_executable(06_Native_Interop
               "main.cpp")
target_link_libraries(06_Native_Interop libocca hip::host)
target_include_directories(06_Native_Interop PRIVATE
                           $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)

add_custom_target(06_Native_Interop_cpp ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.cpp addVectors.cpp)
add_dependencies(06_Native_Interop 06_Native_Interop_cpp)
