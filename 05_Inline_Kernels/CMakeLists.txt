add_executable(05_Inline_Kernels
               "main.cpp")
target_link_libraries(05_Inline_Kernels libocca)
target_include_directories(05_Inline_Kernels PRIVATE
                           $<BUILD_INTERFACE:${OCCA_SOURCE_DIR}/src>)
