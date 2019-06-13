cmake_minimum_required(VERSION 3.13)

find_package(TensorflowCC REQUIRED)

add_executable(VectorExpression VectorExpression.cxx)
set_property(TARGET VectorExpression PROPERTY CXX_STANDARD 17)

target_link_libraries(VectorExpression TensorflowCC::Shared)
