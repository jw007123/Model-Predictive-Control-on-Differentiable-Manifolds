add_library(Lib_MPC "BicycleModel.cpp" "Quadcopter.cpp")

target_include_directories (Lib_MPC PUBLIC "../Include")
target_include_directories (Lib_MPC PUBLIC "../External/eigen")

find_package(osqp REQUIRED)
target_link_libraries(Lib_MPC PRIVATE osqp::osqpstatic)
