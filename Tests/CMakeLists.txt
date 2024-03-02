# osqp
find_package(osqp REQUIRED)

# bigobj required for each. pthreads needed on gcc
if (MSVC)
	add_compile_options(/bigobj)
elseif (UNIX)
	add_compile_options(-Wa,-mbig-obj)

	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

add_executable(Tests "Main.cpp")

target_link_libraries(Tests PRIVATE osqp::osqpstatic)
target_include_directories(Tests PUBLIC "../Include")
target_include_directories(Tests PUBLIC "../Examples")
target_include_directories(Tests PUBLIC "../External/eigen")
