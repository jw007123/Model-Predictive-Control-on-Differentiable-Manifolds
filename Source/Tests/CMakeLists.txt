# OpenMP for Eigen
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

    add_compile_definitions(HAS_OMP)
endif()

find_package(osqp REQUIRED)

# bigobj required for each. pthreads needed on gcc
if (MSVC)
	add_compile_options(/bigobj)
elseif (UNIX)
	add_compile_options(-Wa,-mbig-obj)

	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# .obj resources
if (EXISTS "../../Resources")
    file(COPY "../../Resources" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# stb available
add_compile_definitions(HAS_STB)

add_executable(Tests "Main.cpp")
target_link_libraries(Tests PRIVATE osqp::osqpstatic)
target_link_libraries(Tests LINK_PUBLIC Lib_MPC)
