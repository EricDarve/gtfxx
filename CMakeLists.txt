cmake_minimum_required(VERSION 3.11)
project(gtfxx LANGUAGES CXX)
add_executable(gtfxx src/main.cpp src/gtfxx.cpp src/util.cpp)
target_compile_features(gtfxx PUBLIC cxx_std_11)

# -------------------------
# UPCXX

project(upcxx-download NONE)
include(ExternalProject)
ExternalProject_Add(upcxx
  PREFIX upcxx
  URL "https://bitbucket.org/upcxx/upcxx/downloads/upcxx-2018.3.2.tar.gz"
  INSTALL_DIR upcxx
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE true
  BUILD_COMMAND     ./install ../../../upcxx
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

set(UPCXX_INSTALL "/Users/darve/git/ctxx/upcxx")
set(UPCXX_THREADMODE "par")

set(ENV "UPCXX_THREADMODE=${UPCXX_THREADMODE}")

# Run UPCXX shell script to get required compile flags
execute_process(COMMAND bash -c "${ENV} ${UPCXX_INSTALL}/bin/upcxx-meta CXX" OUTPUT_VARIABLE CXX OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "${ENV} ${UPCXX_INSTALL}/bin/upcxx-meta PPFLAGS" OUTPUT_VARIABLE PPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "${ENV} ${UPCXX_INSTALL}/bin/upcxx-meta LDFLAGS" OUTPUT_VARIABLE LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "${ENV} ${UPCXX_INSTALL}/bin/upcxx-meta LIBFLAGS" OUTPUT_VARIABLE LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

message("UPC++ compiling options")
message("Compiler: ${CXX}")
message("Pre-processing flags: ${PPFLAGS}")
message("Linker flags: ${LDFLAGS}")
message("Libraries: ${LIBS}")

# -------------------------
# Google test cmake commands
# Download and unpack googletest at configure time

configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

# Example instructions for using Google tests:
# Link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
#target_link_libraries(example gtest_main)
#add_test(NAME example_test COMMAND example)

# -------------------------
# Ready to compile our code

# Set compiler
set(CMAKE_CXX_COMPILER "${CXX}")

# Pre-processing flags
separate_arguments(PPFLAGS UNIX_COMMAND "${PPFLAGS}")
target_compile_options(gtfxx PRIVATE ${PPFLAGS}) # UPCXX pre-processing flags

# Include directories for compilation
target_include_directories(gtfxx PRIVATE /usr/local/Cellar/eigen/3.3.4/include/eigen3)
target_include_directories(gtfxx PRIVATE include)

# Options for linking
target_link_libraries(gtfxx ${LDFLAGS}) # UPCXX linker flags
# Required libraries for linking
target_link_libraries(gtfxx ${LIBS}) # UPCXX libraries
target_link_libraries(gtfxx gtest_main) # Google test library

# Installation Options
set(CMAKE_INSTALL_PREFIX "../") # Installing under "bin/" in the git root directory of the user
install(TARGETS gtfxx DESTINATION bin)
