# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nazar/IdeaProjects/torch_scala/src/native

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib

# Include any dependencies generated for this target.
include CMakeFiles/test_lib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_lib.dir/flags.make

CMakeFiles/test_lib.dir/test.cpp.o: CMakeFiles/test_lib.dir/flags.make
CMakeFiles/test_lib.dir/test.cpp.o: /home/nazar/IdeaProjects/torch_scala/src/native/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_lib.dir/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_lib.dir/test.cpp.o -c /home/nazar/IdeaProjects/torch_scala/src/native/test.cpp

CMakeFiles/test_lib.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_lib.dir/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nazar/IdeaProjects/torch_scala/src/native/test.cpp > CMakeFiles/test_lib.dir/test.cpp.i

CMakeFiles/test_lib.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_lib.dir/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nazar/IdeaProjects/torch_scala/src/native/test.cpp -o CMakeFiles/test_lib.dir/test.cpp.s

CMakeFiles/test_lib.dir/test.cpp.o.requires:

.PHONY : CMakeFiles/test_lib.dir/test.cpp.o.requires

CMakeFiles/test_lib.dir/test.cpp.o.provides: CMakeFiles/test_lib.dir/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_lib.dir/build.make CMakeFiles/test_lib.dir/test.cpp.o.provides.build
.PHONY : CMakeFiles/test_lib.dir/test.cpp.o.provides

CMakeFiles/test_lib.dir/test.cpp.o.provides.build: CMakeFiles/test_lib.dir/test.cpp.o


# Object files for target test_lib
test_lib_OBJECTS = \
"CMakeFiles/test_lib.dir/test.cpp.o"

# External object files for target test_lib
test_lib_EXTERNAL_OBJECTS =

test_lib: CMakeFiles/test_lib.dir/test.cpp.o
test_lib: CMakeFiles/test_lib.dir/build.make
test_lib: libtorch_scala.so
test_lib: /home/nazar/libtorch/lib/libtorch.so
test_lib: /home/nazar/libtorch/lib/libc10_cuda.so
test_lib: /home/nazar/libtorch/lib/libcaffe2.so
test_lib: /home/nazar/libtorch/lib/libc10.so
test_lib: /usr/local/cuda/lib64/libcufft.so
test_lib: /usr/local/cuda/lib64/libcurand.so
test_lib: /usr/local/cuda/lib64/libcudnn.so
test_lib: /usr/local/cuda/lib64/libculibos.a
test_lib: /usr/local/cuda/lib64/libcublas.so
test_lib: /usr/lib/x86_64-linux-gnu/libcuda.so
test_lib: /usr/local/cuda/lib64/libnvrtc.so
test_lib: /usr/local/cuda/lib64/libnvToolsExt.so
test_lib: /usr/local/cuda/lib64/libcudart_static.a
test_lib: /usr/lib/x86_64-linux-gnu/librt.so
test_lib: /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libjawt.so
test_lib: /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so
test_lib: CMakeFiles/test_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_lib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_lib.dir/build: test_lib

.PHONY : CMakeFiles/test_lib.dir/build

CMakeFiles/test_lib.dir/requires: CMakeFiles/test_lib.dir/test.cpp.o.requires

.PHONY : CMakeFiles/test_lib.dir/requires

CMakeFiles/test_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_lib.dir/clean

CMakeFiles/test_lib.dir/depend:
	cd /home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nazar/IdeaProjects/torch_scala/src/native /home/nazar/IdeaProjects/torch_scala/src/native /home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib /home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib /home/nazar/IdeaProjects/torch_scala/src/main/resources/torch_lib/CMakeFiles/test_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_lib.dir/depend

