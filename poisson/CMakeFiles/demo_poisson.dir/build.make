# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/johnny/fenics_practice

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/johnny/fenics_practice

# Include any dependencies generated for this target.
include CMakeFiles/demo_poisson.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo_poisson.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo_poisson.dir/flags.make

CMakeFiles/demo_poisson.dir/main.cpp.o: CMakeFiles/demo_poisson.dir/flags.make
CMakeFiles/demo_poisson.dir/main.cpp.o: main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/johnny/fenics_practice/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/demo_poisson.dir/main.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/demo_poisson.dir/main.cpp.o -c /home/johnny/fenics_practice/main.cpp

CMakeFiles/demo_poisson.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_poisson.dir/main.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/johnny/fenics_practice/main.cpp > CMakeFiles/demo_poisson.dir/main.cpp.i

CMakeFiles/demo_poisson.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_poisson.dir/main.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/johnny/fenics_practice/main.cpp -o CMakeFiles/demo_poisson.dir/main.cpp.s

CMakeFiles/demo_poisson.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/demo_poisson.dir/main.cpp.o.requires

CMakeFiles/demo_poisson.dir/main.cpp.o.provides: CMakeFiles/demo_poisson.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/demo_poisson.dir/build.make CMakeFiles/demo_poisson.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/demo_poisson.dir/main.cpp.o.provides

CMakeFiles/demo_poisson.dir/main.cpp.o.provides.build: CMakeFiles/demo_poisson.dir/main.cpp.o

CMakeFiles/demo_poisson.dir/Poisson.cpp.o: CMakeFiles/demo_poisson.dir/flags.make
CMakeFiles/demo_poisson.dir/Poisson.cpp.o: Poisson.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/johnny/fenics_practice/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/demo_poisson.dir/Poisson.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/demo_poisson.dir/Poisson.cpp.o -c /home/johnny/fenics_practice/Poisson.cpp

CMakeFiles/demo_poisson.dir/Poisson.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_poisson.dir/Poisson.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/johnny/fenics_practice/Poisson.cpp > CMakeFiles/demo_poisson.dir/Poisson.cpp.i

CMakeFiles/demo_poisson.dir/Poisson.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_poisson.dir/Poisson.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/johnny/fenics_practice/Poisson.cpp -o CMakeFiles/demo_poisson.dir/Poisson.cpp.s

CMakeFiles/demo_poisson.dir/Poisson.cpp.o.requires:
.PHONY : CMakeFiles/demo_poisson.dir/Poisson.cpp.o.requires

CMakeFiles/demo_poisson.dir/Poisson.cpp.o.provides: CMakeFiles/demo_poisson.dir/Poisson.cpp.o.requires
	$(MAKE) -f CMakeFiles/demo_poisson.dir/build.make CMakeFiles/demo_poisson.dir/Poisson.cpp.o.provides.build
.PHONY : CMakeFiles/demo_poisson.dir/Poisson.cpp.o.provides

CMakeFiles/demo_poisson.dir/Poisson.cpp.o.provides.build: CMakeFiles/demo_poisson.dir/Poisson.cpp.o

CMakeFiles/demo_poisson.dir/Error.cpp.o: CMakeFiles/demo_poisson.dir/flags.make
CMakeFiles/demo_poisson.dir/Error.cpp.o: Error.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/johnny/fenics_practice/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/demo_poisson.dir/Error.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/demo_poisson.dir/Error.cpp.o -c /home/johnny/fenics_practice/Error.cpp

CMakeFiles/demo_poisson.dir/Error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_poisson.dir/Error.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/johnny/fenics_practice/Error.cpp > CMakeFiles/demo_poisson.dir/Error.cpp.i

CMakeFiles/demo_poisson.dir/Error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_poisson.dir/Error.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/johnny/fenics_practice/Error.cpp -o CMakeFiles/demo_poisson.dir/Error.cpp.s

CMakeFiles/demo_poisson.dir/Error.cpp.o.requires:
.PHONY : CMakeFiles/demo_poisson.dir/Error.cpp.o.requires

CMakeFiles/demo_poisson.dir/Error.cpp.o.provides: CMakeFiles/demo_poisson.dir/Error.cpp.o.requires
	$(MAKE) -f CMakeFiles/demo_poisson.dir/build.make CMakeFiles/demo_poisson.dir/Error.cpp.o.provides.build
.PHONY : CMakeFiles/demo_poisson.dir/Error.cpp.o.provides

CMakeFiles/demo_poisson.dir/Error.cpp.o.provides.build: CMakeFiles/demo_poisson.dir/Error.cpp.o

# Object files for target demo_poisson
demo_poisson_OBJECTS = \
"CMakeFiles/demo_poisson.dir/main.cpp.o" \
"CMakeFiles/demo_poisson.dir/Poisson.cpp.o" \
"CMakeFiles/demo_poisson.dir/Error.cpp.o"

# External object files for target demo_poisson
demo_poisson_EXTERNAL_OBJECTS =

demo_poisson: CMakeFiles/demo_poisson.dir/main.cpp.o
demo_poisson: CMakeFiles/demo_poisson.dir/Poisson.cpp.o
demo_poisson: CMakeFiles/demo_poisson.dir/Error.cpp.o
demo_poisson: CMakeFiles/demo_poisson.dir/build.make
demo_poisson: /usr/lib/x86_64-linux-gnu/libdolfin.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libxml2.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_system.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_thread.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_mpi.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_timer.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libpthread.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libhdf5.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libpthread.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libz.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libdl.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libm.so
demo_poisson: /usr/lib/slepcdir/3.4.2/linux-gnu-c-opt/lib/libslepc.so
demo_poisson: /usr/lib/petscdir/3.4.2/linux-gnu-c-opt/lib/libpetsc.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libumfpack.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libamd.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcholmod.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libccolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
demo_poisson: /usr/lib/x86_64-linux-gnu/librt.so
demo_poisson: /usr/lib/liblapack.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
demo_poisson: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
demo_poisson: /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcholmod.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libccolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
demo_poisson: /usr/lib/x86_64-linux-gnu/librt.so
demo_poisson: /usr/lib/liblapack.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so
demo_poisson: /usr/lib/libptscotch.so
demo_poisson: /usr/lib/libptesmumps.so
demo_poisson: /usr/lib/libptscotcherr.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libz.so
demo_poisson: /usr/lib/libmpi_cxx.so
demo_poisson: /usr/lib/libmpi.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libdl.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libhwloc.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libQtGui.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libQtCore.so
demo_poisson: /usr/lib/libvtkCommon.so
demo_poisson: /usr/lib/libvtkFiltering.so
demo_poisson: /usr/lib/libvtkImaging.so
demo_poisson: /usr/lib/libvtkGraphics.so
demo_poisson: /usr/lib/libvtkGenericFiltering.so
demo_poisson: /usr/lib/libvtkIO.so
demo_poisson: /usr/lib/libvtkRendering.so
demo_poisson: /usr/lib/libvtkVolumeRendering.so
demo_poisson: /usr/lib/libvtkHybrid.so
demo_poisson: /usr/lib/libvtkWidgets.so
demo_poisson: /usr/lib/libvtkParallel.so
demo_poisson: /usr/lib/libvtkInfovis.so
demo_poisson: /usr/lib/libvtkGeovis.so
demo_poisson: /usr/lib/libvtkViews.so
demo_poisson: /usr/lib/libvtkCharts.so
demo_poisson: /usr/lib/libQVTK.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libhdf5.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libz.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libdl.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libm.so
demo_poisson: /usr/lib/slepcdir/3.4.2/linux-gnu-c-opt/lib/libslepc.so
demo_poisson: /usr/lib/petscdir/3.4.2/linux-gnu-c-opt/lib/libpetsc.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libumfpack.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libamd.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcholmod.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libccolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
demo_poisson: /usr/lib/x86_64-linux-gnu/librt.so
demo_poisson: /usr/lib/liblapack.so
demo_poisson: /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so
demo_poisson: /usr/lib/libptscotch.so
demo_poisson: /usr/lib/libptesmumps.so
demo_poisson: /usr/lib/libptscotcherr.so
demo_poisson: /usr/lib/libmpi_cxx.so
demo_poisson: /usr/lib/libmpi.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libz.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libdl.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libm.so
demo_poisson: /usr/lib/slepcdir/3.4.2/linux-gnu-c-opt/lib/libslepc.so
demo_poisson: /usr/lib/petscdir/3.4.2/linux-gnu-c-opt/lib/libpetsc.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libumfpack.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libamd.so
demo_poisson: /usr/lib/libblas.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcholmod.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libcolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libccolamd.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
demo_poisson: /usr/lib/x86_64-linux-gnu/librt.so
demo_poisson: /usr/lib/liblapack.so
demo_poisson: /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so
demo_poisson: /usr/lib/libptscotch.so
demo_poisson: /usr/lib/libptesmumps.so
demo_poisson: /usr/lib/libptscotcherr.so
demo_poisson: /usr/lib/libmpi_cxx.so
demo_poisson: /usr/lib/libmpi.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libhwloc.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libQtGui.so
demo_poisson: /usr/lib/x86_64-linux-gnu/libQtCore.so
demo_poisson: /usr/lib/libvtkCommon.so
demo_poisson: /usr/lib/libvtkFiltering.so
demo_poisson: /usr/lib/libvtkImaging.so
demo_poisson: /usr/lib/libvtkGraphics.so
demo_poisson: /usr/lib/libvtkGenericFiltering.so
demo_poisson: /usr/lib/libvtkIO.so
demo_poisson: /usr/lib/libvtkRendering.so
demo_poisson: /usr/lib/libvtkVolumeRendering.so
demo_poisson: /usr/lib/libvtkHybrid.so
demo_poisson: /usr/lib/libvtkWidgets.so
demo_poisson: /usr/lib/libvtkParallel.so
demo_poisson: /usr/lib/libvtkInfovis.so
demo_poisson: /usr/lib/libvtkGeovis.so
demo_poisson: /usr/lib/libvtkViews.so
demo_poisson: /usr/lib/libvtkCharts.so
demo_poisson: /usr/lib/libQVTK.so
demo_poisson: CMakeFiles/demo_poisson.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable demo_poisson"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_poisson.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo_poisson.dir/build: demo_poisson
.PHONY : CMakeFiles/demo_poisson.dir/build

CMakeFiles/demo_poisson.dir/requires: CMakeFiles/demo_poisson.dir/main.cpp.o.requires
CMakeFiles/demo_poisson.dir/requires: CMakeFiles/demo_poisson.dir/Poisson.cpp.o.requires
CMakeFiles/demo_poisson.dir/requires: CMakeFiles/demo_poisson.dir/Error.cpp.o.requires
.PHONY : CMakeFiles/demo_poisson.dir/requires

CMakeFiles/demo_poisson.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo_poisson.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo_poisson.dir/clean

CMakeFiles/demo_poisson.dir/depend:
	cd /home/johnny/fenics_practice && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/johnny/fenics_practice /home/johnny/fenics_practice /home/johnny/fenics_practice /home/johnny/fenics_practice /home/johnny/fenics_practice/CMakeFiles/demo_poisson.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo_poisson.dir/depend
