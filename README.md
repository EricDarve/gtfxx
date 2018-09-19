# GTF++
Global Task Flow library

A library that performs calculations on distributed and shared memory machines using tasks with dependencies. The execution is fully asynchronous. The communications are done using active messages. The implementation uses [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) and [GASNet-EX](https://gasnet.lbl.gov/) as backends.

The project is currently being developed and tested on macOS High Sierra.

You can compile the file `main.cpp` (which implements some basic tests) with the following:
<pre>
cd <root directory containing the git clone repository of gtfxx>
mkdir build
cd build
cmake ..
make install
</pre>
This will create the executable `gtfxx` in the `bin/` directory. The source files are in `src/`. The include files are in `include/`.

`build/` contains compiled object files and cmake files. You can delete the content of `build/` if you want to rebuild the entire project (including UPCXX, GASNet and the Google test framework) from scratch.

The CMake file `CMakeLists.txt` is configured to download and install [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home). The UPC++ installation script in turn downloads and install [GASNet-EX](https://gasnet.lbl.gov/). This happens when you run `make` in the `build/` directory.

For more information about the installation process for UPC++, see the [installation instructions for UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md).

`CMakeLists.txt` also has instructions to download and install the [Google Test framework](https://github.com/google/googletest). The file `main.cpp` is configured to run Google tests at this time. This is used to test and develop the GTF++ library.
