# GTF++

*A Global Task Flow library*

A library that performs calculations on distributed and shared memory machines using tasks with dependencies. The execution is fully asynchronous. The communications are done using active messages. The implementation uses [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) and [GASNet-EX](https://gasnet.lbl.gov/) as backends.

The project is currently being developed and tested on macOS High Sierra.

## Installation instructions

You can compile the file `main.cpp` (which implements some basic tests) with the following:
<pre>
cd &lt;root directory containing the git clone repository of gtfxx&gt;
mkdir build
cd build
cmake ..
make install
</pre>
This will create the executable `gtfxx` in the `bin/` directory. The source files are in `src/`. The include files are in `include/`.

You can use
<pre>
make VERBOSE=1
</pre>
to get more information about the building process (compilation flags being used, etc).

## What happens when you run `make install`

The CMake file `CMakeLists.txt` is configured to download and install [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home). The UPC++ installation script  
`build/upcxx-build/upcxx-download/src/upcxx/install`  
in turn downloads and installs [GASNet-EX](https://gasnet.lbl.gov/).

For more information about the installation process for UPC++, see the [installation instructions for UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md).

`CMakeLists.txt` also has instructions to download and install the [Google Test framework](https://github.com/google/googletest). The file `src/main.cpp` is configured to run Google tests at this time. This is used to test and develop the GTF++ library.

`build/` contains compiled object files and cmake files. You can delete the content of `build/` if you want to rebuild the entire project (including UPC++, GASNet, and the Google test framework) from scratch. This will download and recompile again the UPC++ and Google Test libraries.

The UPC++ files after installation are in `build/upcxx/`. The UPC++ binaries, `upcxx-run` (required to run the code) and `upcxx-meta` (used in the compilation process), are in `build/upcxx/bin/`.

## How to run a GTF++ code

To run the code, use the following command:
<pre>
cd &lt;root directory containing the git clone repository of gtfxx&gt;
./build/upcxx/bin/upcxx-run -n 16 ./bin/gtfxx --gtest_filter=* --gtest_break_on_failure --gtest_repeat=1
</pre>
This will run the code using the [Google test](https://github.com/google/googletest) framework.

`-n 16` specifies the number of processes (ranks) to use. `--gtest_filter=*` allows you to select a subset of the Google tests if needed. See the [Google test documentation for details](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md).

[Gtest primer](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)

[Gtest samples to get started](https://github.com/google/googletest/tree/master/googletest/samples)

`--gtest_break_on_failure` means that the code will stop as soon as one test fails (rather than continuing on with the remaining tests).

`--gtest_repeat=1` allows you to re-run the tests many times. This can be useful for multi-threaded codes where the order of execution of tasks is somewhat random. It can help uncover parallel race conditions. Try `--gtest_repeat=8` to run the tests 8 times.
