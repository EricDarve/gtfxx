# GTF++
Global Task Flow library

A library that performs calculations on distributed and shared memory machines using tasks with dependencies. The execution is fully asynchronous. The communications are done using active messages. The implementation uses [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) and [GASNet-EX](https://gasnet.lbl.gov/) as backends.

## Installation instructions

We need to first install [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home).

[Installation instructions for UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md)

Make a note of the directory where UPC++ was installed. See the [installation instructions for UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md).

In the file CMakeLists.txt, update the following settings:
<pre>
set(UPCXX_INSTALL "/path/to/upcxx/install/directory")
set(UPCXX_THREADMODE "par")
</pre>
to match your installation of UPC++. See
<pre>
cd <upcxx-source-path>
./install <upcxx-install-path>
</pre>
in the [UPC++ instructions](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md).

Then, you can compile the test files using:
<pre>
mkdir build
cd build
cmake ..
make
make install
</pre>
This will create an executable in the `bin/` directory. The source files are in `src/`. 

`build/` contains compiled object files and cmake files.

To run the code, use the following command:
<pre>
/path/to/upcxxinstall/upcxx/bin/upcxx-run -n 16 ./gtfxx --gtest_filter=* --gtest_break_on_failure --gtest_repeat=1
</pre>
This will run the code using the [Google test](https://github.com/google/googletest) framework.

`-n 16` specifies the number of processes (ranks) to use. `--gtest_filter=*` allows you to select a subset of the Google tests if needed. See the [Google test documentation for details](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md).

[Gtest primer](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)

[Gtest samples to get started](https://github.com/google/googletest/tree/master/googletest/samples)

`--gtest_break_on_failure` means that the code will stop as soon as one test fails (rather than continuing on with the remaining tests).

`--gtest_repeat=1` allows you to re-run the tests many times. This can be useful for multi-threaded codes where the order of execution of tasks is somewhat random. It can help uncover parallel race conditions. Try `--gtest_repeat=8` to run the tests 8 times.
