# GTF++
Global Task Flow library

A library that performs calculations on distributed and shared memory machines using tasks with dependencies. The execution is fully asynchronous. The communications are done using active messages. The implementation uses [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) and [GASNet-EX](https://gasnet.lbl.gov/) as backends.

You can compile the test files using:
<pre>
mkdir build
cd build
cmake ..
make
make install
</pre>
This will create an executable in the bin/ directory. The source files are in src/. build/ contains compiled object files and cmake files.

[UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) must be installed.

[Installation instructions for UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md)

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
in the [installation instructions for UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL.md)
