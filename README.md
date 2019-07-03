# L2

A C++ deep learning library

## Structure

-   .h files in include/
-   .cpp files in L2/
-   compiled library is in lib/
-   use #pragma once in header files
-   for additional files, add the path to the cpp file to add_library in CMakeLists.txt

## ToDo

## Debugger

-   https://code.visualstudio.com/docs/cpp/config-wsl
-   https://github.com/microsoft/vscode-cpptools/issues/2998 (external_console: false)
-   https://github.com/microsoft/vscode-cpptools/issues/2998 ("pipeProgram": "bash.exe")
-   compile with -g flag to use debugger
-   can't use debugger on library files

## Running

### manually

-   g++ -g -o build/L2.o -c L2/L2.cpp
-   ar rcs build/libL2.a build/L2.o
-   g++ -g -o main.out main.cpp -Lbuild -lL2

### cmake

-   "ctrl-shift-b" or:

-   mkdir build
-   cd build
-   cmake .. -DCMAKE_INSTALL_PREFIX=../
-   make
-   make install
-   cd ..
-   g++ -g -o main.out main.cpp -Llib -lL2
