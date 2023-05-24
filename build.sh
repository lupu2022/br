if [ ! -d 'build' ]; then
    mkdir build
fi

cd build 
cmake -DCMAKE_INSTALL_PREFIX=../install ../kernels && make && make install
rm -f Makefile CMakeCache.txt
cmake -DCMAKE_INSTALL_PREFIX=../install ../engine && make && make install

