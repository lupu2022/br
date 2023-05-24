function clean {
    rm -rf install build
}

function kernels {
    if [ ! -d 'build' ]; then
        mkdir build
    fi
    
    cd build
    cmake ../kernels -DCMAKE_INSTALL_PREFIX=../install
    make
    make install
}

function engine {
    if [ ! -d 'build' ]; then
        mkdir build
    fi
    
    cd build
    cmake ../engine -DCMAKE_INSTALL_PREFIX=../install
    make
    make install
}


$1
