function clean {
    rm -rf install build
}

function kernels {
    mkdir -p kernels/build
    cd kernels/build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../install
    make
    make install
}

function engine {
    mkdir -p engine/build 
    cd engine/build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../install
    make
    make install
}

function tokenizers {
    cd tokenizers.c
    make all
}


$1
