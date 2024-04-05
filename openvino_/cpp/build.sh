set -e

mkdir -p build
cd build

NUM_THREADS=8

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/lijingyi/code/PitchVC/openvino_/cpp/libtorch ..
cmake --build . -- -j $NUM_THREADS

cd ..