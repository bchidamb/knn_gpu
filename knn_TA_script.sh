make clean
make knn
./knn 1 2 1
if [ -d "ml-20m" ]; then
    ./knn 0 1 1 3 1
fi