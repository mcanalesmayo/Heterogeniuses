
echo "---------------------------------------------------------"
./kmeans_openmp/kmeans -n 8 -k 10 -i dataset/scale-d-10d.csv
./kmeans_openmp/kmeans -n 8 -k 100 -i dataset/scale-d-80d.csv
./kmeans_openmp/kmeans -n 8 -k 1000 -i dataset/scale-d-640d.csv

echo "---------------------------------------------------------"
./kmeans_openmp/kmeans -n 8 -k 10 -i dataset/scale-d-10d.csv
./kmeans_openmp/kmeans -n 8 -k 100 -i dataset/scale-d-80d.csv
./kmeans_openmp/kmeans -n 8 -k 1000 -i dataset/scale-d-640d.csv

echo "---------------------------------------------------------"
./kmeans_openmp/kmeans -n 8 -k 10 -i dataset/scale-d-10d.csv
./kmeans_openmp/kmeans -n 8 -k 100 -i dataset/scale-d-80d.csv
./kmeans_openmp/kmeans -n 8 -k 1000 -i dataset/scale-d-640d.csv