target:
	g++ KNN.cc
	./a.out 7 4000 8 input.txt
 
cu:
	nvcc KNN.cu
	./a.out 7 4000 8 input.txt