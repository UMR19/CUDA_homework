#include<iostream>
#include<map>
#include<vector>
#include<stdio.h>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<fstream>
 
using namespace std;
 
typedef string tLabel;
typedef float tData;
typedef pair<int,double>  PAIR;
const int MaxColLen = 10;
const int MaxRowLen = 10010;
const int test_data_num = 400;
ifstream fin;

float gpu_time = 0.0;
class KNN
{
private:
		tData dataSet[MaxRowLen][MaxColLen];
		tLabel labels[MaxRowLen];
		tData testData[MaxColLen];
		tData trainingData[3600][8];
		int rowLen;
		int colLen;
		int k;
		map<int,double> map_index_dis;
		map<tLabel,int> map_label_freq;
		double get_distance(tData *d1,tData *d2);
public:
		KNN(int k , int rowLen , int colLen , char *filename);
		void get_all_distance();
		tLabel get_max_freq_label();
		void auto_norm_data();
		void get_error_rate();
		void get_training_data();
		struct CmpByValue
		{
			bool operator() (const PAIR& lhs,const PAIR& rhs)
			{
				return lhs.second < rhs.second;
			}
		};
 
		~KNN();	
};
 
KNN::~KNN()
{
	fin.close();
	map_index_dis.clear();
	map_label_freq.clear();
}
 
KNN::KNN(int k , int row ,int col , char *filename)
{
	this->rowLen = row;
	this->colLen = col;
	this->k = k;
	
	fin.open(filename);
 
	if( !fin )
	{
		cout<<"can not open the file"<<endl;
		exit(0);
	}
 
	for(int i=0;i<rowLen;i++)
	{
		for(int j=0;j<colLen;j++)
		{
			fin>>dataSet[i][j];
		}
		fin>>labels[i];
	}
 
}
 
void KNN:: get_training_data()
{
	for(int i=test_data_num;i<rowLen;i++)
	{
		for(int j=0;j<colLen;j++)
		{
			trainingData[i-test_data_num][j] = dataSet[i][j];
		}
	}
}
 
void KNN:: get_error_rate()
{
	int i,j,count = 0;
	tLabel label;
 
	cout<<"the test data number is : "<<test_data_num<<endl;
 
	get_training_data();
 
	//get testing data and calculate
	for(i=0;i<test_data_num;i++)
	{
		for(j=0;j<colLen;j++)
		{
			testData[j] = dataSet[i][j];		
		}
		
		get_all_distance();
		label = get_max_freq_label();
		if( label!=labels[i] )
			count++;
		map_index_dis.clear();
		map_label_freq.clear();
	}
	cout<<"the error rate is = "<<(double)count/(double)test_data_num<<endl;
}
 
//global function
__global__ void cal_dis(tData *train_data,tData *test_data,tData* dis,int pitch,int N , int D)
{
	int tid = blockIdx.x;
	if(tid<N)
	{
		tData temp = 0;
		tData sum = 0;
		for(int i=0;i<D;i++)
		{
			temp = *( (tData*)( (char*)train_data+tid*pitch  )+i ) - test_data[i];
			sum += temp * temp;
		}
		dis[tid] = sum;
	}
}
 
//Parallel calculate the distance
void KNN:: get_all_distance()
{
	// cal GPU run time
	// cudaEvent_t gpu_start, gpu_stop;
	// cudaEventCreate(&gpu_start);
	// cudaEventCreate(&gpu_stop);
	// cudaEventRecord(gpu_start, 0);

	int height = rowLen - test_data_num;
	tData *distance = new tData[height];
	tData *d_train_data,*d_test_data,*d_dis;
	size_t pitch_d ;
	size_t pitch_h = colLen * sizeof(tData);
	//allocate memory on GPU
	cudaMallocPitch( &d_train_data,&pitch_d,colLen*sizeof(tData),height);
	cudaMalloc( &d_test_data,colLen*sizeof(tData) );
	cudaMalloc( &d_dis, height*sizeof(tData) );
 
	cudaMemset( d_train_data,0,height*colLen*sizeof(tData) );
	cudaMemset( d_test_data,0,colLen*sizeof(tData) );
	cudaMemset( d_dis , 0 , height*sizeof(tData) );
 
	//copy training and testing data from host to device
	cudaMemcpy2D( d_train_data,pitch_d,trainingData,pitch_h,colLen*sizeof(tData),height,cudaMemcpyHostToDevice);
	cudaMemcpy( d_test_data,testData,colLen*sizeof(tData),cudaMemcpyHostToDevice);
	//calculate the distance
	cal_dis<<<height,1>>>( d_train_data,d_test_data,d_dis,pitch_d,height,colLen );
	//copy distance data from device to host
	cudaMemcpy( distance,d_dis,height*sizeof(tData),cudaMemcpyDeviceToHost);
	
	// cudaEventRecord(gpu_stop, 0);
	// cudaEventSynchronize(gpu_stop);

	// float timestamp;
	// cudaEventElapsedTime(&timestamp, gpu_start, gpu_stop);
	// gpu_time += timestamp;

	// cudaEventDestroy(gpu_start);
	// cudaEventDestroy(gpu_stop);
	int i;
	for( i=0;i<rowLen-test_data_num;i++ )
	{
		map_index_dis[i+test_data_num] = distance[i];
	}
 
}
 
tLabel KNN:: get_max_freq_label()
{
	vector<PAIR> vec_index_dis( map_index_dis.begin(),map_index_dis.end() );
	sort(vec_index_dis.begin(),vec_index_dis.end(),CmpByValue());
 
	for(int i=0;i<k;i++)
	{
		/*
		cout<<"the index = "<<vec_index_dis[i].first<<" the distance = "<<vec_index_dis[i].second<<" the label = "<<labels[ vec_index_dis[i].first ]<<" the coordinate ( ";
		int j;
		for(j=0;j<colLen-1;j++)
		{
			cout<<dataSet[ vec_index_dis[i].first ][j]<<",";
		}
		cout<<dataSet[ vec_index_dis[i].first ][j]<<" )"<<endl;
		*/
		map_label_freq[ labels[ vec_index_dis[i].first ]  ]++;
	}
 
	map<tLabel,int>::const_iterator map_it = map_label_freq.begin();
	tLabel label;
	int max_freq = 0;
	while( map_it != map_label_freq.end() )
	{
		if( map_it->second > max_freq )
		{
			max_freq = map_it->second;
			label = map_it->first;
		}
		map_it++;
	}
	// cout<<"The test data belongs to the "<<label<<" label"<<endl;
	return label;
}
 
void KNN::auto_norm_data()
{
	tData maxa[colLen] ;
	tData mina[colLen] ;
	tData range[colLen] ;
	int i,j;
 
	for(i=0;i<colLen;i++)
	{
		maxa[i] = max(dataSet[0][i],dataSet[1][i]);
		mina[i] = min(dataSet[0][i],dataSet[1][i]);
	}
 
	for(i=2;i<rowLen;i++)
	{
		for(j=0;j<colLen;j++)
		{
			if( dataSet[i][j]>maxa[j] )
			{
				maxa[j] = dataSet[i][j];
			}
			else if( dataSet[i][j]<mina[j] )
			{
				mina[j] = dataSet[i][j];
			}
		}
	}
 
	for(i=0;i<colLen;i++)
	{
		range[i] = maxa[i] - mina[i] ; 
		//normalize the test data set
		testData[i] = ( testData[i] - mina[i] )/range[i] ;
	}
 
	//normalize the training data set
	for(i=0;i<rowLen;i++)
	{
		for(j=0;j<colLen;j++)
		{
			dataSet[i][j] = ( dataSet[i][j] - mina[j] )/range[j];
		}
	}
}
 
int main(int argc , char** argv)
{
	int k,row,col;
	char *filename;
	
	if( argc!=5 )
	{
		cout<<"The input should be like this : ./a.out k row col filename"<<endl;
		exit(1);
	}
 
	k = atoi(argv[1]);
	row = atoi(argv[2]);
	col = atoi(argv[3]);
	filename = argv[4];
 
	KNN knn(k,row,col,filename);

	knn.auto_norm_data();
	knn.get_error_rate();

	printf("GPU:Time used:%.2f ms\n", gpu_time);
	return 0;
}