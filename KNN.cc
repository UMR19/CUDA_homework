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
typedef double tData;
typedef pair<int,double>  PAIR;
const int MaxColLen = 10;
const int MaxRowLen = 10000;
ifstream fin;
 
class KNN
{
private:
		tData dataSet[MaxRowLen][MaxColLen];
		tLabel labels[MaxRowLen];
		tData testData[MaxColLen];
		int rowLen;
		int colLen;
		int k;
		int test_data_num;
		map<int,double> map_index_dis;
		map<tLabel,int> map_label_freq;
		double get_distance(tData *d1,tData *d2);
public:
		KNN(int k , int rowLen , int colLen , char *filename);
		void get_all_distance();
		tLabel get_max_freq_label();
		void auto_norm_data();
		void get_error_rate();
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
	test_data_num = 0;
	
	fin.open(filename);
 
	if( !fin )
	{
		cout<<"can not open the file"<<endl;
		exit(0);
	}
	
	//read data from file
	for(int i=0;i<rowLen;i++)
	{
		for(int j=0;j<colLen;j++)
		{
			fin>>dataSet[i][j];
		}
		fin>>labels[i];
	}
 
}
 
void KNN:: get_error_rate()
{
	int i,j,count = 0;
	tLabel label;
	cout<<"please input the number of test data : "<<endl;
	cin>>test_data_num;
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
 
double KNN:: get_distance(tData *d1,tData *d2)
{
	double sum = 0;
	for(int i=0;i<colLen;i++)
	{
		sum += pow( (d1[i]-d2[i]) , 2 );
	}
 
	//cout<<"the sum is = "<<sum<<endl;
	return sqrt(sum);
}
 
//get distance between testData and all dataSet
void KNN:: get_all_distance()
{
	double distance;
	int i;
	for(i=test_data_num;i<rowLen;i++)
	{
		distance = get_distance(dataSet[i],testData);
		map_index_dis[i] = distance;
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
	//cout<<"The test data belongs to the "<<label<<" label"<<endl;
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

	clock_t begin = clock();
	KNN knn(k,row,col,filename);

	knn.auto_norm_data();
	knn.get_error_rate();
	clock_t end = clock();
	double time_spent = (double)(end -begin);
	printf("CPU:Time used:%.2f ms\n", time_spent);
 
	return 0;
}