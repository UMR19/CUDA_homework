#!/usr/bin/python3
 
import re
import sys
import random
import os
 
filename = "input.txt"

if(os.path.exists(filename)):
	print("%s exists and del" % filename)
	os.remove(filename)
 
fout = open(filename,"w")
 
for i in range( 0,int(sys.argv[1]) ): #str to int
	x = []
	for j in range(0,int(sys.argv[2])):
		#generate random data and limit the digits into 4
		x.append( "%4f" % random.uniform(-1,1) ) 
		fout.write("%s\t" % x[j])
		#fout.write(x) : TypeError:expected a character buffer object 
 
	if(x[0][0] == '-'):
		fout.write(" Negative"+"\n")
	else:
		fout.write(" Positive"+"\n")
 
fout.close()