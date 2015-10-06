import point_matrix_transform
import numpy as np
import sys

X=[]
Y=[]

offset_lat =  -89.5
offset_lon = -179.5

input_file = open("Dec1_2012.csv","r")
counter = 0
val = {}
for line in input_file:
	if not line:
		break
	contents = line.split(",")

	lat = offset_lat + counter
	for idx, value in enumerate(contents):
		lon = offset_lon + idx
		if (contents[idx].strip() != "NaN"):
			# print contents[idx]
			# break
			X.append([counter,idx])
			Y.append([float(contents[idx].strip())])
			val[(lat,lon)] = float(contents[idx].strip())

	counter += 1

X = np.array(X)
Y = np.array(Y)

# print val[(-56.5,-179.5)]
# print val[ point_matrix_transform.index_to_point( (33,0) ) ]
newlong = 144.5
newlat = 40.5
# newlong = -37.5
# newlat = 27.5
def getnadjacentpoints(newlat,newlong,n):
	''' returns atmost n*n adjacent points around the point. give odd n '''
	data = {}
	rangelat = np.linspace(newlat+n/2,newlat-n/2,n)
	rangelong = np.linspace(newlong+n/2,newlong-n/2,n)
	print rangelat
	print rangelong
	for i in rangelat:
		for j in rangelong:
			if val.get( (i,j), None) != None:
				data [ (i,j) ] = val.get( (i,j) )						
			else:
				print 'missing value'
	return data

points = getnadjacentpoints(newlat,newlong,5)
print points
print len(points)
