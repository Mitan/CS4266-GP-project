def point_to_index(point): 
	''' 
	input: a tuple . point[0] -> latitude , point[1] -> longitude
	returns index of the matrix representing the data
	latitude range: (-89.5,89.5) longitude range: (-179.5,179.5)
    usage: point_to_index((-51.5,-175.5))
      '''
	return (point[0]+89.5,point[1]+179.5)

def index_to_point(index):
	''' 
	returns (latitude,longitude) tuple 

	usage: index_to_point((38,4)). #Here, 38-> row, 4 -> column
	'''

	x_coordinate = index[0] - 89.5
	y_coordinate = index[1] - 179.5
	return (x_coordinate,y_coordinate)

def test(function,args,desired_output):
	if function(args) != desired_output:
		print 'test failed'
		return
	print 'test succeeded'

if __name__ == '__main__':
	test(index_to_point,(0,0),(-89.5,-179.5))
	test(index_to_point,(179,359),(89.5,179.5))
	test(index_to_point,(33,0),(-56.5,-179.5))
	test(point_to_index,  (-51.5,-175.5),(38,4),)
