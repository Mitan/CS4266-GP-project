# input: a tuple (x-coordinate,y-coordinate)
#output: an index in the traning matrix
def point_to_matrix(point):
	return (point[0]+89.5,point[1]+179.5)
def index_to_point(index):
	x_coordinate = index[0] - 89.5
	y_coordinate = index[1] - 179.5
	return (x_coordinate,y_coordinate)

def test(function,args,desired_output):
	if function(args) != desired_output:
		print 'test failed'
		return
	print 'test succeeded'

test(index_to_point,(0,0),(-89.5,-179.5))
test(index_to_point,(179,359),(89.5,179.5))

