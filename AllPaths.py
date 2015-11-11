graphOld = {'A': ['B', 'C'],
         'B': ['C', 'D'],
         'C': ['D'],
         'D': ['C'],
         'E': ['F'],
         'F': ['C']}
		 
graph = {'A': {'B':10},
         'B': {'C':20, 'D':30},
         'C': {'D':15,'F':80},
         'D': {'C':10,'E':2,'F':10},		 
		 'E': {},
		 'F': {'G':5}}
		 
def findAllPaths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not start in graph.keys():
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = findAllPaths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths
	
print(findAllPaths(graph, 'A', 'G'))

def shortestPath(graph, start, end):
    paths = findAllPaths(graph, start, end);
    max = 99999999
    mpath=[]
    print('\tAll paths from {} to {}: {}'.format(start,end,paths))
    for path in paths:
        t=sum(graph[i][j] for i,j in zip(path,path[1::]))
        print('\t\tevaluating: {}, cost: {}'.format(path, t))
        if t<max:
            max=t
            mpath=path
    bp=' '.join('{}->{}:{}'.format(i,j,graph[i][j]) for i,j in zip(mpath, mpath[1::]))
    tc=str(sum(graph[i][j] for i,j in zip(mpath,mpath[1::])))
    print('Best path: '+bp+' Total: '+tc+'\n')
	
shortestPath(graph,'A','G')
	