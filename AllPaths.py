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
		 'F': {'G':5},
         'G': {}}


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
	
"""print(findAllPaths(graph, 'A', 'G'))"""


def shortestPath(graph, start, end):
    paths = findAllPaths(graph, start, end);
    max = 99999999
    mpath=[]
    print('All paths from {} to {}: {}'.format(start,end,paths))
    for path in paths:
        t=sum(graph[i][j] for i,j in zip(path,path[1::]))
        print('\t\tevaluating: {}, cost: {}'.format(path, t))
        if t<max:
            max=t
            mpath=path
    bp=' '.join('{}->{}:{}'.format(i,j,graph[i][j]) for i,j in zip(mpath, mpath[1::]))
    tc=str(sum(graph[i][j] for i,j in zip(mpath,mpath[1::])))
    print('Best path: '+bp+' Total: '+tc+'\n')
	
"""shortestPath(graph,'A','G')"""


def dijkstra(graph, start, end, visited=[], distances={}, predecessors={}):
    if start not in graph:
        raise TypeError('the root of the shortest path tree cannot be found in the graph')
    if end not in graph:
        raise TypeError('the target of the shortest path cannot be found in the graph')
    if start == end:
        path=[]
        pred = end 
        while pred != None:
            path.append(pred)
            pred=predecessors.get(pred,None)
        print('Best path: '+str(path)+' Total:'+str(distances[end]))
    else :
        if not visited:
            distances[start]=0
        for neighbour in graph[start]:
            if neighbour not in visited:
                newDistance = distances[start] + graph[start][neighbour]
                if newDistance < distances.get(neighbour, float('inf')):
                    distances[neighbour] = newDistance
                    predecessors[neighbour] = start
        visited.append(start)
        unvisited={}
        for v in graph:
            if v not in visited:
                unvisited[v] = distances.get(v,float('inf'))
        vertex = min(unvisited, key=unvisited.get)
        dijkstra(graph, vertex, end, visited, distances, predecessors)

dijkstra(graph, 'A', 'G')                    
