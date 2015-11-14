from __future__ import division
import numpy as np
np.set_printoptions(threshold=np.nan)
import csv
import sys
import heapq
import matplotlib.pyplot as plt
import math as Math
import utm
from random import randint
from src import FullGP_RBF, SparseGP, SVGP
import GPy
#from src import FullGP_RBF, SparseGP, SVGP, GRAPH_SEARCH, CS4246_SPECIAL_SEARCH
from StringIO import StringIO

from GPy.kern._src.custom_kern import RationalQuadratic as ratquadkern





#=======================DO NOT TOUCH THE CODE BELOW!!!================================
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.adjacent_updated_time = {}
        self.distance = sys.maxint      
        self.visited = False  
        self.previous = None

    def reset_node(self):
        self.distance = sys.maxint      
        self.visited = False  
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        if neighbor in self.adjacent.keys():
            updated_times = self.adjacent_updated_time[neighbor]
            self.adjacent[neighbor] = (1.0*(updated_times*self.adjacent[neighbor] + weight))/(1.0*(updated_times+1))
            self.adjacent_updated_time[neighbor] = updated_times + 1
            
        if neighbor not in self.adjacent.keys():
            self.adjacent[neighbor] = weight
            self.adjacent_updated_time[neighbor] = 1

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0
        self.node_list = []
        self.edge_list = []
        self.floydwarshall_node_matrix = []
        self.belief_state = []
        self.gp_weight = np.empty((0,1), dtype = np.int32)

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        self.node_list.append(node)
        return new_vertex
    def update_gp_weight(self, a):
        self.gp_weight = a

    def get_gp_weight(self, nameA, nameB):
        return self.gp_weight[self.get_edge_index(nameA, nameB)][0]

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None
    def get_edge_index(self, nameA, nameB):
        if [nameA,nameB] in self.edge_list:
            return self.edge_list.index([nameA,nameB])
        else:
            return self.edge_list.index([nameB,nameA])

    def has_edge(self, nameA, nameB):
        if [nameA,nameB] in self.edge_list:
            return True
        if [nameB,nameA] in self.edge_list:
            return True
        else:
            return False
    def initialize_node_matrix(self):
        self.floydwarshall_node_matrix = np.matrix(np.zeros([self.num_vertices,self.num_vertices]))
        self.floydwarshall_edge_matrix = np.matrix(np.zeros([len(self.edge_list),len(self.edge_list)]))

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        if frm in self.vert_dict or to in self.vert_dict:
            self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
            self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)
            if [frm,to] not in self.edge_list and [to, frm] not in self.edge_list:
                self.edge_list.append([frm,to])
            else:
                print "same!"


    def get_node_distance(self, frm, to):
        return self.get_node_distance_iter(frm, 0, to, [frm.get_id()])

    def get_node_distance_iter(self, current, depth, to, visited):
        if to in current.adjacent.keys():
                return depth + 1
        if to.get_id() == current.get_id():
                return depth
        else:
            ans = 9999999999
            for v in current.adjacent.keys():
                if v.get_id() not in visited:
                    visited.append(v.get_id())
                    t = self.get_node_distance_iter(v,depth+1,to,visited)
                    if t < ans:
                        ans = t
            return ans
    def all_distance(self):
        for v in self.vert_dict.keys():
            for m in self.vert_dict.keys():
                a = self.node_list.index(v)
                b = self.node_list.index(m)
                self.floydwarshall_node_matrix[a,b] = self.get_node_distance(self.get_vertex(v),self.get_vertex(m))
                self.floydwarshall_node_matrix[b,a] = self.floydwarshall_node_matrix[a,b]

    def edge_distance(self, nodeAA, nodeAB, nodeBA, nodeBB):
        aa = self.node_list.index(nodeAA.get_id())
        ab = self.node_list.index(nodeAB.get_id())
        ba = self.node_list.index(nodeBA.get_id())
        bb = self.node_list.index(nodeBB.get_id())
        ansA = self.floydwarshall_node_matrix[aa,ba]
        ansB = self.floydwarshall_node_matrix[aa,bb]
        ansC = self.floydwarshall_node_matrix[ab,ba]
        ansD = self.floydwarshall_node_matrix[ab,bb]
        return 1 + min(ansA,ansB,ansC,ansD)

    def build_edge_martix(self):
        for i in range(len(self.edge_list)):
            for j in range(len(self.edge_list)):
                self.floydwarshall_edge_matrix[i,j] = self.edge_distance(self.get_vertex(self.edge_list[i][0]), self.get_vertex(self.edge_list[i][1]), self.get_vertex(self.edge_list[j][0]), self.get_vertex(self.edge_list[j][1]))
                self.floydwarshall_edge_matrix[j,i] = self.floydwarshall_edge_matrix[i,j]
        return np.array(self.floydwarshall_edge_matrix)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous


    def delete_vertex(self, node):
        self.num_vertices = self.num_vertices - 1
        self.vert_dict = {key: value for key, value in self.vert_dict.items() if value != node}
    

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return

def dijkstra(aGraph, start, target):
    print '''Dijkstra's shortest path'''
    start.set_distance(0)
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()
        for next in current.adjacent:
            if next.visited:
                continue
            new_dist = current.get_distance() + aGraph.get_gp_weight(current.get_id(),next.get_id())
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)



#=======================DO NOT TOUCH THE CODE BELOW!!!================================
def OSGB36toWGS84(E,N):

    #E, N are the British national grid coordinates - eastings and northings
    pi = Math.pi
    a, b = 6377563.396, 6356256.909     #The Airy 180 semi-major and semi-minor axes used for OSGB36 (m)
    F0 = 0.9996012717                   #scale factor on the central meridian
    lat0 = 49*pi/180                    #Latitude of true origin (radians)
    lon0 = -2*pi/180                    #Longtitude of true origin and central meridian (radians)
    N0, E0 = -100000, 400000            #Northing & easting of true origin (m)
    e2 = 1 - (b*b)/(a*a)                #eccentricity squared
    n = (a-b)/(a+b)

    #Initialise the iterative variables
    lat,M = lat0, 0

    while N-N0-M >= 0.00001: #Accurate to 0.01mm
        lat = (N-N0-M)/(a*F0) + lat;
        M1 = (1 + n + (5./4)*n**2 + (5./4)*n**3) * (lat-lat0)
        M2 = (3*n + 3*n**2 + (21./8)*n**3) * Math.sin(lat-lat0) * Math.cos(lat+lat0)
        M3 = ((15./8)*n**2 + (15./8)*n**3) * Math.sin(2*(lat-lat0)) * Math.cos(2*(lat+lat0))
        M4 = (35./24)*n**3 * Math.sin(3*(lat-lat0)) * Math.cos(3*(lat+lat0))
        #meridional arc
        M = b * F0 * (M1 - M2 + M3 - M4)          

    #transverse radius of curvature
    nu = a*F0/Math.sqrt(1-e2*Math.sin(lat)**2)

    #meridional radius of curvature
    rho = a*F0*(1-e2)*(1-e2*Math.sin(lat)**2)**(-1.5)
    eta2 = nu/rho-1

    secLat = 1./Math.cos(lat)
    VII = Math.tan(lat)/(2*rho*nu)
    VIII = Math.tan(lat)/(24*rho*nu**3)*(5+3*Math.tan(lat)**2+eta2-9*Math.tan(lat)**2*eta2)
    IX = Math.tan(lat)/(720*rho*nu**5)*(61+90*Math.tan(lat)**2+45*Math.tan(lat)**4)
    X = secLat/nu
    XI = secLat/(6*nu**3)*(nu/rho+2*Math.tan(lat)**2)
    XII = secLat/(120*nu**5)*(5+28*Math.tan(lat)**2+24*Math.tan(lat)**4)
    XIIA = secLat/(5040*nu**7)*(61+662*Math.tan(lat)**2+1320*Math.tan(lat)**4+720*Math.tan(lat)**6)
    dE = E-E0
    #These are on the wrong ellipsoid currently: Airy1830. (Denoted by _1)
    lat_1 = lat - VII*dE**2 + VIII*dE**4 - IX*dE**6
    lon_1 = lon0 + X*dE - XI*dE**3 + XII*dE**5 - XIIA*dE**7
    #Want to convert to the GRS80 ellipsoid. 
    #First convert to cartesian from spherical polar coordinates
    H = 0 #Third spherical coord. 
    x_1 = (nu/F0 + H)*Math.cos(lat_1)*Math.cos(lon_1)
    y_1 = (nu/F0+ H)*Math.cos(lat_1)*Math.sin(lon_1)
    z_1 = ((1-e2)*nu/F0 +H)*Math.sin(lat_1)
    #Perform Helmut transform (to go between Airy 1830 (_1) and GRS80 (_2))
    s = -20.4894*10**-6 #The scale factor -1
    tx, ty, tz = 446.448, -125.157, + 542.060 #The translations along x,y,z axes respectively
    rxs,rys,rzs = 0.1502,  0.2470,  0.8421  #The rotations along x,y,z respectively, in seconds
    rx, ry, rz = rxs*pi/(180*3600.), rys*pi/(180*3600.), rzs*pi/(180*3600.)
    x_2 = tx + (1+s)*x_1 + (-rz)*y_1 + (ry)*z_1
    y_2 = ty + (rz)*x_1  + (1+s)*y_1 + (-rx)*z_1
    z_2 = tz + (-ry)*x_1 + (rx)*y_1 +  (1+s)*z_1  
    a_2, b_2 =6378137.000, 6356752.3141 #The GSR80 semi-major and semi-minor axes used for WGS84(m)
    e2_2 = 1- (b_2*b_2)/(a_2*a_2)   #The eccentricity of the GRS80 ellipsoid
    p = Math.sqrt(x_2**2 + y_2**2)
    lat = np.arctan2(z_2,(p*(1-e2_2))) #Initial value
    latold = 2*pi
    while abs(lat - latold)>10**-16: 
        lat, latold = latold, lat
        nu_2 = a_2/Math.sqrt(1-e2_2*Math.sin(latold)**2)
        lat = np.arctan2(z_2+e2_2*nu_2*Math.sin(latold), p)

    #Lon and height
    lon = np.arctan2(y_2,x_2)
    H = p/Math.cos(lat) - nu_2
    #Convert to degrees
    lat = lat*180/pi
    lon = lon*180/pi
    return lat, lon
#=======================DO NOT TOUCH THE CODE ABOVE!!!================================
road_map = {}
graph = Graph()
i = 0

#Preprocess the data
Time = "8"

with open('Raw-count-data-major-roads-derby.csv', 'rb') as f:
    reader = csv.reader(f)
    added_edges = []
    for row in reader:
    	if i > 10000000:
    		break
        i = i+1
        #print row
        try:
            if row[7]!=row[10] and row[17]==Time and (row[2]=="Derby" or row[2]=="Defrbyshire"):
                print i
                lata, lona = OSGB36toWGS84(float(row[8]), float(row[9]))
                latb, lonb = OSGB36toWGS84(float(row[11]), float(row[12]))
                a=(str(lata)+","+str(lona)+","+(row[8])+","+(row[9]))
                b=(str(latb)+","+str(lonb)+","+(row[11])+","+(row[12]))
                graph.add_edge(a,b,float(row[30]))
                if [a,b] not in added_edges:
                    plt.annotate(str(graph.get_edge_index(a,b))+"\n"+row[30], 
                    xy = ((lona+lonb)/2, (lata+latb)/2), xytext = (0, randint(-10,10)),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 1),
                    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
                added_edges.append([a,b])
                added_edges.append([b,a])
        except Exception:
            pass  # or use 'continue'
X = []
Y = []
XL = []
YL = []

i = 0
for v in graph:
    vid = v.get_id()
    X.append(float(vid.split(",")[1]))
    XL.append(float(vid.split(",")[3]))
    Y.append(float(vid.split(",")[0]))
    YL.append(float(vid.split(",")[2]))
    for w in v.get_connections():
        wid = w.get_id()
        plt.plot([float(vid.split(",")[1]), float(wid.split(",")[1])], [float(vid.split(",")[0]), float(wid.split(",")[0])], 'k-', lw=2, color='black')
"""
lata, lona = OSGB36toWGS84(438594, 333304)

latb, lonb = OSGB36toWGS84(431990, 335030)

a=(str(lata)+","+str(lona)+","+str(438594)+","+str(333304))
b=(str(latb)+","+str(lonb)+","+str(431990)+","+str(335030))
dijkstra(graph, graph.get_vertex(a), graph.get_vertex(b)) 

target = graph.get_vertex(b)
path = [target.get_id()]
shortest(target, path)
print path[::]
for s in range(len(path[::]) -1):
    fr = path[s]
    t = path[s+1]
    plt.plot([float(fr.split(",")[1]), float(t.split(",")[1])], [float(fr.split(",")[0]), float(t.split(",")[0])], 'k-', lw=3)  
    print graph.get_edge_index(fr,t)
"""
plt.subplots_adjust(bottom = 0.1)
plt.scatter(X, Y)

for x, y, xx, yy in zip(XL, YL, X, Y):
    plt.annotate(
        str(x) + "\n" + str(y), 
        xy = (xx, yy), xytext = (0, randint(-10,10)),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.2', fc = 'blue', alpha = 0.2),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))




lata, lona = OSGB36toWGS84(438594, 333304)
latb, lonb = OSGB36toWGS84(431990, 335030)

starting=(str(lata)+","+str(lona)+","+str(438594)+","+str(333304))
goal=(str(latb)+","+str(lonb)+","+str(431990)+","+str(335030))
found = False
current = starting
graph.initialize_node_matrix()
graph.all_distance()
edge_matrix_structure = graph.build_edge_martix()
result_prediction = np.empty((0,1), dtype = np.int32)
#print graph.edge_list
#print result_prediction
#print len(graph.edge_list)

for x in graph.edge_list:
    if [graph.get_edge_index(x[0],x[1])] not in result_prediction:
        result_prediction = np.append(result_prediction, np.array([ [graph.get_edge_index(x[0],x[1])]]), axis=0)
observed_edges = np.empty((0,1), dtype = np.int32)
observed_congestions = np.empty((0,1), dtype = np.int32)
ker1 = ratquadkern(1,edge_matrix_structure)
time = 0
#print current
print graph.edge_list
print ker1
print edge_matrix_structure
is_pos_def(x)
while not found:
    time = time + 1
    #print time
    #print current
    #print goal
    #observe
    for n in graph.get_vertex(current).get_connections():
        if [graph.get_edge_index(n.get_id(),current)] not in observed_edges:
            observed_edges = np.append(observed_edges, np.array([[graph.get_edge_index(n.get_id(),current)]]), axis=0)
            observed_congestions = np.append(observed_congestions, np.array([[n.get_weight(graph.get_vertex(current))]]), axis=0)

    #print observed_edges
    #print "======="
    #print observed_congestions
    #print result_prediction

    m = GPy.models.GPRegression(observed_edges,observed_congestions, ker1)
    m.optimize(messages=False)
    updated_weight = m.predict(result_prediction)[0]
    #print "=======!!!!!!!!!!!!!!!!!!!!!!!!!!!==============================="
    #print updated_weight.shape[0]
    for k in range (updated_weight.shape[0]):
        if updated_weight[k][0]<0:
            #print "negative"
            updated_weight[k][0] = 0

    graph.update_gp_weight(updated_weight)

    #print m.predict(result_prediction)[0]
    #print graph.gp_weight
    
    dijkstra(graph, graph.get_vertex(current), graph.get_vertex(goal)) 

    target = graph.get_vertex(goal)
    path = [target.get_id()]
    shortest(target, path)

    for s in range(len(path[::]) -1):
        fr = path[len(path[::])-s-1]
        t = path[len(path[::])-s-2]
        plt.plot([float(fr.split(",")[1]), float(t.split(",")[1])], [float(fr.split(",")[0]), float(t.split(",")[0])], 'k-', lw=7, color='red')  
    
    for n in graph.get_vertices():
        graph.get_vertex(n).reset_node()

    fr = current
    t = path[len(path[::])-2]
    plt.plot([float(fr.split(",")[1]), float(t.split(",")[1])], [float(fr.split(",")[0]), float(t.split(",")[0])], 'k-', lw=6, color='green') 
    current = path[len(path[::])-2]

    

    print current
    if current.split(",")[1] == goal.split(",")[1] and current.split(",")[0]== goal.split(",")[0]:
        break
    

plt.show()

















#GP here...but i dont know how to use custom kernal.
