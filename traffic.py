import numpy as np
import csv
import matplotlib.pyplot as plt
import math as Math
import utm
from random import randint
from src import FullGP_RBF, SparseGP, SVGP
#from src import FullGP_RBF, SparseGP, SVGP, GRAPH_SEARCH, CS4246_SPECIAL_SEARCH
from StringIO import StringIO

#=======================DO NOT TOUCH THE CODE BELOW!!!================================
class Graph(object):
    def __init__(self, mapping={}):
        self.__graph_mapping = mapping

    def vertices(self):
        #ALL Nodes
        return list(self.__graph_mapping.keys())
    def add_vertex(self, vertex):
        if vertex not in self.__graph_mapping:
            self.__graph_mapping[vertex] = []

    def edges(self):
        #All Edges
        return self.__connect()
    def add_edge(self, edge):
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_mapping:
            self.__graph_mapping[vertex1].append(vertex2)
        else:
            self.__graph_mapping[vertex1] = [vertex2]

    def __connect(self):
        #private
        edges = []
        for vertex in self.__graph_mapping:
            for neighbour in self.__graph_mapping[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        #Stringlization, for debug
        result = "nodes: "

        for k in self.__graph_mapping:
            result += str(k) + " "

        result += "\nedges: "

        for edge in self.__connect():
            result += str(edge) + " "

        return result

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
graph = Graph(road_map)
i = 0

#Preprocess the data
Time = 8.0
X = []
Y = []
Z= []
with open('Raw-count-data-major-roads.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
    	if i>100000:
    		break
        i = i+1
        #print row
        if row[7]!=row[10] and float(row[17])==Time and (row[2]=="Derbyshire" or row[2]=="Derby"):
            print i
            lata, lona = OSGB36toWGS84(float(row[8]), float(row[9]))
            latb, lonb = OSGB36toWGS84(float(row[11]), float(row[12]))
        	#print utm.to_latlon(float(row[8]), float(row[9]), 30, 'U')
            all_nodes = graph.vertices()
            all_edges = graph.edges()

            #Code below avoids adding nodes repeatly on same position.
            if str(lona)+str(lata) not in all_nodes:
                graph.add_vertex(str(lona)+str(lata))
                #plt.plot([lona],[lata], 'go')
                X.append(lona)
                Y.append(lata)
                Z.append(row[7])
            if str(lonb)+str(latb) not in all_nodes:
                #plt.plot([lonb],[latb], 'go')
                graph.add_vertex(str(lonb)+str(latb))
                X.append(lonb)
                Y.append(latb)
                Z.append(row[10])

            if {str(lona)+str(lata),str(lonb)+str(latb)} not in all_edges:
                graph.add_edge({str(lona)+str(lata),str(lonb)+str(latb)})
                plt.plot([lona, lonb], [lata, latb], 'k-', lw=1)
            
plt.subplots_adjust(bottom = 0.1)
plt.scatter(X, Y)
"""
for x, y, z in zip(X, Y, Z):
    print x
    plt.annotate(
        str(x) + "," + str(y) + ":" + z, 
        xy = (x, y), xytext = (randint(-25,25), randint(-25,25)),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.4),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
"""
plt.show()
print graph.edges()

#GP here...but i dont know how to use custom kernal.
