import numpy as np
import matplotlib.pyplot as plt
import random
import heapq as heap


#Read data from input file
def input_data(input_file):
	inp = open(input_file, mode = "r")

	#First line
	l = inp.readline().split(sep=',')
	max_X = int(l[0]) #Vertical bound
	max_Y = int(l[1]) #Horizontal bound

	#Second line
	l = inp.readline().split(sep=',')
	Vertex = [] #Aggregate of vertex in world
	Point = []
	for i in range(len(l)):		
		Point.append(int(l[i]))
		if i % 2 == 1:
			Vertex.append(Point)
			Point = []

	#Third line
	n_shape = int(inp.readline()) #Number of shapes in graph

	#The rest
	Shape_List = [] #Aggregate of shapes in world
	for i in range(n_shape):
		l = inp.readline().split(sep=',')
		Shape_Vertex = [] #Aggregate of vertex of a shape
		j = 0
		while j < len(l):			
			Point = (int(l[j]), int(l[j + 1]))
			Shape_Vertex.append(Point)
			j += 2
		Shape_List.append(Shape_Vertex)
	inp.close()
	return max_X, max_Y, Vertex, Shape_List

#Draw world and shapes
def create_world(max_X, max_Y, Vertex, Shape_List): 
	world = {}
	
	for i in range(0, max_X + 1):
		world[(i,0)] = {'visited':False, 'dist':np.inf, 'valid':False, 'F':np.inf}
		world[(i,max_Y)] = {'visited':False, 'dist':np.inf, 'valid':False, 'F':np.inf}

	for j in range(0, max_Y + 1):
		world[(0,j)] = {'visited':False, 'dist':np.inf, 'valid':False, 'F':np.inf}
		world[(max_X,j)] = {'visited':False, 'dist':np.inf, 'valid':False, 'F':np.inf}

	for i in range(1, max_X):
		for j in range(1, max_Y):
			world[(i,j)] = {'visited':False, 'dist':np.inf, 'valid':True, 'F':np.inf}

	for Shape_Vertex in Shape_List:
		for i in range(2):
			Start = Shape_Vertex[i]
			End = Shape_Vertex[i + 1]
			if i == len(Shape_Vertex) - 1: 
				End = Shape_Vertex[0]
			aStarAlg(world, Start, End, mode = 'edge')
			world[Start]['valid'] = False
	
	return world

def isValid(coordinate, world):
	#isValid determines if coordinates are valid for the given world
	for itr in coordinate: 
		#If outside the board
		if itr not in range(0, 34):
			return False
		for key in world.keys():
			if coordinate == key:
				if not world[key]['valid']:
					return False
	return True

def H(coordinate, target):
	#Function H represents the Heuristic value for A* 
	#The Heuristic Value = distance from a given node (or coordinate) to target 
	x_dist = coordinate[0] - target[0]
	y_dist = coordinate[1] - target[1]
	diag_dist = np.sqrt((x_dist)**2 + (y_dist)**2)

	return diag_dist

def aStarAlg(world, start, end, mode = "path"):
	#where world is given in Q2, start is the start, end is at finish
	
	q = []
	
	#Direction of new path
	direct = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

	#Heap with coordinates(x,y) and distance from the start
	heap.heappush(q, (0, start))

	world[start]['dist'] = 0	
	#Where F is the total distance (distance travelled + heuristic value)
	world[start]['F'] = 0

	while q != []:
		v = heap.heappop(q)
		coordinate = v[1]

		if coordinate == end:
			break

		#The code above is almost identical to Q2, below I update the code for Q3
		for direction in direct:
			new_coordinate = (direction[0] + coordinate[0], direction[1] + coordinate[1])
			if isValid(new_coordinate, world) or mode != 'path':
				if direction in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
					dist = world[coordinate]['dist'] + 1.5
					F = dist + H(new_coordinate, end)
				else:
					dist = world[coordinate]['dist'] + 1
					F = dist + H(new_coordinate, end)
				if F < world[new_coordinate]['F']:
					world[new_coordinate]['parent'] = coordinate
					world[new_coordinate]['dist'] = dist
					world[new_coordinate]['F'] = F
				if world[new_coordinate]['visited'] != True and coordinate != new_coordinate:
					world[new_coordinate]['visited'] = True
					heap.heappush(q, (world[new_coordinate]['F'], new_coordinate))

	#To find shortest path - backtrack through parents of nodes
	itr = end
	p = [end]
	shortest_p = world[end]['dist']

	while itr != start:
		itr = world[itr]['parent']
		p.append(itr)

	if mode != "path":
		for E in p:
			world[E]['valid'] = False

		#Reset visited state
		for i in range(max_X):
			for j in range(max_Y):
				world[(i,j)]['visited'] = False

	return shortest_p, p

def draw(world):
	#Create boundary
	boundary = []
	for key in world.keys():
		if key[0] in [0, max_X]:
			boundary.append(key)
		elif key[1] in [0, max_Y]:
			boundary.append(key)

	x_coord = []
	for key in boundary:
		x_coord.append(key[0])

	y_coord = []
	for key in boundary:
		y_coord.append(key[1])

	plt.axis([-1, max_X + 1, -1, max_Y + 1])
	plt.scatter(x_coord, y_coord, color = 'grey')

	#Create coords of shapes
	coordinates = []
	for key in world.keys():
		if world[key]['valid'] != True and key[0] not in (0, max_X) and key[1] not in (0, max_Y):
			coordinates.append(key)


	x_coord = []
	for keys in coordinates:
		x_coord.append(keys[0])

	y_coord = []
	for keys in coordinates:
		y_coord.append(keys[1])

	plt.scatter(x_coord, y_coord, color = 'magenta')

	

	#Initalize start and end points (from diagram in 2)
	start = Vertex.pop(0)
	end = Vertex.pop(0)

	plt.plot([start[0]], [start[1]], marker='o', color = 'green')
	plt.plot([end[0]], [end[1]], marker='o', color = 'red')

	plt.title("Finding a way from Start (Green point) to End (Red point)")
	plt.show()

def main():
	print()


if __name__ == '__main__':
	max_X, max_Y, Vertex, Shape_List = input_data("in.inp")
	world = create_world(max_X, max_Y, Vertex, Shape_List)
	draw(world)
