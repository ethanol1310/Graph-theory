import numpy as np
import matplotlib.pyplot as plt
import random
import heapq as heap


class graph:

    def __init__(self):
        self.graph = {}
        self.visited = {}
        self.dist = {}
        self.F = {}
        self.parent = {}

    def append(self, vertexid, isPath):
        if vertexid not in self.graph.keys():
            self.graph[vertexid] = {}
            self.visited[vertexid] = False
            self.dist[vertexid] = np.inf
            self.F[vertexid] = np.inf
        self.graph[vertexid]['valid'] = isPath

    def reveal(self):
        return self.graph

    def vertex(self):
        return list(self.graph.keys())

    def infoVertex(self, vertexid):
        return list(self.graph[vertexid].keys())

    def isValid(self, vertexid):
        return self.graph[vertexid]['valid']

    def updateInvalid(self, vertexid):
        self.graph[vertexid]['valid'] = False

    def updateValid(self, vertexid):
        self.graph[vertexid]['valid'] = True

    def size(self):
        return len(self.graph)

    def getDist(self, vertexid):
        return self.dist[vertexid]

    def updateDist(self, vertexid, value):
        self.dist[vertexid] = value

    def getF(self, vertexid):
        return self.F[vertexid]

    def updateF(self, vertexid, value):
        self.F[vertexid] = value

    def getParent(self, vertexid):
        return self.parent[vertexid]

    def updateParent(self, vertexid, coordinate):
        self.parent[vertexid] = coordinate

    def visit(self, vertexid):
        self.visited[vertexid] = True

    def isVisited(self, vertexid):
        return self.visited[vertexid]

    def route(self):
        return self.visited


def draw():
    df = graph()
    world = {}
    for i in range(34):
        for j in range(34):
            world[(i, j)] = {'valid': True}
            df.append((i, j), world[(i, j)]['valid'])
            # Shape A

            if i in range(6, 13) and j in range(4, 11):
                if j >= 14 - i:
                    world[(i, j)]['valid'] = False
                    df.updateInvalid((i, j))

            # Shape C

            if i in range(14, 18) and j in range(11, 16):
                world[(i, j)]['valid'] = False
                df.updateInvalid((i, j))

            # Shape D

            if i in range(9, 13) and j in range(16, 21):
                world[(i, j)]['valid'] = False
                df.updateInvalid((i, j))

            # Shape E

            if i in range(18, 25) and j in range(16, 20):
                world[(i, j)]['valid'] = False
                df.updateInvalid((i, j))

            # Shape B

            if i in range(20, 29) and j in range(6, 20):
                if j <= 13 * i / 8 - 212 / 8:
                    world[(i, j)]['valid'] = False
                    df.updateInvalid((i, j))

            # Shape F

            if i in range(12, 29) and j in range(25, 29) or i in range(25, 29) \
                    and j in range(22, 26):
                world[(i, j)]['valid'] = False
                df.updateInvalid((i, j))

    return df


def isValid(coordinate, df):
    for itr in coordinate:

        # If outside the board

        if itr not in range(0, 34):
            return False
    return df.isValid(coordinate)


def dijAlg(df, start, end):

    # where world is given in Q2, start is the start, end is at finish

    q = []

    # Direction of new path

    direct = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    # Heap with coordinates(x,y) and distance from the start

    heap.heappush(q, (0, start))
    df.updateDist(start, 0)

    while q != []:
        v = heap.heappop(q)
        coordinate = v[1]

        # Identify nodes visited

        plt.scatter(coordinate[0], coordinate[1], marker='*',
                    color='.75')

        for direction in direct:
            new_coordinate = (
                direction[0] + coordinate[0], direction[1] + coordinate[1])
            if isValid(new_coordinate, df):
                if direction in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    dist = np.sqrt(2) + df.getDist(coordinate)
                else:
                    dist = 1 + df.getDist(coordinate)
                if dist < df.getDist(new_coordinate):
                    df.updateParent(new_coordinate, coordinate)
                    df.updateDist(new_coordinate, dist)
                if df.isVisited(new_coordinate) != True:
                    df.visit(new_coordinate)
                    heap.heappush(q, (df.getDist(new_coordinate),
                                      new_coordinate))

    # To find shortest path - backtrack through parents of nodes

    itr = end
    p = [end]
    shortest_p = df.getDist(end)

    while itr != start:
        itr = df.getParent(itr)
        p.append(itr)

    print 'Distance of the shortest path is %f' % shortest_p
    return shortest_p, p


def H(coordinate, target):
        # Function H represents the Heuristic value for A*
        # The Heuristic Value = distance from a given node (or coordinate) to target
    x_dist = coordinate[0] - target[0]
    y_dist = coordinate[1] - target[1]
    diag_dist = np.sqrt((x_dist)**2 + (y_dist)**2)

    return diag_dist


def aStarAlg(df, start, end):
    # where world is given in Q2, start is the start, end is at finish
    q = []

    # Direction of new path
    direct = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]

    # Heap with coordinates(x,y) and distance from the start
    heap.heappush(q, (0, start))

    df.updateDist(start, 0)
    # Where F is the total distance (distance travelled + heuristic value)
    df.updateF(start, 0)

    while q != []:
        v = heap.heappop(q)
        coordinate = v[1]

        if coordinate == end:
            break

        # Identify nodes visited
        plt.scatter(coordinate[0], coordinate[1], marker='*', color='.75')

        # The code above is almost identical to Q2, below I update the code for Q3
        for direction in direct:
            new_coordinate = (
                direction[0] + coordinate[0], direction[1] + coordinate[1])
            if isValid(new_coordinate, df):
                if direction in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    dist = np.sqrt(2) + df.getDist(coordinate)
                    F = dist + H(new_coordinate, end)
                else:
                    dist = df.getDist(coordinate) + 1
                    F = dist + H(new_coordinate, end)
                if F < df.getF(new_coordinate):
                    df.updateParent(new_coordinate, coordinate)
                    df.updateDist(new_coordinate, dist)
                    df.updateF(new_coordinate, F)
                if df.isVisited(new_coordinate) != True and coordinate != new_coordinate:
                    df.visit(new_coordinate)
                    heap.heappush(q, (df.getF(new_coordinate), new_coordinate))

    # To find shortest path - backtrack through parents of nodes
    itr = end
    p = [end]
    shortest_p = df.getDist(end)

    while itr != start:
        itr = df.getParent(itr)
        p.append(itr)

    print 'Distance of the shortest path is %f' % (shortest_p)
    return shortest_p, p


def bfs(df, start, end):
    q = []

    # Direction of new path
    direct = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]
    heap.heappush(q, start)

    df.updateDist(start, 0)
    df.updateF(start, 0)

    while q != []:
        v = heap.heappop(q)
        coordinate = v

        if coordinate == end:
            break
        plt.scatter(coordinate[0], coordinate[1], marker='*', color='.75')
        for direction in direct:
            new_coordinate = (
                direction[0] + coordinate[0], direction[1] + coordinate[1])
            if isValid(new_coordinate, df):
                if df.isVisited(new_coordinate) == False and new_coordinate not in q:
                    df.visit(new_coordinate)
                    df.updateParent(new_coordinate, coordinate)
                    heap.heappush(q, (new_coordinate))

    itr = end
    p = [end]

    while itr != start:
        itr = df.getParent(itr)
        p.append(itr)

    return p

def demoBfs(start, end):
    p = bfs(world, start, end)

    x_coord = [itr[0] for itr in p]
    y_coord = [itr[1] for itr in p]

    # Plot start and end points

    plt.plot([2], [2], marker='o', color='green')
    plt.plot([32], [32], marker='o', color='red')

    # Plot path

    plt.plot(x_coord, y_coord, color='cyan')
    plt.title("BFS's Algorithm")
    plt.show()

def demoUcs(start, end):
    shortest_p, p = dijAlg(world, start, end)

    x_coord = [itr[0] for itr in p]
    y_coord = [itr[1] for itr in p]

    # Plot start and end points

    plt.plot([2], [2], marker='o', color='green')
    plt.plot([32], [32], marker='o', color='red')

    # Plot path

    plt.plot(x_coord, y_coord, color='cyan')
    plt.title("Dijkstra's Algorithm")
    plt.show()

def demoAstar(start, end):
    shortest_p, p = aStarAlg(world, start, end)

    x_coord = [itr[0] for itr in p]
    y_coord = [itr[1] for itr in p]

    # Plot start and end points

    plt.plot([2], [2], marker='o', color='green')
    plt.plot([32], [32], marker='o', color='red')

    # Plot path

    plt.plot(x_coord, y_coord, color='cyan')
    plt.title("A star's Algorithm")
    plt.show()

if __name__ == '__main__':

    # Create the world

    world = draw()

    # Create coords

    coordinates = []
    for key in world.vertex():
        if world.isValid(key) != True:
            coordinates.append(key)

    x_coord = []
    for keys in coordinates:
        x_coord.append(keys[0])

    y_coord = []
    for keys in coordinates:
        y_coord.append(keys[1])

    # Plot world

    plt.axis([-2, 36, -2, 36])
    plt.scatter(x_coord, y_coord, color='magenta')

    # Initalize start and end points (from diagram in 2)

    start = (2, 2)
    end = (32, 32)

    #shortest_p, p = dijAlg(world, start, end)
    #shortest_p, p = aStarAlg(world, start, end)
    #p = bfs(world, start, end)
    demoBfs(start, end)