import sys; args = sys.argv[1:]
import re
import math

def defaultGraph(size, width = 0, rwd = 12, type ='N'):
    return {0:None, "size":size, "rwd":rwd, "width": width, "type" : type}

def grfParse(args):
    size = 0
    rwd = '12'
    width = 0
    if(m:= re.search(r"^G([GN]?)(\d+)(W(\d+))?(R(\d+))?", args[0])):
        size = int(m.group(2))
        width = m.group(4)
        if width == None:
            w = math.ceil(math.sqrt(size))
            while size % w != 0:
                w += 1
            width = w
        rwd = m.group(6) 
        if rwd == None:
            rwd = 12
        if m.group(1) == 'N':
            return defaultGraph(size, int(width), rwd)
        if int(width) == 0:
            return defaultGraph(size, 0, rwd, 'G')
    graph = {i: None for i in range(0, int(size))}
    graph["size"] = int(size)
    graph["rwd"] = int(rwd)
    graph["width"] = int(width)
    graph["type"] = "G"
    if(len(args) > 1):
        if(m:= re.search(r"^V([-0-9,:]+)((B)|(R\d*))*$", args[1])):
            vertices = []
            vslcs_list = m.group(1).split(',')
            for vslc in vslcs_list:
                if '::' in vslc:
                    start, step = map(int, vslc.split('::'))
                    if start < 0:
                        start += size
                    vertices.extend(list(range(start, size, step)))
                elif ':' in vslc:
                    parts = vslc.split(':')
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if parts[1] else size
                    if start < 0:
                        start += size
                    if end < 0:
                        end += size
                    vertices.extend(list(range(start, end)))
                else:
                    vertex = int(vslc)
                    if vertex < 0:
                        vertex += size
                    vertices.append(vertex)
            # graph = processV(graph, vertices)
            for vertex in vertices:
                graph[vertex] = 'B'
        # print(vertices)
    # print(graph)
    return graph


def grfSize(graph):
    print(type(graph["size"]))
    return graph["size"]

# def processV(graph, vertices):
#     for vertex in vertices:
#         graph[vertex] = 'B'
#     return graph

def grfNbrs(graph, v):

    size = graph['size']
    width = graph['width']
    type = graph['type']

    if type == 'N' or width == 0:
        return []

    if v < 0 or v >= size:
        return "Invalid v. Please provide an v between 0 and {}.".format(size-1)
    neighbors = []
    # left neighbor
    if graph[v] == 'B':
            return neighbors
    if v % width > 0 and graph[v - 1] != 'B':
        neighbors.append(v - 1)
    # right neighbor
    if v % width < width - 1 and graph[v + 1] != 'B':
        neighbors.append(v + 1)
    # upper neighbor
    if v >= width and graph[v - width] != 'B':
        neighbors.append(v - width)
    # lower neighbor
    if v < size - width and graph[v + width] != 'B':
        neighbors.append(v + width)
    # print(neighbors)
    
    return neighbors

def grfGProps(graph):
    if graph['type'] == 'N':
        return {"rwd":graph['rwd']}
    return {"rwd":graph['rwd'],"width":graph['width']}

def grfVProps(graph, v):
    return {}
def grfEProps(graph, v1, v2):
    return {}

def grfStrEdges(graph):
    directions = {
        'J': 'NW', 'N': 'N', '^': 'WNE', 'L': 'NE', '<': 'NWS',
        'W': 'W', '+': 'NEWS', 'E': 'E', '>': 'NES', '7': 'SW',
        'v': 'WSE', 'S': 'S', 'r': 'SE', '-': 'EW', '|': 'NS',
        '.': '', '*': 'NEWS'  # Assuming '*' has all neighbors
    }
    size = graph['size']
    width = graph['width']
    type = graph['type']

    if type == 'N' or width == 0:
        return ""
    
    def get_neighbors(index):
        neighbors = ''
        # left neighbor
        if graph[index] == 'B':
            return neighbors
        if index // width == (index - 1) // width and graph[index - 1] != 'B':
            neighbors += 'W'
        # right neighbor
        if index // width == (index + 1) // width and graph[index + 1] != 'B':
            neighbors += 'E'
        # upper neighbor
        if index >= width and graph[index - width] != 'B':
            neighbors += 'N'
        # lower neighbor
        if index < size - width and graph[index + width] != 'B':
            neighbors += 'S'
        return neighbors

    graph_string = ''
    for i in range(size):
    # Check if the current index is a barrier
        if graph[i] == 'B':
            graph_string += '.'
        else:
            neighbors = get_neighbors(i)
            # Find the character that matches the neighbors
            for char, dirs in directions.items():
                if set(neighbors) == set(dirs):
                    graph_string += char
                    break

    return graph_string

def grfStrProps(graph):
    return ', '.join(f"'{k}': {v}" for k, v in grfGProps(graph).items())

def main():
    graph = grfParse(['G8'])
    
    # graph = grfParse(args)
    size = grfSize(graph)
    print(grfNbrs(graph, 0))
    edgesStr = grfStrEdges(graph)
    propsStr = grfStrProps(graph)
    # print(grfGProps(graph))
    # print(edgesStr)
    for i in range(0, len(edgesStr), graph['width']):
        print(edgesStr[i:i+graph['width']])
    print(propsStr)

if __name__ == '__main__': main()

# if(m:= re.search(r"^G([GN]?)(\d+)(W(\d+))?", args)):
# python
# import graphs as g
# grf = g.grfParse(['GG15'])
# g.grfSite(grf)
# g.grfNbrs(grf, 1)
#Om Gole, 6, 2024