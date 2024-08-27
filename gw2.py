import sys; args = sys.argv[1:]
import re
import math

#Solve with BFS

def grfParse(args):
    #default props
    graph = {}
    nbrs, nbrsStr, vprops, eprops, jumps, width, dRwd, size, type = [], [], [], {}, [], 0, None, 0, "N"
    # gdir = None
    # vdir = None
    # edir = None
    # if len(args) > 1:
    #     vdir = args[1]
    # if len(args) > 2:
    #     edir = args[2]
    for arg in args:
        gdir = None
        vdir = None
        edir = None
        if "G" in arg:
            gdir = arg
        if "V" in arg:
            vdir = arg
        if "E" in arg:
            edir = arg
    #GDIRECTIVE
        if gdir != None:
            if (m := re.search(r"^G([GN]?)(\d+)(W(\d+))?(R(\d+))?", gdir)):
                type = m.group(1) or "G"
                size = int(m.group(2))
                if m.group(4): 
                    width = int(m.group(4))
                else:
                    width = getWidth(size)
                dRwd = int(m.group(6)) if m.group(6) else 12
                # print(size)
                # print(type)
                # print(width)


            if type == "N":
                graph = {"props":{"rwd":dRwd}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vprops}
                return graph

            if width == 0:
                graph = {"props":{"rwd":dRwd, "width": width}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vprops}
                return graph 
            for vertex in range(0, size):
                temp_nbrs = set()
                if vertex - 1 >= 0 and (vertex - 1)//width == vertex//width:
                    temp_nbrs.add(vertex - 1)
                if vertex + 1 < size and (vertex + 1)//width == vertex//width:
                    temp_nbrs.add(vertex + 1)
                if vertex + width < size and (vertex + width)%width == vertex%width:
                    temp_nbrs.add(vertex + width)
                if vertex - width >= 0 and (vertex - width)%width == vertex%width:
                    temp_nbrs.add(vertex - width)
                nbrs.append(temp_nbrs)
                for n in temp_nbrs:
                    eprops[(vertex, n)] = {"rwd": 0}
                nbrsStr.append(getChar(vertex, width, temp_nbrs))
                vprops.append({})
        #VDIRECTIVE 
        if vdir != None:
            parsed_slice_instructions = re.findall("((-?\d*:*-?\d*,?)+)", vdir[1:])[0][0]
            block_directive = 'B' in vdir[1:]
            new_reward = re.findall("R(\d*)", vdir)[0] if 'R' in vdir else None #DEFAULT V SLC REWARD!
            wSet = set(sliceParse(parsed_slice_instructions, size))
            if block_directive: 
                for v in wSet:
                    tempp_neighbors = nbrs[v]
                    temp_nbrs = set()
                    if v - 1 >= 0 and (v - 1)//width == v//width:
                        temp_nbrs.add(v - 1)
                    if v + 1 < size and (v + 1)//width == v//width:
                        temp_nbrs.add(v + 1)
                    if v + width < size and (v + width)%width == v%width:
                        temp_nbrs.add(v + width)
                    if v - width >= 0 and (v - width)%width == v%width:
                        temp_nbrs.add(v - width)
                    allNbrs = tempp_neighbors.union(temp_nbrs)
                    # Update tempp_neighbors
                    tempp_neighbors = allNbrs - tempp_neighbors
                    for n in allNbrs:
                        curr_nbrs_n = nbrs[n]
                        curr_nbrs_n ^= {v}
                        nbrs[n] = curr_nbrs_n
                        nbrsStr[n] = getChar(n, width, curr_nbrs_n)
                    nbrs[v] = tempp_neighbors
                    nbrsStr[v] = getChar(v, width, tempp_neighbors)
            if new_reward:
                for v in wSet:
                    vprops[v] = {"rwd":new_reward}
        #EDGE DIRECTIVE
        if edir != None:
            edgeargs = re.findall("(!|\+|\*|~|@)", arg)[0] if re.findall("(!|\+|\*|~|@)", arg) else "~"
            vs1 = re.findall("([-:\,\d]+)(?=[NSEW=~])", edir[1:])[0]
            vs2 = re.findall("([-:\,\d]*)(?!.*[~=])", edir[1:])[0]
            dirtype = re.findall("[=~]", edir[1:])[-1]
            edgerwd = re.findall("R(\d*)", edir)[0] if re.findall("R(\d*)", edir) else dRwd
            vLst1 = sliceParse(vs1, size)
            nesw = re.findall("(W|E|S|N)", arg[1:]) if not vs2 else []
            nedges = [(v, edgeAdj(v, width, size//width, direction)) for v in vLst1 for direction in nesw if edgeAdj(v, width, size//width, direction) != "."]
            nedges = nedges if nesw else [(vLst1[i], sliceParse(vs2, size)[i]) for i in range(len(vLst1))]

            if dirtype == "=":
                nedges += [(tup[1], tup[0]) for tup in nedges]
            #GOOD TILL HERE
            for edge in nedges:
                    begin, finish = edge[0], edge[1]
                    curr_nbrs = nbrs[begin]
                    if edgeargs == "~":
                        if finish in curr_nbrs:
                            curr_nbrs.remove(finish)
                        elif finish not in findNeighbors(begin, size, width):
                            jumps.append((begin, finish, dirtype))
                            curr_nbrs.add(finish)
                    elif edgeargs == "@":
                        if edgerwd:
                          eprops[(begin, finish)] = {"rwd":edgerwd}

                    elif edgeargs == "*":
                        if finish not in (curr_nbrs | findNeighbors(begin, size, width)):
                            jumps.append((begin, finish, dirtype))
                            curr_nbrs.add(finish)

                        if edgerwd:
                            eprops[(begin, finish)] = {"rwd":edgerwd}

                    elif edgeargs == "!":
                        curr_nbrs.discard(finish)
                        eprops.pop((begin, finish), None)
                    elif edgeargs == "+":
                        if finish not in curr_nbrs and finish not in findNeighbors(begin, size, width):
                            jumps.append((begin, finish, dirtype))
                            curr_nbrs.add(finish)
                        if edgerwd and (begin, finish) not in eprops:
                            eprops[(begin, finish)] = {"rwd": edgerwd}

                        eprops.pop((begin, finish), None)
                        if edgerwd:
                            eprops[(begin, finish)] = {"rwd":edgerwd}


                    nbrs[begin] = curr_nbrs
                    nbrsStr[begin] = getChar(begin, width, curr_nbrs)

        graph = {"props":{"rwd":dRwd, "width": width}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vprops, "eProps": eprops, "jumps": jumps}
    return graph


def grfSize(graph):
    # print(type(str(graph["size"])))
    # number = str(graph["size"])
    # return int(number)
    return graph["size"]

def grfNbrs(graph, v):  
    try:
        if graph["nbrs"]:
            return graph["nbrs"][v]
    except KeyError:
        print("KeyError: The key does not exist in the graph.")
    return {}


def grfGProps(graph):
    return graph["props"]

def grfVProps(graph, v):
    # return graph["vProps"][v] if graph.get("vProps") and graph["vProps"].get(v) else {}
    if graph["vProps"] and graph["vProps"][v] != 0:
        return graph["vProps"][v]
    else:
        return {}

def grfEProps(graph, v1, v2):
    return graph["eProps"].get((v1, v2), {}) if graph["eProps"].get((v1, v2), {}).get("rwd", 0) != 0 else {}


def grfStrEdges(input_graph):
    neighbors_string = "".join(input_graph["nbrsStr"])
    graph_width = input_graph["props"].get("width", 0)
    if graph_width == 0:
        return ""
    for i in range(0, input_graph["size"], graph_width):
        print(neighbors_string[i:i+graph_width])
    jump_list = [f"{begin}{direction}{finish}" for begin, finish, direction in input_graph["jumps"]]
    for vx in range(len(input_graph["vProps"])):
        prop = input_graph["vProps"][vx]
        if prop and prop["rwd"] != 0 and prop["rwd"] != input_graph["props"]["rwd"]:
            print(str(vx) + ":", str(prop))

    for tuple_key, properties in input_graph["eProps"].items():
        if properties["rwd"] != 0 and properties["rwd"] != input_graph["props"]["rwd"]:
            print(f"{tuple_key}: {properties}")
    jumps_string = ";".join(jump_list)
    if jump_list:
        print(f"Jumps: {jumps_string}")
        return neighbors_string + "\n" + jumps_string
    return neighbors_string



def grfStrProps(graph):
    return ', '.join(f"'{k}': {v}" for k, v in graph["props"].items())



def getWidth(size):
    begin = math.ceil(math.sqrt(size))
    # Iterate from begin to size (inclusive)
    for x in range(begin, size + 1):
        # If size is divisible by i, return i
        if size % x == 0:
            return x

def getChar(vertex, width, nbrs):
    directions = {
    "": ".", 
    "E": "E", 
    "ES": "r", 
    "ESW": "v", 
    "EW": "-", 
    "N": "N", 
    "NE": "L", 
    "NES": ">", 
    "NESW": "+", 
    "NEW": "^", 
    "NS": "|", 
    "NSW": "<", 
    "NW": "J", 
    "S": "S", 
    "SW": "7", 
    "W": "W"
}

    dirs = ""
    

    if vertex - width in nbrs:
        dirs += "N"
    if vertex + 1 in nbrs:
        dirs += "E"
    if vertex + width in nbrs:
        dirs += "S"
    if vertex - 1 in nbrs:
        dirs += "W"
    # print(dirs)
    return directions[dirs]

#COPY FROM G DIRECTIVE 
# RETURNS NEIGHBORS FROM VDIR AND EDIR
def findNeighbors(vertex, width, size):
    toRet = set()
    if vertex % width > 0:
        toRet.add(vertex - 1)
    if vertex % width < width - 1:
        toRet.add(vertex + 1)
    if vertex // width > 0:
        toRet.add(vertex - width)
    if vertex // width < size // width - 1:
        toRet.add(vertex + width)
    return toRet



#FOR EDGE DIRECTIVE
def edgeAdj(node, gridWidth, gridHeight, direction):
    if direction == "W" and node % gridWidth > 0:
        return node - 1
    elif direction == "E" and node % gridWidth < gridWidth - 1:
        return node + 1
    elif direction == "S" and node // gridWidth < gridHeight - 1:
        return node + gridWidth
    elif direction == "N" and node // gridWidth > 0:
        return node - gridWidth
    else:
        return "."



#KEEP EXTERNAL SLICING

def sliceParse(framework, size):
    wSet = []
    for slce in framework.split(","):
        sliceFramework = re.findall("(-?\d*)(:?)(-?\d*)(:?)(-?\d*)", slce)[0]
        begin, finish, increase, front, back = sliceFramework[0], sliceFramework[2], sliceFramework[4], sliceFramework[1], sliceFramework[3]
        if begin:
            begin = int(begin) + size*(int(begin)<0)
        if not front and not back:
            wSet.append(begin)
        else:
            increase = int(increase) if increase else 1
            finish = int( finish) + size*(int( finish)<0) if finish else -1 if increase < 0 else size
            begin = int(begin) + size*(int(begin)<0) if begin else size - 1 if increase < 0 else 0
            wSet.extend(range(begin, finish, increase))

    return wSet

def solveGW2():
    return

def showPolicy():
    return

def calculate_policy(graph):
    # Initialize the policy with '.' for all squares
    policy = ['.' for _ in range(graph["size"])]
    jumps = []

    # Function to determine the direction character based on available moves
    def get_direction_char(moves):
        direction_map = {
            frozenset(['N']): 'N', frozenset(['S']): 'S',
            frozenset(['E']): 'E', frozenset(['W']): 'W',
            frozenset(['N', 'S']): '|', frozenset(['E', 'W']): '-',
            frozenset(['N', 'E']): 'L', frozenset(['S', 'E']): 'r',
            frozenset(['S', 'W']): '7', frozenset(['N', 'W']): 'J',
            frozenset(['N', 'S', 'E', 'W']): '+',
            frozenset(['W', 'N', 'E']): '^', frozenset(['W', 'S', 'E']): 'v',
            frozenset(['N', 'E', 'S']): '>', frozenset(['N', 'W', 'S']): '<',
        }
        return direction_map.get(frozenset(moves), '.')

    # Function to perform BFS and update the policy
    def bfs_update_policy(start):
        queue = [(start, 0, [])]
        visited = set()
        while queue:
            current, distance, path = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Check if current square has a reward
            if "rwd" in grfVProps(graph, current):
                policy[start] = '.'  # Mark as terminal square with reward
                return

            # Get possible moves from the current square
            moves = []
            for neighbor in grfNbrs(graph, current):
                if neighbor not in visited:
                    direction = None
                    if neighbor == current - 1: direction = 'W'
                    if neighbor == current + 1: direction = 'E'
                    if neighbor == current - graph["size"]: direction = 'N'
                    if neighbor == current + graph["size"]: direction = 'S'
                    if direction:
                        moves.append(direction)
                        queue.append((neighbor, distance + 1, path + [direction]))

            # Update the policy for the start square
            if moves:
                policy[start] = get_direction_char(moves)

    # Run BFS for each square in the graph
    for square in range(graph["size"]):
        bfs_update_policy(square)

    # Convert policy list to a string representation
    policy_str = [policy[i:i+graph["size"]] for i in range(0, graph["size"], graph["size"])]

    # Include jumps in the policy string if any
    if jumps:
        jumps_str = ';'.join(f"{j[0]}~{j[1]}" for j in jumps)
        policy_str.append(jumps_str)

    return policy_str

# Example usage:
# Assuming 'graph' is an instance of Graph with size, edges, and rewards properly defined.
# policy_str = calculate_policy(graph)
# print("Policy:")
# print(policy_str)
def processBFS(graph):
    # Extracting properties from the graph
    reward_vs = graph["props"]["rwd"]
    width = graph["props"]["width"]
    neighbors = graph["nbrs"]
    size = graph["size"]
    vertex_properties = graph["vProps"]
    edge_properties = graph["eProps"]
    jumps = graph["jumps"]

    # Initialize directions
    directions = {v: set() for v in range(size)}

    # Process jumps to create edges
    edges = dict()
    for jump in jumps:
        if '~' in jump:
            value = jump.split('~')
        else:
            value = jump.split('=')
        
        # Add edges to the dictionary
        v1, v2 = int(value[0]), int(value[1])
        if v1 in edges:
            edges[v1].append(v2)
        else:
            edges[v1] = [v2]
        
        # For '=' jumps, add the reverse direction as well
        if '=' in jump:
            if v2 in edges:
                edges[v2].append(v1)
            else:
                edges[v2] = [v1]

    # BFS algorithm to find shortest paths
    def BFS(neighbors, start, end):
        queue = [(start, [start])]
        visited = set()
        while queue:
            vertex, path = queue.pop(0)
            if vertex == end:
                return path
            elif vertex not in visited:
                visited.add(vertex)
                for neighbor in neighbors.get(vertex, []):
                    queue.append((neighbor, path + [neighbor]))
        return []

    # Process each vertex
    for vert in range(size):
        if vertex_properties[vert]["rwd"]:
            directions[vert] = '*'
        else:
            shortest_paths = []
            for goal in reward_vs:
                path = BFS(neighbors, vert, goal)
                if path:
                    steps = len(path)
                    shortest_paths.append((steps, path))
            shortest_paths.sort(key=lambda x: x[0])  # Sort by number of steps

            # Update directions based on shortest paths
            if shortest_paths:
                min_steps = shortest_paths[0][0]
                for steps, path in shortest_paths:
                    if steps == min_steps:
                        for p in path[1:]:  # Exclude the starting vertex
                            directions[vert].add(p)
    edge_rew_verts = []
    for edge in edge_properties:
        if edge_properties[edge]["active"]:
            v1, v2 = edge
            edge_rew_verts.extend([v1, v2])
            if directions[v1] != '*':
                directions[v1].add(v2)
            if directions[v2] != '*':
                directions[v2].add(v1)
    # ... continue adapting the rest of the original function

    return directions, reward_vs, edge_rew_verts



def main():
    # graph = grfParse(['GG44R18', 'V4R', 'V1BR7'])
    graph = grfParse(['G17', 'V2R16'])
    # graph = grfParse(['GG5W5 E+0S~R12 V1R])
    # graph = grfParse(['GG48W6 V23,28,29,22B E~22=21'])
    # graph = grfParse([GG14W7 V0B E~-13~0])

    graph = grfParse(args)
    # policy_str = calculate_policy(graph)
    print("Policy:")
    # print(". . . .")
    # print(policy_str)
    # for item in policy_str[0]:
    #     print(item)
    directions, reward_vs, edge_rew_verts = processBFS(graph)
    print(directions)
    # showPolicy(policy, graph)
    # size = grfSize(graph)
    # print(grfNbrs(graph, 0))
    # print(grfVProps(graph, 0))
    # edgesStr = grfStrEdges(graph) 
    # propsStr = grfStrProps(graph)
    # # output edgesStr
    # # print(grfNbrs(graph, 7))
    # # print(graph["nbrs"])
    # # print(edgesStr)
    # print(propsStr)
if __name__ == '__main__':main()

#Om Gole, 6, 2024