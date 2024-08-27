import sys; args = sys.argv[1:]
import re, math

#pythonu rlsetup.py 16 <- SIZE u? <- Width: you are either solving G0 or G1.

#G0

#G0 is the same thing with gridworld, but trying to find the maximum reward

#Default to G0 or G1

#3 reward args: R6 (default RWD) R 6:u (vertex and RWD) R:5 (sets default reward), if none, default is 12
# B# is a blocking vertex || B#SE (Vertex number and direction) -> toggle edge.

def parseDirectives(args):
    nbrs, nbrsStr, vprops, eprops, width, dRwd, size, type, gametype = set(), [], None, {}, 0, 12, 0, "N", "G1"
    size = int(args[0])
    width = getWidth(size)  #or int(args[1]) if len(args) > 1 and args[1].isdigit() else getWidth(graph["size"])
    # width = int(args[1]) if len(args) > 1 and args[1].isdigit() else getWidth(graph["size"])
    vprops = [-float("inf")] * size
    graph = {"dRwd":dRwd, "width": width, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vprops":vprops, "eProps": eprops, "game": gametype}

    # graph["vprops"] = [-float("inf")] * graph["size"]

    for arg in args[1:]:
        gdir = None
        rdir = None
        bdir = None
        if "G" in arg:
            gdir = arg
        if "R" in arg:
            rdir = arg
        if "r" in arg:
            rdir = arg
        if "B" in arg:
            bdir = arg
        if gdir != None: #CHECK G DIRECTIVE
            if arg == "G1":
                graph["game"] = "G1"
            graph["game"] = "G0"


        if rdir != None: #CHECK R DIRECTIVE

            rwdcase, strval, rwdval = distinguish_string(rdir)
            if rwdcase == "R:#": #Sets the implied reward to the number indicated (default is 12)
                graph["dRwd"] = int(strval)
            
            if rwdcase == "R#": #Sets the reward at the vertex indicated by number equal to the implied reward
                vprops[int(strval)] = graph["dRwd"]
            if rwdcase == "R#:#": #Sets the reward for the vertex at the first # to be equal to the 2nd #
                vprops[int(strval)] = int(rwdval)

        elif bdir != None: #CHECK B DIRECTIVE 
            bcase, vertex, dirs = distinguish_bstring(bdir)
            vertex = int(vertex)
            width = graph["width"]
            size = graph["size"]
            
            if len(dirs):
                for d in dirs:
                    if d == "N" and vertex // width: #NORTH
                        update_neighbors(graph, vertex, -width)
                    elif d == "S" and vertex // width < (size // width - 1): #SOUTH
                        update_neighbors(graph, vertex, width)
                    elif d == "E" and vertex % width < (width - 1): #EAST
                        update_neighbors(graph, vertex, 1)
                    elif d == "W" and vertex % width: #WEST
                        update_neighbors(graph, vertex, -1)
            else:
                if vertex // width:
                    update_neighbors(graph, vertex, -width)
                if vertex // width < (size // width - 1):
                    update_neighbors(graph, vertex, width)
                if vertex % width < (width - 1):
                    update_neighbors(graph, vertex, 1)
                if vertex % width:
                    update_neighbors(graph, vertex, -1)
    return graph
#OLD
def update_neighbors(graph, vertex, offset):
    neighbor_pair = (vertex, vertex + offset)
    if neighbor_pair in graph["nbrs"]:
        graph["nbrs"].remove(neighbor_pair)
        graph["nbrs"].remove((neighbor_pair[1], neighbor_pair[0]))
    else:
        graph["nbrs"].add(neighbor_pair)
        graph["nbrs"].add((neighbor_pair[1], neighbor_pair[0]))

def getWidth(size):
    begin = math.ceil(math.sqrt(size))
    # Iterate from begin to size (inclusive)
    for x in range(begin, size + 1):
        # If size is divisible by i, return i
        if size % x == 0:
            return x

#OLD
def distinguish_bstring(s):
    # Regular expression pattern for strings of the form B# and B#[NSEW]+
    pattern = r"^B(\d+)([NSEW]*)$"
    
    # Search for the pattern in the input string
    match = re.search(pattern, s)
    
    if match:
        # If the string matches the pattern, extract the number and the letters
        number = int(match.group(1))
        letters = match.group(2)
        
        # Determine the type of the string
        if letters:
            string_type = "B#[NSEW]+"
        else:
            string_type = "B#"
        
        return string_type, number, letters
    
    else:
        # If the string does not match the pattern, return None
        return None

#OLD
def distinguish_string(s):
    # Define the regex patterns
    pattern1 = r"^[Rr]:(\d+)$"
    pattern2 = r"^[Rr](\d+)$"
    pattern3 = r"^[Rr](\d+):(\d+)$"
    # Check if the string matches the first pattern
    match = re.match(pattern1, s)
    if match:
        return ("R:#", int(match.group(1)), 0)

    # Check if the string matches the second pattern
    match = re.match(pattern2, s)
    if match:
        return ("R#", int(match.group(1)), 0)

    # Check if the string matches the third pattern
    match = re.match(pattern3, s)
    if match:
        return ("R#:#", int(match.group(1)), int(match.group(2)))

    # If the string does not match any pattern, return None
    return None

def process_queue_G0(vertexqueue, maxrwds, iters, done, rwds, width, vpropers): #BFS
    while vertexqueue:
        starting = vertexqueue.pop(0)
        nbors = getNeighbors(starting[0], graph)
        for nbor in nbors:
            x, y = nbor // width, nbor % width
            if done.get((nbor, starting[2]), False) or nbor in rwds: continue
            done[(nbor, starting[2])] = True
            if starting[2] > maxrwds[x][y]:
                maxrwds[x][y] = starting[2]
                iters[x][y] = starting[1] + 1
            elif starting[2] == maxrwds[x][y]:
                iters[x][y] = min(iters[x][y], starting[1] + 1)
            vertexqueue.append((nbor, starting[1] + 1, starting[2]))

def process_queue_G1(vertexqueue, proportion, jbrunson, iters, done, rwds, width, vpropers): #NEW TYPE
    while vertexqueue:
        starting = vertexqueue.pop(0)
        nbors = getNeighbors(starting[0], graph)
        for nbor in nbors:
            x, y = nbor // width, nbor % width
            ratio = starting[2] / (starting[1] + 1)
            if ratio > proportion[x][y]:
                proportion[x][y] = ratio
                jbrunson[nbor] = [starting[0]]
                iters[x][y] = starting[1] + 1
            elif ratio == proportion[x][y]:
                jbrunson[nbor].append(starting[0])
            if done.get((nbor, starting[2]), False) or nbor in rwds: continue
            done[(nbor, starting[2])] = True
            vertexqueue.append((nbor, starting[1] + 1, starting[2]))

def getNeighbors(vertex, graph):
    size = graph["size"]
    width = graph["width"]
    tempneighbors = [
        (vertex - width, vertex >= width),
        (vertex + 1, vertex % width < width - 1),
        (vertex + width, vertex < size - width),
        (vertex - 1, vertex % width > 0)
    ]
    
    nbors = [v for v, condition in tempneighbors if condition and (vertex, v) not in graph["nbrs"]]
    return nbors

def findDir(src, dest, width):
    direction_map = { -width: "U", 1: "R", width: "D", -1: "L"}
    return direction_map.get(dest - src, "")

def initialize(graph):
    size = graph['size']
    width = graph['width']
    vpropers = graph['vprops']
    game = graph['game']
    inf = float('inf')
    dirst = [[[]] * width for _ in range(size // width)]
    vertexqueue = []
    rwds = []
    for i in range(size):
        if vpropers[i] != -inf: 
            vertexqueue.append((i, 0, vpropers[i]))
            rwds.append(i)
    return size, width, vpropers, game, inf, dirst, vertexqueue, rwds

def findPolicy(graph):
    size, width, vpropers, game, inf, dirst, vertexqueue, rwds = initialize(graph)
    if game == 'G0':
        # maxrwds = [[-inf] * width for _ in range(size // width)]
        # iters = [[inf] * width for _ in range(size // width)]
        # done = {}
        maxrwds, iters, done = [[-inf] * width for _ in range(size // width)], [[inf] * width for _ in range(size // width)], {}

        for x in rwds: 
            x, y = x // width, x % width
            maxrwds[x][y] = vpropers[x] 
            iters[x][y] = 0
        process_queue_G0(vertexqueue, maxrwds, iters, done, rwds, width, vpropers)
        # ... rest of the code ...
        for i in range(size):
            temp1, temp2 = i // width, i % width
            nbors = getNeighbors(i, graph)
            dirs = ''
            for nbor in nbors:
                x2, y2 = nbor // width, nbor % width
                if (maxrwds[temp1][temp2] == maxrwds[x2][y2]) and (iters[x2][y2] + 1 == iters[temp1][temp2]) and maxrwds[temp1][temp2] != -inf:
                    dirs += findDir(i, nbor, width)
            dirst[temp1][temp2] = dirs
    else:
        # proportion = [[-inf] * width for _ in range(size // width)]
        # jbrunson = [[] for _ in range(size)]
        # iters = [[inf] * width for _ in range(size // width)]
        # done = {}
        proportion, jbrunson, iters, done = [[-inf] * width for _ in range(size // width)], [[] for _ in range(size)], [[inf] * width for _ in range(size // width)], {}
        for x in rwds: 
            iters[x // width][x % width] = 0
        process_queue_G1(vertexqueue, proportion, jbrunson, iters, done, rwds, width, vpropers)
        # ... rest of the code ...
        for i in range(size):
            temp1, temp2 = i // width, i % width
            dirs = ''.join(findDir(i, nbor, width) for nbor in getNeighbors(i, graph) if proportion[temp1][temp2] != -inf and nbor in jbrunson[i])
            dirst[temp1][temp2] = dirs

    return dirst

def findPolicy2(graph):
    size, width, vpropers, game, inf, dirst, vertexqueue, rwds = initialize(graph)
    for i in range(size):
        if vpropers[i] != -inf: 
            vertexqueue.append((i, 0, vpropers[i]))
            rwds.append(i)
    if game == 'G0':
        mRwds, iters, done = [[-inf] * width for _ in range(size // width)], [[inf] * width for _ in range(size // width)], {}
        for x in rwds: 
            mRwds[x // width][x % width] = vpropers[x] 
            iters[x // width][x % width] = 0
        while vertexqueue:
            strats = vertexqueue.pop(0)
            nbors = getNeighbors(strats[0], graph)
            # Iterate over each neighbor
            for nbor in nbors:
                # Calculate the x and y coordinates
                x_coordinate = nbor // width
                y_coordinate = nbor % width

                # Create a tuple for the neighbor and strategy
                neighbor_strategy_tuple = (nbor, strats[2])

                # Check if the neighbor is already done or is a reward
                if neighbor_strategy_tuple in done and done[neighbor_strategy_tuple]:
                    continue
                if nbor in rwds:
                    continue

                # Mark the neighbor as done
                done[neighbor_strategy_tuple] = True

                # Update the rewards and iterations based on the strategy
                strategy_reward = strats[2]
                if strategy_reward > mRwds[x_coordinate][y_coordinate]:
                    mRwds[x_coordinate][y_coordinate] = max(mRwds[x_coordinate][y_coordinate], strategy_reward)
                    iters[x_coordinate][y_coordinate] = strats[1] + 1
                elif strategy_reward == mRwds[x_coordinate][y_coordinate]:
                    iters[x_coordinate][y_coordinate] = min(iters[x_coordinate][y_coordinate], strats[1] + 1)

                # Add the neighbor to the queue
                vertexqueue.append((nbor, strats[1] + 1, strategy_reward))

        for i in range(size):
            temp1, temp2 = i // width, i % width
            dirs = ''.join(findDir(i, nbor, width) for nbor in getNeighbors(i, graph) if (mRwds[temp1][temp2] == mRwds[nbor // width][nbor % width]) and (iters[nbor // width][nbor % width] + 1 == iters[temp1][temp2]) and mRwds[temp1][temp2] != -inf)
            dirst[temp1][temp2] = dirs

    else:
        mPropor, jbrunson, iters, done = [[-inf] * width for _ in range(size // width)], [[] for _ in range(size)], [[inf] * width for _ in range(size // width)], {}
        for x in rwds: 
            iters[x // width][x % width] = 0
        while vertexqueue:
            strats = vertexqueue.pop(0)
            nbors = getNeighbors(strats[0], graph)
            # Iterate over each neighbor
            for nbor in nbors:
                # Calculate the x and y coordinates
                x_coordinate = nbor // width
                y_coordinate = nbor % width

                # Calculate the ratio
                reward = strats[2]
                iteration = strats[1] + 1
                ratio = reward / iteration

                # Update the proportions, iterations, and jbrunson if the ratio is greater than the current maximum
                if ratio > mPropor[x_coordinate][y_coordinate]:
                    mPropor[x_coordinate][y_coordinate] = ratio
                    jbrunson[nbor].clear()
                    jbrunson[nbor].append(strats[0])
                    iters[x_coordinate][y_coordinate] = iteration
                # Append to jbrunson if the ratio is equal to the current maximum
                elif ratio == mPropor[x_coordinate][y_coordinate]:
                    jbrunson[nbor].append(strats[0])

                # Create a tuple for the neighbor and strategy
                neighbor_strategy_tuple = (nbor, reward)

                # Check if the neighbor is already done or is a reward
                if neighbor_strategy_tuple in done and done[neighbor_strategy_tuple]:
                    continue
                if nbor in rwds:
                    continue

                # Mark the neighbor as done
                done[neighbor_strategy_tuple] = True

                # Add the neighbor to the queue
                vertexqueue.append((nbor, iteration, reward))


        for i in range(size):
            temp1, temp2 = i // width, i % width
            dirs = ''.join(findDir(i, nbor, width) for nbor in getNeighbors(i, graph) if mPropor[temp1][temp2] != -inf and nbor in jbrunson[i])
            dirst[temp1][temp2] = dirs

    return dirst



# graph = parseDirectives(['4', 'G0', 'r0', 'B1'])
# graph = parseDirectives(['4', 'G0', 'R1'])
graph = parseDirectives(args)
# print(graph)
# policy = calculateOptimalDirections(graph)
policy = findPolicy2(graph)

#DISPLAY

final = ""
for i in range(graph['size']):
    if graph['vprops'][i] > 0: final = final + '*'
    else: final += {'': '.', 'U': 'U', 'R': 'R', 'D': 'D', 'L': 'L','RU': 'V', 'DRU': 'W', 'DR': 'S', 'DLR': 'T', 'DL': 'E','DLU': 'F', 'LU': 'M', 'LRU': 'N', 'DU': '|', 'LR': '-','DLRU': '+'}[''.join(sorted(policy[i // graph['width']][i % graph['width']]))]
for i in range(0, graph['size'], graph['width']):
    print(final[i:graph['width']+i])

# Om Gole, 6, 2024