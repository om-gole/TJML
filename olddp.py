import sys; args = sys.argv[1:]
import re
import math

def grfParse(args):
    #default props
    graph = {}
    nbrs, nbrsStr, vprops, eprops, jumps, width, dRwd, size, type = [], [], [], {}, [], 0, 12, 0, "N"
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

            if width == 0:
                # width = size
                # type = "N"
                width = size
                type = "N"

            if type == "N":
                # width = size
                graph = {"props":{"rwd":dRwd}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vprops, "type":"N"}
                # return graph
            
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
                # print("vertex", vertex, width, temp_nbrs)
                if type == "N":     
                    nbrsStr.append(set())
                else:
                    nbrsStr.append(getChar(vertex, width, temp_nbrs))
                vprops.append({})
        #VDIRECTIVE 
        if vdir != None:
            # print(vdir)
            parsed_slice_instructions = re.findall("((-?\d*:*-?\d*,?)+)", vdir[1:])[0][0]
            block_directive = 'B' in vdir[1:]
            new_reward = re.findall("R(\d*)", vdir)[0] if 'R' in vdir else None #DEFAULT V SLC REWARD!
            if new_reward == "":
                new_reward = dRwd
            # print("new reward:", new_reward)
            wSet = set(sliceParse(parsed_slice_instructions, size))
            # print("wSet", wSet)
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
                    # if type != "N":
                    nbrsStr[v] = getChar(v, width, tempp_neighbors)
            # print("Wset", wSet)
            # print("RWDDD", new_reward)
            if new_reward:
                for v in wSet:
                    # print("v", v)
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
        if type == "N":
            nbrs = set()
        graph = {"props":{"rwd":dRwd, "width": width}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vprops, "eProps": eprops, "jumps": jumps, "type": type}
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
        return graph["vProps"][v]  #Should return a dict at the vertex
    else:
        return {}



def grfEProps(graph, v1, v2):
    return graph["eProps"].get((v1, v2), {}) if graph["eProps"].get((v1, v2), {}).get("rwd", 0) != 0 else {}


def grfStrEdges(input_graph):
    neighbors_string = "".join(str(input_graph["nbrsStr"]))
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

def get_dirs(neighbors, width):
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
    dir_list = []
    for i, nbrs in enumerate(neighbors):
        dir_str = ''
        if i - width in nbrs:  # North
            dir_str += 'N'
        if i + 1 in nbrs and (i + 1) % width != 0:  # East
            dir_str += 'E'
        if i + width in nbrs:  # South
            dir_str += 'S'
        if i - 1 in nbrs and i % width != 0:  # West
            dir_str += 'W'
        dir_list.append(directions[dir_str])
    return dir_list

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

def grfValuePolicy():
    return
def grfPolicyFromValuation():
    return
def grfFindOptimalPolicy():
    return


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

def grfValuePolicy(graph, policy, gamma):
    # Initialize the valuation with empty strings
    valuation = ['' for _ in range(grfSize(graph))]
    while True:
        new_valuation = valuation.copy()
        for state in range(grfSize(graph)):
            if 'rwd' in grfVProps(graph, state):
                new_valuation[state] = grfVProps(graph, state)['rwd']
            elif policy[state]:  # Check if the set of neighbors is not empty
                average_val = float(sum((float(valuation[nbr]) if valuation[nbr] != '' else 0) for nbr in policy[state]) / len(policy[state]))
                # Use multiplication if gamma > 0.5, subtraction otherwise
                new_valuation[state] = float(average_val * gamma if gamma > 0.5 else average_val - gamma)
        if all(abs((float(valuation[i]) if valuation[i] != '' else 0) - (float(new_valuation[i]) if new_valuation[i] != '' else 0)) < 1e-3 for i in range(grfSize(graph))):
            break
        valuation = new_valuation
    for i in range(len(valuation)):
        if valuation[i] != '':
            valuation[i] = float(valuation[i])
    return valuation




def grfPolicyFromValuation(graph, valuation):
    policy = [set() for _ in range(grfSize(graph))]
    for state in range(grfSize(graph)):
        if 'rwd' not in grfVProps(graph, state) and grfNbrs(graph, state):  # Check if the set of neighbors is not empty
            max_val = max(float(valuation[nbr]) if valuation[nbr] != '' else 0 for nbr in grfNbrs(graph, state))
            policy[state] = {nbr for nbr in grfNbrs(graph, state) if valuation[nbr] == max_val}
    return policy

def grfFindOptimalPolicy(graph):
    policy = [grfNbrs(graph, state) for state in range(grfSize(graph))]
    while True:
        old_policy = policy
        valuation = grfValuePolicy(graph, policy, 0.01)
        policy = grfPolicyFromValuation(graph, valuation)
        if all(old_policy[i] == policy[i] for i in range(grfSize(graph))):
            break
    return policy

def print_graph(directions, width):
    for i in range(0, len(directions), width):
        print(' '.join('00' if d == 0 or d == '' else str(d) for d in directions[i:i+width]))

def main():
    # graph = grfParse(['GG44R18', 'V4R', 'V1BR7'])
    # graph = grfParse(['GG13W0', 'V9R'])
    # graph = grfParse(['GG5W5 E+0S~R12 V1R])
    # graph = grfParse(['GG48W6 V23,28,29,22B E~22=21'])
    # graph = grfParse([GG14W7 V0B E~-13~0])
    # graph = grfParse(['G9', 'V5R'])
    # graph = grfParse(['GN4W23', 'V3R'])
    # graph = grfParse(['GG44', 'V4::11B', 'V20R34',  'V18R25'])
    # print(graph)
    graph = grfParse(args)
    # print("NEIGHBORS",    grfNbrs(graph, 0))

    # print(graph)
    # size = grfSize(graph)
    # print(grfNbrs(graph, 0))
    # print(grfVProps(graph, 0))
    print("Graph:")
    edgesStr = grfStrEdges(graph) 
    propsStr = grfStrProps(graph)
    # print(grfNbrs(graph, 3))
    # output edgesStr
    # print(grfNbrs(graph, 7))
    # print(graph["nbrs"])
    # print(edgesStr)
    # print(propsStr)
    optimal_policy = grfFindOptimalPolicy(graph)
    printpolicy = get_dirs(optimal_policy, graph["props"]["width"])
    print("Optimal Policy:")
    print_graph(printpolicy, graph["props"]["width"])  # Print the optimal policy

    # Calculate the value of the optimal policy
    value_policy = grfValuePolicy(graph, optimal_policy, 0.01)
    print(value_policy)
    print("Valuation:")
    print_graph(value_policy, graph["props"]["width"])  # Print the valuation

    # Generate a new policy from the valuation
    new_policy = grfPolicyFromValuation(graph, value_policy)
    printpolicy = get_dirs(new_policy, graph["props"]["width"])
    print("New Policy from Valuation:")
    print_graph(printpolicy, graph["props"]["width"])

#"C:\Users\westo\OneDrive\Desktop\TJHSST\TJHSST ML"


    # print("Policy:")
    
if __name__ == '__main__':main()


#Om Gole, 6, 2024