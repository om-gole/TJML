import sys; args = sys.argv[1:]
import re
import math

def grfParse(args):
    #default props
    graph = {}
    nbrs, nbrsStr, vprops, eprops, jumps, width, dRwd, size, type = [], [], [], {}, [], 0, 12, 0, "N"
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

def main():
    # graph = grfParse(['GG44R18', 'V4R', 'V1BR7'])
    # graph = grfParse(['GG4R18'])
    # graph = grfParse(['GG5W5 E+0S~R12 V1R])
    # graph = grfParse(['GG48W6 V23,28,29,22B E~22=21'])
    # graph = grfParse([GG14W7 V0B E~-13~0])

    graph = grfParse(args)
    # print(graph)
    # size = grfSize(graph)
    # print(grfNbrs(graph, 0))
    # print(grfVProps(graph, 0))
    # graph = grfParse(['G7W0', 'V7R'])
    # print(grfVProps(graph, 0))
    edgesStr = grfStrEdges(graph) 
    propsStr = grfStrProps(graph)
    # output edgesStr
    # print(grfNbrs(graph, 7))
    # print(graph["nbrs"])
    # print(edgesStr)
    print(propsStr)





    # print("Policy:")
    
if __name__ == '__main__':main()