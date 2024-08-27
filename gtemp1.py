import sys; args = sys.argv[1:]
import re


def grfParse(lstArgs): 
    nbrs = []
    nbrsStr = []
    vertex_properties = []
    edge_properties = {}
    jumps = []
    width, defaultReward, size, graphType = 0, 12, 0, "N"
    for arg in lstArgs:
        if "G" in arg:
            size = int(re.findall("(\d+)", arg)[0])
            graphType = re.findall("(G(?:G|N)?)", arg)[0]
            width = re.findall("W(\d+)", arg)
            defaultReward = re.findall("R(-?\d+)", arg)
            if width:
                width = int(width[0])
            else:
                width = getDefaultWidth(size)
            if defaultReward:
                defaultReward = int(defaultReward[0])
            else:
                defaultReward = 12
            if "N" in graphType:
                return {"properties":{"rwd":defaultReward}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vertex_properties}
            if width == 0:
                return {"properties":{"rwd":defaultReward, "width":width}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vertex_properties}
            else:
                for v in range(size):
                    this_vertex_nbrs = getNbrs(vertex=v, size=size, width=width)
                    nbrs.append(this_vertex_nbrs)
                    for nbr in this_vertex_nbrs:
                        edge_properties[(v, nbr)] = {"rwd":0}
                    nbrsStr.append(getSymbol(v, width=width, nbrs=this_vertex_nbrs))
                    vertex_properties.append({})
        elif "V" in arg:
            parsed_slice_instructions = re.findall("((-?\d*:*-?\d*,?)+)(R?\d*)(B?)", arg[1:])[0][0]
            block_directive = re.findall("B", arg[1:])
            new_reward = re.findall(("R(\d*)"), arg)
            if block_directive:
                block_directive = block_directive[0]
            if len(new_reward) >= 1:
                new_reward = new_reward[0]
                if not new_reward: 
                    new_reward = defaultReward
            wSet = set(parseSlices(parsed_slice_instructions, size))
            if block_directive: 
                for v in wSet:
                    curr_nbrs = nbrs[v]
                    default_nbrs = getNbrs(v, size, width)
                    allNbrs = curr_nbrs.union(default_nbrs)
                    removeSet = allNbrs.intersection(curr_nbrs)
                    addSet = allNbrs - curr_nbrs
                    curr_nbrs = curr_nbrs - removeSet
                    for nbr in addSet:
                        curr_nbrs.add(nbr)
                    for n in allNbrs:
                        curr_nbrs_n = nbrs[n]
                        if v in curr_nbrs_n:
                            curr_nbrs_n = curr_nbrs_n - {v}
                        else:
                            curr_nbrs_n.add(v)
                        nbrs[n] = curr_nbrs_n
                        nbrsStr[n] = getSymbol(n, width=width, nbrs=curr_nbrs_n)
                    nbrs[v] = curr_nbrs
                    nbrsStr[v] = getSymbol(v, width=width, nbrs=curr_nbrs)
            if new_reward:
                for v in wSet:
                    vertex_properties[v] = {"rwd":new_reward}
        elif "E" in arg:
            mngmnt = re.findall("(!|\+|\*|~|@)", arg)
            if not mngmnt:
                mngmnt = "~"
            else:
                mngmnt = mngmnt[0]
            vslices1 = re.findall("([-:\,\d]+)(?=[NSEW=~])", arg[1:])[0]
            vslices2 = re.findall("([-:\,\d]*)(?!.*[~=])", arg[1:])[0]
            direction_type = re.findall("[=~]", arg[1:])[-1]
            new_edge_reward = re.findall(("R(\d*)"), arg)
            if len(new_edge_reward) >= 1:
                new_edge_reward = new_edge_reward[0]
                if not new_edge_reward:
                    new_edge_reward = defaultReward
            vLst1 = parseSlices(vslices1, size)
            if not vslices2:
                new_edges = []
                cardinal_directions = re.findall("(N|S|E|W)", arg[1:])
                for direction in cardinal_directions:
                    new_edges += [(v, getAdjacent(v, width, size//width, direction)) for v in vLst1 if getAdjacent(v, width, size//width, direction) != "."]
            else:
                new_edges = list(zip(vLst1, parseSlices(vslices2, size)))
            if direction_type == "=":
                new_edges += [(tup[1], tup[0]) for tup in new_edges]
            for edge in new_edges:
                    start, end = edge[0], edge[1]
                    curr_nbrs = nbrs[start]
                    if mngmnt == "!":
                        if end in curr_nbrs:
                            curr_nbrs -= {end}
                        if (start,end) in edge_properties:
                            del edge_properties[(start, end)]
                    elif mngmnt == "+":
                        if not end in curr_nbrs:
                            if not end in getNbrs(start, size, width):
                                jumps.append((start, end, direction_type))
                            curr_nbrs.add(end)
                        if not (start, end) in edge_properties:
                            if new_edge_reward:
                                edge_properties[(start, end)] = {"rwd":new_edge_reward}
                    elif mngmnt == "~":
                        if end in curr_nbrs:
                            curr_nbrs -= {end}
                        else:
                            if not end in getNbrs(start, size, width):
                                jumps.append((start, end, direction_type))
                            curr_nbrs.add(end)
                        if (start,end) in edge_properties:
                            del edge_properties[(start, end)]
                        elif new_edge_reward:
                            edge_properties[(start, end)] = {"rwd":new_edge_reward}
                    elif mngmnt == "@":
                        if (start,end) in edge_properties and new_edge_reward:
                            edge_properties[(start, end)] = {"rwd":new_edge_reward}
                    elif mngmnt == "*":
                        if not end in curr_nbrs:
                            if not end in getNbrs(start, size, width):
                                jumps.append((start, end, direction_type))
                            curr_nbrs.add(end)
                        if new_edge_reward:
                            edge_properties[(start, end)] = {"rwd":new_edge_reward}
                    nbrs[start] = curr_nbrs
                    nbrsStr[start] = getSymbol(start, width, curr_nbrs)
    # print(nbrs)
    # print(vertex_properties)
    # print(edge_properties)
    # print(jumps)
    return {"properties":{"rwd":defaultReward, "width":width}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vertex_properties, "eProps":edge_properties, "jumps":jumps}

def getAdjacent(vertex, width, height, direction):
    if direction == "W":
        if vertex - 1 >= 0 and (vertex - 1)//width == vertex//width:
            return vertex - 1
        else:
            return "."
    if direction == "E":
        if vertex + 1 < width*height and (vertex + 1)//width == vertex//width:
            return vertex + 1
        else:
            return "."
    if direction == "S":
        if vertex + width < width*height and (vertex + width)%width == vertex%width:
            return vertex + width
        else:
            return "."
    if direction == "N":
        if vertex - width >= 0 and (vertex - width)%width == vertex%width:
            return vertex - width
        else:
            return "."
        
def parseSlices(instructions, size):
    wSet = []
    for slce in instructions.split(","):
        slice_instructions = re.findall("(-?\d*)(:?)(-?\d*)(:?)(-?\d*)", slce)[0]
        start, end, increment = slice_instructions[0], slice_instructions[2], slice_instructions[4]
        if start:
            start = int(slice_instructions[0]) + size*(int(slice_instructions[0])<0)
        if not slice_instructions[1] and not slice_instructions[3]:
            wSet.append(start)
        else:
            if increment:
                increment = int(increment)
            else:
                increment = 1
            if end:
                end = int(end) + size*(int(end)<0)
            else:
                if increment < 0:
                    end = -1
                else:
                    end = size
            if not start:
                if increment < 0:
                    start = size - 1
                else:
                    start = 0
            for v in range(start, end, increment):
                wSet.append(v)
    return wSet

def getDefaultWidth(sizeArg):
    sqrtSize = sizeArg**0.5
    start = int(sqrtSize)
    if start < sqrtSize:
        start += 1
    for i in range(start, sizeArg + 1):
        if sizeArg%i==0:
            return i

def getSymbol(vertex, width, nbrs):
    symbol = ""
    symbols_dct = {"N":"N", "S":"S", "E":"E", "W":"W", "NW":"J", "SW":"7", "ENW":"^", "ENSW":"+", "ESW":"v", "EN":"L", "ES":"r", "ENS":">", 
                "EW":"-", "NS":"|", "":".", "NSW":"<"}
    if vertex - 1 in nbrs:
        symbol += "W"
    if vertex + 1 in nbrs:
        symbol += "E"
    if vertex + width in nbrs:
        symbol += "S"
    if vertex - width in nbrs:
        symbol += "N"
    return symbols_dct[''.join(sorted(symbol))]

def getNbrs(vertex, size, width):
    nbrs = set()
    if vertex - 1 >= 0 and (vertex - 1)//width == vertex//width:
        nbrs.add(vertex - 1)
    if vertex + 1 < size and (vertex + 1)//width == vertex//width:
        nbrs.add(vertex + 1)
    if vertex + width < size and (vertex + width)%width == vertex%width:
        nbrs.add(vertex + width)
    if vertex - width >= 0 and (vertex - width)%width == vertex%width:
        nbrs.add(vertex - width)
    return nbrs

def grfSize(graph): 
    print(type(graph["size"]))
    return graph["size"]
def grfNbrs(graph, v): 
    if graph["nbrs"]:
        return graph["nbrs"][v]
    else:
        return {}
def grfGProps(graph): 
    return graph["properties"]

def grfVProps(graph, v): 
    if graph["vProps"] and graph["vProps"][v] != 0:
        return graph["vProps"][v]
    else:
        return {}
    
def grfEProps(graph, v1, v2):
    eProps = graph["eProps"]
    if (v1, v2) in eProps:
        if eProps[(v1, v2)]["rwd"] != 0:
            return eProps[(v1, v2)]
        else:
            return {}
    else:
        return {}
    
def grfStrEdges(graph): 
    nbrsToString = graph["nbrsStr"]
    toRet = "".join(nbrsToString)
    if not "width" in graph["properties"]:
        return ""
    width = graph["properties"]["width"]
    if width == 0:
        return ""
    for i in range(0, graph["size"], width):
        print(toRet[i:i+width])
    jump_str_lst = []
    for jump in graph["jumps"]:
        start, end, direc = jump[0], jump[1], jump[2]
        jump_str_lst.append(str(start) + str(direc) + str(end))
    for vx, prop in enumerate(graph["vProps"]):
        if prop and prop["rwd"] != 0 and prop["rwd"] != graph["properties"]["rwd"]:
            print(str(vx) + ":", str(prop))
    for tup in graph["eProps"]:
        if graph["eProps"][tup]["rwd"] != 0 and graph["eProps"][tup]["rwd"] != graph["properties"]["rwd"]:
            print(str(tup) + ":", str(graph["eProps"][tup]))            
    jumpRet = ";".join(jump_str_lst)
    if graph["jumps"]:
        print("Jumps:", jumpRet)
        return toRet + "\n" + jumpRet
    else:
        return toRet

def grfStrProps(graph): 
    return str(graph["properties"]) 

def main():
    # graph = grfParse(['GG36', 'V14,15,20,21B', 'E~21=27'])
    # graph = grfParse(['GN25W5R28', 'V3R'])
    graph = grfParse(['GN12W6R25', 'V0R'])
    temp = grfSize(graph)
    # print(grfVProps(temp, 0))
    # graph = grfParse(args)
    edgesStr = grfStrEdges(temp)
    propsStr = grfStrProps(temp)
    # print(grfVProps(graph, 3))
    # output edgesStr
    # print(edgesStr)
    # print(grfNbrs(graph, 7))
    print(propsStr)
if __name__ == '__main__':main()

#Om Gole, 6, 2024