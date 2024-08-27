def main(args):
    process = os.environ
    lstArgs = list(args)
    graph = grfParse(lstArgs)
    edgesStr = grfStrEdges(graph)
    propsStr = grfStrProps(graph)
    print(propsStr)

def grfParse(lstArgs):
    nbrs = []
    nbrsStr = []
    vertex_properties = []
    edge_properties = {}
    jumps = []
    width = 0
    defaultReward = 12
    size = 0
    graphType = "N"
    for arg in lstArgs:
        if "G" in arg:
            size = int(re.sub("[^0-9]", "", arg))
            graphType = re.sub("[^G]", "", arg)
            width = int(re.sub("[^W0-9]", "", arg))
            defaultReward = int(re.sub("[^R0-9-]", "", arg))
            if width == 0:
                return {"properties": {"rwd": defaultReward, "width": width}, "nbrs": nbrs, "nbrsStr": nbrsStr, "size": size, "vProps": vertex_properties}
            else:
                for v in range(size):
                    this_vertex_nbrs = getNbrs(v, size, width)
                    nbrs.append(this_vertex_nbrs)
                    for nbr in this_vertex_nbrs:
                        edge_properties[(v, nbr)] = {"rwd": 0}
                    nbrsStr.append(getSymbol(v, width, this_vertex_nbrs))
                    vertex_properties.append({})
        elif "V" in arg:
            parsed_slice_instructions = re.sub("[^0-9:-]", "", arg[1:])
            block_directive = re.sub("[^B]", "", arg)
            new_reward = re.sub("[^R0-9]", "", arg)
            if len(block_directive) >= 1:
                block_directive = block_directive[:1]
            if len(new_reward) >= 1:
                if new_reward == "":
                    new_reward = str(defaultReward)
            wSet = set(parseSlices(parsed_slice_instructions, size))
            if len(block_directive) >= 1:
                for v in wSet:
                    curr_nbrs = nbrs[v]
                    default_nbrs = getNbrs(v, size, width)
                    allNbrs = curr_nbrs.union(default_nbrs)
                    removeSet = allNbrs.intersection(curr_nbrs)
                    addSet = allNbrs - curr_nbrs
                    curr_nbrs -= removeSet
                    curr_nbrs |= addSet
                    for nbr in allNbrs:
                        curr_nbrs_n = nbrs[nbr]
                        if v in curr_nbrs_n:
                            curr_nbrs_n.remove(v)
                        else:
                            curr_nbrs_n.add(v)
                        nbrs[nbr] = curr_nbrs_n
                        nbrsStr[nbr] = getSymbol(nbr, width, curr_nbrs_n)
                    nbrs[v] = curr_nbrs
                    nbrsStr[v] = getSymbol(v, width, curr_nbrs)
            if len(new_reward) >= 1:
                for v in wSet:
                    vertex_properties[v] = {"rwd": int(new_reward)}
        elif "E" in arg:
            mngmnt = re.sub("[^!+~@*]", "", arg)
            if len(mngmnt) == 0:
                mngmnt = "~"
            else:
                mngmnt = mngmnt[:1]
            vslices1 = re.sub("(?=[NSEW=~])", "", arg[1:])
            vslices2 = re.sub("(?!.*[~=])", "", arg[1:])
            direction_type = re.sub("[^=~]", "", arg[1:])[len(re.sub("[^=~]", "", arg[1:])) - 1]
            new_edge_reward = re.sub("[^R0-9]", "", arg)
            if len(new_edge_reward) >= 1:
                if new_edge_reward == "":
                    new_edge_reward = str(defaultReward)
            vLst1 = parseSlices(vslices1, size)
            new_edges = []
            if len(vslices2) == 0:
                cardinal_directions = re.sub("[^NSEW]", "", arg)
                for direction in cardinal_directions:
                    new_edges.extend([(v, getAdjacent(v, width, size // width, direction)) for v in vLst1 if getAdjacent(v, width, size // width, direction) != "."])
            else:
                new_edges = [(v, *parseSlices(vslices2, size)) for v in vLst1]
            if direction_type == "=":
                new_edges.extend([(edge[1], edge[0]) for edge in new_edges])
            for edge in new_edges:
                start, end = edge[0], edge[1]
                curr_nbrs = nbrs[start]
                if mngmnt == "!":
                    if end in curr_nbrs:
                        curr_nbrs.remove(end)
                    if (start, end) in edge_properties:
                        del edge_properties[(start, end)]
                elif mngmnt == "+":
                    if end not in curr_nbrs:
                        if getNbrs(start, size, width).isdisjoint({end}):
                            jumps.append((start, end, direction_type))
                        curr_nbrs.add(end)
                    if (start, end) not in edge_properties:
                        if len(new_edge_reward) >= 1:
                            edge_properties[(start, end)] = {"rwd": int(new_edge_reward)}
                elif mngmnt == "~":
                    if end in curr_nbrs:
                        curr_nbrs.remove(end)
                    else:
                        if getNbrs(start, size, width).isdisjoint({end}):
                            jumps.append((start, end, direction_type))
                        curr_nbrs.add(end)
                    if (start, end) in edge_properties:
                        del edge_properties[(start, end)]
                    elif len(new_edge_reward) >= 1:
                        edge_properties[(start, end)] = {"rwd": int(new_edge_reward)}
                elif mngmnt == "@":
                    if (start, end) in edge_properties and len(new_edge_reward) >= 1:
                        edge_properties[(start, end)] = {"rwd": int(new_edge_reward)}
                elif mngmnt == "*":
                    if end not in curr_nbrs:
                        if getNbrs(start, size, width).isdisjoint({end}):
                            jumps.append((start, end, direction_type))
                        curr_nbrs.add(end)
                    if len(new_edge_reward) >= 1:
                        edge_properties[(start, end)] = {"rwd": int(new_edge_reward)}
                nbrs[start] = curr_nbrs
                nbrsStr[start] = getSymbol(start, width, curr_nbrs)
    return {"properties": {"rwd": defaultReward, "width": width}, "nbrs": nbrs, "nbrsStr": nbrsStr, "size": size, "vProps": vertex_properties, "eProps": edge_properties, "jumps": jumps}

def getAdjacent(vertex, width, height, direction):
    if direction == "W":
        if vertex - 1 >= 0 and (vertex - 1) // width == vertex // width:
            return {vertex - 1}
        else:
            return set()
    if direction == "E":
        if vertex + 1 < width * height and (vertex + 1) // width == vertex // width:
            return {vertex + 1}
        else:
            return set()
    if direction == "S":
        if vertex + width < width * height and (vertex + width) % width == vertex % width:
            return {vertex + width}
        else:
            return set()
    if direction == "N":
        if vertex - width >= 0 and (vertex - width) % width == vertex % width:
            return {vertex - width}
        else:
            return set()
    return set()

def parseSlices(instructions, size):
    wSet = set()
    for slce in instructions.split(","):
        slice_instructions = slce.split(":")
        start = int(slice_instructions[0])
        end = int(slice_instructions[1])
        increment = int(slice_instructions[2])
        if start != 0:
            start = int(slice_instructions[0]) + size * (1 if int(slice_instructions[0]) < 0 else 0)
        if slice_instructions[1] == "" and slice_instructions[3] == "":
            wSet.add(start)
        else:
            if increment == 0:
                increment = 1
            if slice_instructions[2] != "":
                increment = int(slice_instructions[2])
            else:
                increment = 1
            if slice_instructions[1] != "":
                end = int(slice_instructions[1]) + size * (1 if int(slice_instructions[1]) < 0 else 0)
            else:
                if increment < 0:
                    end = -1
                else:
                    end = size
            if start == 0:
                if increment < 0:
                    start = size - 1
                else:
                    start = 0
            for v in range(start, end, increment):
                wSet.add(v)
    return list(wSet)

def getDefaultWidth(sizeArg):
    sqrtSize = int(math.sqrt(sizeArg))
    start = math.floor(sqrtSize)
    if start < sqrtSize:
        start += 1
    for i in range(start, sizeArg + 1):
        if sizeArg % i == 0:
            return i
    return 0

def getSymbol(vertex, width, nbrs):
    symbol = []
    symbols_dct = {"N": "N", "S": "S", "E": "E", "W": "W", "NW": "J", "SW": "7", "ENW": "^", "ENSW": "+", "ESW": "v", "EN": "L", "ES": "r", "ENS": ">", "EW": "-", "NS": "|", "": ".", "NSW": "<"}
    if vertex - 1 in nbrs:
        symbol.append("W")
    if vertex + 1 in nbrs:
        symbol.append("E")
    if vertex + width in nbrs:
        symbol.append("S")
    if vertex - width in nbrs:
        symbol.append("N")
    return symbols_dct["".join(sorted(symbol))]

def getNbrs(vertex, size, width):
    nbrs = set()
    if vertex - 1 >= 0 and (vertex - 1) // width == vertex // width:
        nbrs.add(vertex - 1)
    if vertex + 1 < size and (vertex + 1) // width == vertex // width:
        nbrs.add(vertex + 1)
    if vertex + width < size and (vertex + width) % width == vertex % width:
        nbrs.add(vertex + width)
    if vertex - width >= 0 and (vertex - width) % width == vertex % width:
        nbrs.add(vertex - width)
    return nbrs

def grfSize(graph):
    return graph["size"]

def grfNbrs(graph, v):
    if "nbrs" in graph:
        return graph["nbrs"][v]
    else:
        return set()

def grfGProps(graph):
    return graph["properties"]

def grfVProps(graph, v):
    if "vProps" in graph and v in graph["vProps"] and graph["vProps"][v] != 0:
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
    if "width" not in graph["properties"]:
        return ""
    width = graph["properties"]["width"]
    if width == 0:
        return ""
    for i in range(0, len(graph["nbrs"]), width):
        print(toRet[i:i + width])
    jump_str_lst = []
    for jump in graph["jumps"]:
        start, end, direc = jump
        jump_str_lst.append(str(start) + direc + str(end))
    for vx in range(len(graph["vProps"])):
        prop = graph["vProps"][vx]
        if "rwd" in prop and prop["rwd"] != 0 and prop["rwd"] != graph["properties"]["rwd"]:
            print(str(vx) + ":" + str(prop))
    for tup in graph["eProps"]:
        prop = graph["eProps"][tup]
        if prop["rwd"] != 0 and prop["rwd"] != graph["properties"]["rwd"]:
            print(str(tup) + ":" + str(prop))
    jumpRet = ";".join(jump_str_lst)
    if "jumps" in graph:
        print("Jumps:" + jumpRet)
        return toRet + "\n" + jumpRet
    else:
        return toRet

def grfStrProps(graph):
    return str(graph["properties"])

if __name__ == "__main__":
    main(sys.argv[1:])


