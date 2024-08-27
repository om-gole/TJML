import sys; args = sys.argv[1:]
import re

def grfParse(lstArgs):
    graph = {'type': 'G', 'size': 0, 'width': 0, 'reward': 12, 'edges': set(), 'vertices': {}, 'edgeRwds': {}, 'gdir': lstArgs[0]}
    dims = parseArgs(lstArgs[0])
    graph['size'] = dims[0]
    graph['width'] = dims[1]
    graph['reward'] = dims[2]
    graph['vertices'] = {i:'none' for i in range(graph['size'])}
    if('N' in lstArgs[0]):
        graph['width'] = 0
    if graph['width'] > 0:
        defaults = initEdge(graph)
        graph['edges'] = defaults
        graph['edgeRwds']={tup:-1 for tup in defaults}

    if len(lstArgs)>1:
        for arg in lstArgs[1:]:
            if arg[0]=='V':
                graph = vDirective(graph, arg[1:])
            if arg[0]=='E':
                graph = edgeDirective(graph, arg)
    return graph

def parseArgs(gDir):
    varlist = list(map(int, re.findall(r'\d+', gDir)))
    ln = varlist[0]
    data = [ln, 0, 12]

    if len(varlist) == 3:
        return varlist

    factors = [x for x in range(1, ln + 1) if ln % x == 0]
    mid_factor = factors[len(factors) // 2]

    if len(varlist) == 1 or 'R' in gDir:
        data[1] = mid_factor
        if 'R' in gDir:
            data[2] = varlist[1]
    elif 'W' in gDir:
        data[1] = varlist[1]

    return data

def grfNbrs(graph, vert):
    # connected_vProps = {edge[1] for edge in graph['edges'] if edge[0] == vert}
    connected_vProps = set()  # Initialize an empty set

    for edge in graph['edges']:  # Iterate over each edge in the graph
        if edge[0] == vert:  # If the first vertex of the edge is the vertex we're interested in
            connected_vProps.add(edge[1])  # Add the second vertex of the edge to the set

    return connected_vProps

def grfSize(graph):
    return graph['size']

def grfNbrs(graph, vert):
    # Return the set of vProps connected to vertex vert
    return {edge[1] for edge in graph['edges'] if edge[0] == vert}

def grfGProps(graph):
    # Return the rwd and width props of the graph
    props = {'rwd': graph['rwd'], 'width': graph['width']}
    return props if 'N' not in graph['gdir'] else {'rwd': graph['rwd']}

def grfVProps(graph, vert):
    # Return the rwd property of vertex vert if it exists
    return {'rwd': graph['vProps'][vert]} if graph['vProps'][vert] != 'none' else {}

def grfEProps(graph, vertex1, vertex2):
    # Return the rwd property of edge (vertex1, vertex2) if it exists
    if (vertex1, vertex2) in graph['edges'] and graph['eProps'].get((vertex1, vertex2), -1) != -1:
        return {'rwd': graph['eProps'][(vertex1, vertex2)]}
    return {}

def initEdge(graph):
    wdth = graph['width']
    adjacent = set()
    for vert in range(graph['size']):
        adjacent.update([(vert, vert - wdth), (vert, vert + wdth)])
        if vert % wdth == 0:
            adjacent.add((vert, vert + 1))
        elif vert % wdth == wdth - 1:
            adjacent.add((vert, vert - 1))
        else:
            adjacent.update([(vert, vert - 1), (vert, vert + 1)])
    result = set()
    for y in adjacent:
        if 0 <= y[1] < graph['size']:
            result.add(y)
    return result

def get_indices(vector_group):
    return [int(num) for num in re.findall(r'-?\d+', vector_group)]

def handle_single_index(indices, index_list):
    if 1 == len(indices):
        return [index_list[indices[0]]]
    return []

def handle_range_with_step(indices, index_list):
    if 3 == len(indices):
        start, end, step = indices
        return index_list[start:end:step]
    return []

def handle_range_without_step(indices, index_list):
    if 2 == len(indices):
        start, end = indices
        return index_list[start:end]
    return []

def handle_range_with_open_end_and_step(vector_group, indices, index_list):
    if '::' in vector_group:
        start = indices[0] if indices else None
        step = indices[1] if len(indices) > 1 else None
        return index_list[start::step]
    return []

# def slice_parse(size, vector_string):
#     index_list = list(range(size))
#     vertex_indices = []
#     vector_groups = vector_string.split(',')

#     for vector_group in vector_groups:
#         indices = get_indices(vector_group)
#         if not indices:
#             continue

#         vertex_indices.extend(handle_single_index(indices, index_list))
#         vertex_indices.extend(handle_range_with_step(indices, index_list))
#         vertex_indices.extend(handle_range_without_step(indices, index_list))
#         vertex_indices.extend(handle_range_with_open_end_and_step(vector_group, indices, index_list))

#     return list(set(vertex_indices))

def slice_parse(size, vPrs):
    allIndices = [i for i in range(size)]
    vIndices = []
    index_r, index_b = vPrs.find('R'), vPrs.find('B')
    if index_r > 0 and index_b > 0:
        vPrs = vPrs[0:min(index_r,index_b)]

    elif index_r > 0:
        vPrs = vPrs[0:index_r]

    elif index_b > 0:
        vPrs = vPrs[0:index_b]

    allSlices = vPrs.split(',')

    for slice in allSlices:
        sliceInts = [int(k) for k in re.findall(r'-*\d+', slice)]
        if ":" not in slice and '::' not in slice:
            vIndices.append(allIndices[sliceInts[0]])
        elif "::" in slice:
            if not sliceInts:
                continue
            elif len(sliceInts)==2:
                toAdd = allIndices[sliceInts[0]::sliceInts[1]]
                vIndices.extend(toAdd)
            else:
                if slice[0]==':':
                    toAdd = allIndices[::sliceInts[0]]
                    if len(toAdd) != size: vIndices.extend(toAdd)
                else:
                    toAdd = allIndices[sliceInts[0]::]
                    if len(toAdd) != size: vIndices.extend(toAdd)
        elif slice.count(":")==2 and ('::') not in slice:
            if len(sliceInts)==3:
                toAdd = allIndices[sliceInts[0]:sliceInts[1]:sliceInts[2]]
                if len(toAdd) != size: vIndices.extend(toAdd)
            elif len(sliceInts)==1:
                toAdd = allIndices[:sliceInts[0]:]
                if len(toAdd) != size: vIndices.extend(toAdd)
            else:
                if slice[0]==':':
                    toAdd = allIndices[:sliceInts[0]:sliceInts[1]]
                    if len(toAdd) != size: vIndices.extend(toAdd)
                else:
                    toAdd = allIndices[sliceInts[0]:sliceInts[1]:]
                    if len(toAdd) != size: vIndices.extend(toAdd)
        else:
            if not sliceInts:
                continue
            elif len(sliceInts)==2:
                toAdd = allIndices[sliceInts[0]:sliceInts[1]]
                vIndices.extend(toAdd)
            else:
                if slice[0]==':':
                    toAdd = allIndices[:sliceInts[0]]
                    vIndices.extend(toAdd)
                else:
                    toAdd = allIndices[sliceInts[0]:]
                    vIndices.extend(toAdd)
    toRet = []
    for v in vIndices:
        if vIndices not in toRet:
            toRet.append(v)
    return toRet

# def slice_parse(size, vectors):
#     indexlist = list(range(size))
#     vertex_index = []
#     vectors = vectors.split(',')
#     for slice in vectors:
#         slice = [int(y) for y in re.findall(r'-*\d+', slice)]
#         if not slice:
#             continue

#         if ":" not in slice and '::' not in slice:
#             vertex_index.append(indexlist[slice[0]])
#         elif "::" in slice:
#             being = slice[0] if len(slice) > 0 else None
#             nxt = slice[1] if len(slice) > 1 else None
#             vertex_index.extend(indexlist[being::nxt])
#         elif slice.count(":") == 2 and '::' not in slice:
#             being = slice[0] if len(slice) > 0 else None
#             finish = slice[1] if len(slice) > 1 else None
#             nxt = slice[2] if len(slice) > 2 else None
#             vertex_index.extend(indexlist[being:finish:nxt])
#         else:
#             being = slice[0] if len(slice) > 0 else None
#             finish = slice[1] if len(slice) > 1 else None
#             vertex_index.extend(indexlist[being:finish])

#     return list(set(vertex_index))

def get_rwd(gr, vertex_direction):
    every = [int(num) for num in re.findall(r'\d+', vertex_direction)]
    rwd = gr['rwd']
    if re.findall(r'R\d+', vertex_direction): rwd = every[-1]
    return rwd

def update_vProps(gr, all_vertices, rwd):
    vProps = gr['vProps']
    for vertex in all_vertices:
        vProps[vertex] = rwd
    gr['vProps'] = vProps
    return gr

def update_edges(gr, all_vertices):
    edges = gr['edges']
    vertex_set = set(all_vertices)
    adjSet = {x for x in range(gr['size'])} - vertex_set
    default = initEdge(gr)

    for vertex in vertex_set:
        for adjV in adjSet:
            vtx = (vertex, adjV)
            if vtx in default:
                if vtx in edges: edges.remove(vtx)
                else: edges.add(vtx)
                if vtx in edges: edges.remove(vtx)
                else: edges.add(vtx)
            else:
                if vtx in edges: edges.remove(vtx)
                if vtx in edges: edges.remove(vtx)

    gr['edges'] = edges
    return gr

# def vDirective(gr, vertex_direction):
#     two = 'B'
#     one = 'R'

#     if one not in vertex_direction and two not in vertex_direction: return gr

#     all_vertices = slice_parse(gr['size'], vertex_direction)
#     rwd = get_rwd(gr, vertex_direction)
#     if one in vertex_direction:
#         gr = update_vProps(gr, all_vertices, rwd)
#     if two in vertex_direction:
#         gr = update_edges(gr, all_vertices)
#     return gr

def vDirective(graph, vDir):
    allInts = [int(k) for k in re.findall(r'\d+', vDir)]
    reward = graph['rwd']
    if re.findall(r'R\d+', vDir):
        reward = allInts[-1]

    if ('R' not in vDir) and ('B' not in vDir):
        return graph

    allV = slice_parse(graph['size'], vDir)
    if 'R' in vDir:
        vertexes = graph['vProps']
        for v in allV:
            vertexes[v] = reward
        graph['vProps'] = vertexes
    if 'B' in vDir:
        edges = graph['eProps']
        W = set(allV)
        X = {i for i in range(graph['size'])} - W
        default = initEdge(graph)

        for h in W:
            for k in X:
                if (h,k) in default:
                    if (h,k) in edges:
                        edges.remove((h,k))
                    else:
                        edges.add((h,k))
                    if (k,h) in edges:
                        edges.remove((k,h))
                    else:
                        edges.add((k,h))
                else:
                    if (h,k) in edges:
                        edges.remove((h,k))
                    if (k,h) in edges:
                        edges.remove((k,h))

        graph['edges'] = edges

    return graph

# def vDirective(graph, vertex_direction):
#     all_numbers = [int(num) for num in re.findall(r'\d+', vertex_direction)]
#     rwd = graph['rwd']
#     if re.findall(r'R\d+', vertex_direction):
#         rwd = all_numbers[-1]

#     if 'R' not in vertex_direction and 'B' not in vertex_direction:
#         return graph

#     all_vertices = slice_parse(graph['size'], vertex_direction)
#     if 'R' in vertex_direction:
#         vProps = graph['vProps']
#         for vertex in all_vertices:
#             vProps[vertex] = rwd
#         graph['vProps'] = vProps
#     if 'B' in vertex_direction:
#         edges = graph['edges']
#         vertex_set = set(all_vertices)
#         complement_set = {x for x in range(graph['size'])} - vertex_set
#         default = initEdge(graph)

#         for vertex in vertex_set:
#             for complement_vertex in complement_set:
#                 if (vertex, complement_vertex) in default:
#                     if (vertex, complement_vertex) in edges:
#                         edges.remove((vertex, complement_vertex))
#                     else:
#                         edges.add((vertex, complement_vertex))
#                     if (complement_vertex, vertex) in edges:
#                         edges.remove((complement_vertex, vertex))
#                     else:
#                         edges.add((complement_vertex, vertex))
#                 else:
#                     if (vertex, complement_vertex) in edges:
#                         edges.remove((vertex, complement_vertex))
#                     if (complement_vertex, vertex) in edges:
#                         edges.remove((complement_vertex, vertex))

#         graph['edges'] = edges

#     return graph


def eDirDepth(graph, edges, edgedir, eDir):
    all_numbers = [int(num) for num in re.findall(r'\d+', eDir)]
    rwd = graph['rwd']
    if re.findall(r'R\d+', eDir):
        rwd = all_numbers[-1]

    current_edges = graph['edges']
    edge_rewards = graph['eProps']

    if edgedir == "!":
        current_edges.difference_update(edges)

    #retsstjvie
    elif edgedir == "*":
        current_edges.update(edges)
        if "R" in eDir:
            for edge in edges:
                edge_rewards[edge] = rwd
    elif edgedir == "@":
        if "R" in eDir:
            for edge in edges:
                edge_rewards[edge] = rwd
    elif edgedir == "~":
        current_edges.symmetric_difference_update(edges)
        if "R" in eDir:
            for edge in edges:
                edge_rewards[edge] = rwd
    elif edgedir == "+":
        current_edges.update(edges)
        if "R" in eDir:
            for edge in edges:
                edge_rewards[edge] = rwd

    graph['edges'] = current_edges
    for edge in graph['edges']:
        if edge not in graph['eProps']:
            graph['eProps'][edge] = -1
            graph['eProps'][edge] = -1

    return graph

def get_management(eDir):
    return eDir[1] if len(eDir) > 1 and eDir[1] in "!+*@" else ""

def get_first_index(eDir):
    return next((idx for idx, ch in enumerate(eDir[2:], start=2) if ch in "NSEW~="), len(eDir)) - 2

def get_v1(eDir, management, first_index):
    v1_slice = 2 + first_index if first_index != -1 else len(eDir)
    return eDir[2:v1_slice] if management != "~" else eDir[1:v1_slice]

def get_second_index(eDir):
    return next((idx for idx, ch in enumerate(eDir[2:], start=2) if ch in "RT"), None)

def get_v2(eDir, v1_slice, second_index):
    v2_end = second_index + 2 if second_index is not None else None
    return eDir[v1_slice + 1:v2_end] if not any(ch in "NSEW" for ch in eDir[1:]) else None

def get_edges(graph, vertex1, vertex2, eDir):
    if vertex2:
        v1s = slice_parse(graph['size'], vertex1)
        v2s = slice_parse(graph['size'], vertex2)
        return list(zip(v1s, v2s))
    else:
        v1s = slice_parse(graph['size'], vertex1)
        return generateEdges(graph, v1s, eDir[1:])

def get_edges_set(edges, doubled):
    edges_set = {(vertex1, vertex2) for (vertex1, vertex2) in edges}
    if doubled:
        edges_set.update({(vertex2, vertex1) for (vertex1, vertex2) in edges})
    return edges_set

def edgeDirective(graph, eDir):
    management = eDir[1] if len(eDir) > 1 and eDir[1] in "!+*@" else ""
    first_index = next((idx for idx, ch in enumerate(eDir[2:], start=2) if ch in "NSEW~="), len(eDir)) - 2
    v1_slice = 2 + first_index if first_index != -1 else len(eDir)
    v1 = eDir[2:v1_slice] if management != "~" else eDir[1:v1_slice]

    second_index = next((idx for idx, ch in enumerate(eDir[2:], start=2) if ch in "RT"), None)
    v2_end = second_index + 2 if second_index is not None else None

    v2 = eDir[v1_slice + 1:v2_end] if not any(ch in "NSEW" for ch in eDir[1:]) else None

    doubled = "=" in eDir

    if v2:
        v1s = slice_parse(graph['size'], v1)
        v2s = slice_parse(graph['size'], v2)
        edges = list(zip(v1s, v2s))
    else:
        v1s = slice_parse(graph['size'], v1)
        edges = generateEdges(graph, v1s, eDir[1:])

    edges_set = {(v1, v2) for (v1, v2) in edges}
    if doubled:
        edges_set.update({(v2, v1) for (v1, v2) in edges})

    return eDirDepth(graph, edges_set, management, eDir)



def get_directions(eDir):
    return re.findall(r'[NSEW]+', eDir[1:])[0]

def add_edge_if_condition(vertex, graph, direction, condition, edge_func):
    if direction == condition and edge_func(vertex, graph):
        return [(vertex, edge_func(vertex, graph))]
    return []


def generateEdges(graph, vProps, eDir):
    edges = []
    for vertex in vProps:
        for direction in get_directions(eDir):
            edges += add_edge_if_condition(vertex, graph, direction, "N", "vertex >= graph['width']")
            edges += add_edge_if_condition(vertex, graph, direction, "E", "vertex % graph['width'] < graph['width'] - 1")
            edges += add_edge_if_condition(vertex, graph, direction, "S", "vertex < graph['size'] - graph['width']")
            edges += add_edge_if_condition(vertex, graph, direction, "W", "vertex % graph['width'] > 0")
    return edges

def maxVals(graph, evals, vertex):
    edge_set = {edge[1] for edge in graph['edges'] if edge[0] == vertex}
    if not edge_set:
        return set()
    max_value = -100000
    optimal_set = set()
    for edge in edge_set:
        edge_reward = graph['eProps'].get((vertex, edge), -1)
        edge_valuation = evals[edge]
        if edge_reward > max_value and edge_reward != -1:
            optimal_set = {edge}
            max_value = edge_reward
        elif edge_valuation > max_value and edge_reward == -1:
            optimal_set = {edge}
            max_value = edge_valuation
        elif edge_valuation == max_value or (edge_reward == max_value and edge_reward != -1):
            optimal_set.add(edge)
    return optimal_set

def initialize_evals_and_endState(graph):
    evals = []
    endState = []
    for x in range(graph['size']):
        value = graph['vProps'][x]
        if value != 'none':
            evals.append(value)
            endState.append(x)
        else:
            evals.append('')
    return evals, endState

def replace_empty_with_zeros(evals):
    for x in range(len(evals)):
        if evals[x] == '':
            evals[x] = 0
    return evals

def calculate_val(vertex, edgeSet, graph, prevValuation, gamma):
    val = 0
    if edgeSet:
        for edge in edgeSet:
            if (vertex, edge) in graph['eProps'] and graph['eProps'][(vertex, edge)] != -1:
                val += graph['eProps'][(vertex, edge)]
            else:
                val += prevValuation[edge]
        val /= len(edgeSet)
        if gamma > 0.5:
            val *= gamma
        else:
            val -= gamma
    return val

def replace_zeros_with_empty(evals):
    for x in range(len(evals)):
        if evals[x] == 0:
            evals[x] = ''
    return evals

def grfValuePolicy(graph, plcy, gamma):
    evals, endState = initialize_evals_and_endState(graph)
    if graph['width'] == 0:
        return evals
    evals = replace_empty_with_zeros(evals)
    delta = 100000
    while(delta > .001):
        prevValuation = evals.copy()
        for vertex in range(len(plcy)):
            if vertex not in endState:
                evals[vertex] = calculate_val(vertex, plcy[vertex], graph, prevValuation, gamma)
        delta = max(abs(evals[x] - prevValuation[x]) for x in range(len(evals)))
    evals = replace_zeros_with_empty(evals)
    return evals

def grfPolicyFromValuation(graph, evals):
    plcy = []
    for x in range(graph['size']):
        if graph['vProps'][x] != 'none':
            plcy.append(set())
        else:
            plcy.append(maxVals(graph, evals, x))
    return plcy

def grfFindOptimalPolicy(graph):
    # Initialize the evals list
    evals = []
    for x in range(graph['size']):
        value = graph['vProps'][x]
        if value != 'none':
            evals.append(value)
        else:
            evals.append(0)

    # If the graph width is zero, return a list of empty sets
    if graph['width'] == 0:
        return [set() for _ in range(graph['size'])]

    # Initialize the plcy and previous plcy lists
    plcy = grfPolicyFromValuation(graph, evals)
    previous_policy = [{'temp'} for _ in range(graph['size'])]

    # Iterate until the plcy and previous plcy are equal
    while plcy != previous_policy:
        # Update the evals and previous plcy
        evals = grfValuePolicy(graph, (previous_policy := plcy), .01)
        plcy = grfPolicyFromValuation(graph, evals)

    # Round the non-empty values in the evals list to 4 decimal places
    for x in range(len(evals)):
        if evals[x] != '':
            evals[x] = round(evals[x], 4)

    # Return the plcy from the updated evals
    return grfPolicyFromValuation(graph, evals)


#TO CHANGE
def get_evals(graph):
    evals = []
    for x in range(graph['size']):
        value = graph['vProps'][x]
        if value != 'none':
            evals.append(value)
        else:
            evals.append(0)
    return evals

def format_evals(evals):
    temp = []
    for vert in evals:
        if vert == 0:
            temp.append(f" {'00'}")
        else:
            temp.append(f" {float(vert):.5g}"[-6:])
    return temp

def get_policy(graph, evals):
    plcy = grfPolicyFromValuation(graph, evals)
    pastPlcy = [{'temp'} for x in range(graph['size'])]
    while(plcy != pastPlcy):
        evals = grfValuePolicy(graph, (pastPlcy := plcy), .01)
        plcy = grfPolicyFromValuation(graph, evals)
    return evals

def format_toRet(temp, graph):
    toRet = ''
    for x in range(len(temp)):
        if x % graph['width'] == 0:
            toRet += temp[x][1:]
        elif x % graph['width'] == graph['width'] - 1:
            toRet = toRet + temp[x] + '\n'
        else:
            toRet += temp[x]
    return toRet[:-1]

def getValuation(graph):
    if graph['width'] == 0:
        evals = get_evals(graph)
        temp = format_evals(evals)
        return "".join(temp)[1:]
    else:
        evals = get_evals(graph)
        evals = get_policy(graph, evals)
        temp = format_evals(evals)
        return format_toRet(temp, graph)


def findDir(wdth, y, x):
    return {wdth: 'N', -wdth: 'S', 1: 'W', -1: 'E'}.get(y - x, '')


def grfParse(args):
    # Initialize the graph dictionary with default values
    graph = {
        'gametype': 'G', 
        'size': 0, 
        'width': 0, 
        'rwd': 12, 
        'edges': set(), 
        'vProps': {}, 
        'eProps': {}, 
        'gdir': args[0]
    }

    # Get dimensions from the first argument
    size, width, rwd = parseArgs(args[0])

    # Update the graph dictionary with the dimensions
    graph.update({
        'size': size,
        'width': width,
        'rwd': rwd,
        'vProps': {x: 'none' for x in range(size)}
    })

    # If 'N' is in the first argument, set the width to 0
    if 'N' in args[0]:
        graph['width'] = 0

    # If the width is greater than 0, update the edges and eProps
    if graph['width'] > 0:
        dflts = initEdge(graph)
        graph['edges'] = dflts
        graph['eProps'] = {tup: -1 for tup in dflts}

    # If there are more than one arguments, process the vProps and edges
    if len(args) > 1:
        for arg in args[1:]:
            if arg[0] == 'V':
                graph = vDirective(graph, arg[1:])
            if arg[0] == 'E':
                graph = edgeDirective(graph, arg)

    return graph

def initialize_allMoves(graph):
    return {i:set() for i in range(graph['size'])}

def update_allMoves(tempDirs, graph, width):
    for tup in graph['edges']:
        k = tup[0]
        i = tup[1]
        if findDir(width, k, i): tempDirs[k].add(findDir(width, k, i))
    return tempDirs

def add_default_direction(tempDirs):
    for i in tempDirs:
        if not tempDirs[i]:
            tempDirs[i].add('.')
    return tempDirs

def get_moveStr(tempDirs, dirdict, width):
    nxtStr = "".join(dirdict.get("".join(sorted(dirs))) for dirs in tempDirs.values())
    newDir = [nxtStr[i:i+width] for i in range(0, len(nxtStr), width)]
    if newDir: nxtStr = '\n'.join(newDir)
    return nxtStr

def get_v1_v2(graph, width):
    vertex1, vertex2 = '',''
    for tup in graph['edges']:
        if not (abs(tup[0] - tup[1]) == 1 or abs(tup[0] - tup[1]) == width):
            vertex1 = vertex1 + str(tup[0]) + ','
            vertex2 = vertex2+ str(tup[1]) + ','
    return vertex1, vertex2

def grfStrEdges(graph):
    width = graph['width']
    if width == 0:
        return ''
    tempDirs = initialize_allMoves(graph)
    tempDirs = update_allMoves(tempDirs, graph, width)
    tempDirs = add_default_direction(tempDirs)

    dirdict = { #DICT of direction list ultimate directions
        'N':'N',
        'E':'E',
        'S':'S',
        'W':'W',
        'EN':'L',
        'NW':'J',
        'SW':'7',
        'ES':'r',
        'EW': '-',
        'NS': '|',
        'ENW':'^',
        'ENS':'>',
        'ESW':'v',
        'NSW':'<',
        'ENSW':'+',
        '':'',
        '.':'.'}

    nxtStr = get_moveStr(tempDirs, dirdict, width)
    vertex1, vertex2 = get_v1_v2(graph, width)

    if vertex1:
        return nxtStr + '\n' +  vertex1[:-1] + '~' + vertex2[:-1]
    return nxtStr



def grfStrProps(graph):
    # Get props of the graph
    props = grfGProps(graph)

    # If 'N' is in graph direction, return rwd
    if 'N' in graph['gdir']: 
        return f"rwd: {props['rwd']}"

    # Create a string of props
    properties_str = ', '.join(f"{key}: {value}" for key, value in props.items())

    # Add vertex rewards to the string
    vertex_rewards = "\n".join(f"{x}: rwd: {graph['vProps'][x]}" for x in graph['vProps'] if graph['vProps'][x] != 'none')

    # Add edge rewards to the string
    edge_rewards = "\n".join(f"{tup}: rwd: {graph['eProps'][tup]}" for tup in graph['eProps'] if graph['eProps'][tup] != -1 and tup in graph['edges'])

    # Combine all strings and return
    return "\n".join(filter(None, [properties_str, vertex_rewards, edge_rewards]))


def initialize_maxDirs(graph):
    return {x:set() for x in range(graph['size'])}

def update_maxDirs(maxDirs, plcy, width):
    for y in range(len(plcy)):
        for x in plcy[y]:
            if findDir(width, y, x): maxDirs[y].add(findDir(width, y, x))
    return maxDirs

def add_default_direction2(maxDirs, graph):
    for x in maxDirs:
        if not maxDirs[x]:
            if graph['vProps'][x] != 'none':
                maxDirs[x].add('*')
            else:
                maxDirs[x].add('.')
    return maxDirs

def get_temp1(maxDirs, dirsDict, width):
    temp1 = "".join(dirsDict.get("".join(sorted(dirs))) for dirs in maxDirs.values())
    nxt = [temp1[x:x+width] for x in range(0, len(temp1), width)]
    if nxt: temp1 = '\n'.join(nxt)
    return temp1

def get_vertex1_vertex2(plcy, width):
    vertex1, vertex2 = '', ''
    for y in range(len(plcy)):
        for x in plcy[y]:
            if findDir(width, y, x) == None:
                vertex1 = vertex1 + str(y) + ','
                vertex2 = vertex2 + str(x) + ','
    return vertex1, vertex2

def grfOptimalStr(graph, plcy):
    width = graph['width']
    if width == 0:
        return '.'*graph['size']
    maxDirs = initialize_maxDirs(graph)
    maxDirs = update_maxDirs(maxDirs, plcy, width)
    maxDirs = add_default_direction2(maxDirs, graph)

    dirsDict = {
        'N': 'N', 'E': 'E', 'S': 'S', 'W': 'W',
        'EN': 'L', 'NW': 'J', 'SW': '7', 'ES': 'r',
        'EW': '-', 'NS': '|', 'ENW': '^', 'ENS': '>',
        'ESW': 'vert', 'NSW': '<', 'ENSW': '+', '': '', '.': '.'
    }

    temp1 = get_temp1(maxDirs, dirsDict, width)
    vertex1, vertex2 = get_vertex1_vertex2(plcy, width)

    if vertex1:
        return temp1 + '\nJumps: ' +  vertex1[:-1] + '~' + vertex2[:-1]
    return temp1



def main():
    # args = 'GG45 E~15,18,41,4=37,14,5,42R E*38,31=27,44R13 V25B'.split(' ')
    # args = 'GG44 V4::11B V20R34 V18R25'.split(' ')
    # args = 'GN4W23 V3R'.split(' ')
    args = 'GG25W5 V24:-10:-3B V21R70 V22R15'.split(' ')
    graph = grfParse(args)
    print(grfNbrs(graph, 13))
    print('Graph:')
    grfStr = grfStrEdges(graph)
    if grfStr: print(grfStr)
    print(grfStrProps(graph))
    print('Optimal plcy:')
    print(grfOptimalStr(graph, grfFindOptimalPolicy(graph)))
    print('Valuation:')
    print(getValuation(graph))

main()

#Om Gole, 6, 2024