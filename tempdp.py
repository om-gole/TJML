import sys; args = sys.argv[1:]
import re

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

def slice_parse(size, vipers):
    indexes = [x for x in range(size)]; vertices = []; r, y = vipers.find('R'), vipers.find('B'); vipers = vipers[0:min(r,y)] if r > 0 and y > 0 else vipers[0:r] if r > 0 else vipers[0:y] if y > 0 else vipers; sliceee = vipers.split(',')

    for gangshi in sliceee:
        skibslicers = [int(k) for k in re.findall(r'-*\d+', gangshi)]
        if ":" not in gangshi and '::' not in gangshi:
            vertices.append(indexes[skibslicers[0]])
        elif "::" in gangshi:
            extend_indices_squig(indexes, skibslicers, size, vertices, gangshi)
        elif gangshi.count(":")==2 and ('::') not in gangshi:
            extend_indices(indexes, skibslicers, size, vertices)
        #movew on
        else:
            if not skibslicers:
                continue
            if len(skibslicers)==2:
                temp = indexes[skibslicers[0]:skibslicers[1]]
                vertices.extend(temp)
            else:
                temp = indexes[:skibslicers[0]] if gangshi[0] == ':' else indexes[skibslicers[0]:]
                vertices.extend(temp)

    # for gangshi in sliceee:
    #     skibslicers = [int(k) for k in re.findall(r'-*\d+', gangshi)]
    #     if not skibslicers: continue

    #     if "::" in gangshi:
    #         step = skibslicers[1] if len(skibslicers) == 2 else skibslicers[0] if gangshi[0] == ':' else None
    #         start = skibslicers[0] if step is not None else None
    #         temp = indexes[start::step] if start is not None else indexes[::step]
    #     elif ":" in gangshi:
    #         start, end, step = (skibslicers + [None, None])[:3]
    #         if gangshi[0] == ':': start = None
    #         temp = indexes[start:end:step]
    #     else:
    #         temp = [indexes[skibslicers[0]]]

    #     if len(temp) != size: vertices.extend(temp)
    
    returning = []
    returning = [x for x in vertices if x not in returning]

    return returning

def extend_indices_squig(indexes, skibslicers, size, vertices, gangshi):
    if not skibslicers:
        return
    elif len(skibslicers) == 2:
        temp = indexes[skibslicers[0]::skibslicers[1]]
    else:
        if gangshi[0] == ':':
            temp = indexes[::skibslicers[0]]
        else:
            temp = indexes[skibslicers[0]::]
    if len(temp) != size:
        vertices.extend(temp)

def extend_indices(indexes, skibslicers, size, vertices):
    if len(skibslicers) == 3:
        temp = indexes[skibslicers[0]:skibslicers[1]:skibslicers[2]]
    elif len(skibslicers) == 1:
        temp = indexes[:skibslicers[0]]
    else:
        temp = indexes[skibslicers[0]:skibslicers[1]:]
    if len(temp) != size:
        vertices.extend(temp)

def vDirective(graph, vertex_direction):
    all_numbers = [int(num) for num in re.findall(r'\d+', vertex_direction)]
    rwd = graph['rwd']
    if re.findall(r'R\d+', vertex_direction):
        rwd = all_numbers[-1]

    if 'R' not in vertex_direction and 'B' not in vertex_direction:
        return graph

    all_vertices = slice_parse(graph['size'], vertex_direction)
    if 'R' in vertex_direction:
        vProps = graph['vProps']
        for vertex in all_vertices:
            vProps[vertex] = rwd
        graph['vProps'] = vProps
    if 'B' in vertex_direction:
        edges = graph['edges']
        vertex_set = set(all_vertices)
        complement_set = {x for x in range(graph['size'])} - vertex_set
        default = initEdge(graph)
        #FOR LOOP IS GOOD
        for vertex in vertex_set:
            for complement_vertex in complement_set:
                if (vertex, complement_vertex) in default:
                    if (vertex, complement_vertex) in edges: edges.remove((vertex, complement_vertex))
                    else: edges.add((vertex, complement_vertex))
                    if (complement_vertex, vertex) in edges: edges.remove((complement_vertex, vertex))
                    else: edges.add((complement_vertex, vertex))
                else:
                    if (vertex, complement_vertex) in edges: edges.remove((vertex, complement_vertex))
                    if (complement_vertex, vertex) in edges: edges.remove((complement_vertex, vertex))

        graph['edges'] = edges

    return graph

def edgeProcess(graph, edges, eArgs, EdgeDir):
    bangout = []
    for k in re.findall(r'\d+', EdgeDir):
        bangout.append(int(k))
    reward = graph['rwd'] if not re.findall(r'R\d+', EdgeDir) else bangout[-1]; current, eProps = graph['edges'], graph['eProps']
    edges_set = set(edges)
    if eArgs == "!":
        current -= edges_set
    elif eArgs == "@":
        common = edges_set & current
        if "R" in EdgeDir:
            eProps.update({tup: reward for tup in common})
    elif eArgs in ["+", "*"]:
        new_edges = edges_set - current if eArgs == "+" else edges_set
        current |= new_edges
        if "R" in EdgeDir:
            eProps.update({tup: reward for tup in new_edges})
    elif eArgs == "~":
        to_remove = edges_set & current
        to_add = edges_set - current
        current -= to_remove
        current |= to_add
        if "R" in EdgeDir:
            eProps.update({tup: reward for tup in to_add})

    graph['edges'], graph['eProps'] = current, eProps
    for tup in graph['edges']: graph['eProps'][tup] = -1 if tup not in graph['eProps'] else graph['eProps'][tup]
    return graph

def process_directions(EdgeDir):
    # Initialize variables
    eArgs, finalIndex, gangshi, temp1 = None, -1, None, None

    # Check if the second character in EdgeDir is in "!+*~@"
    if len(EdgeDir) > 1 and EdgeDir[1] in "!+*~@":
        eArgs = EdgeDir[1]
        # Find the index of the first occurrence of "NSEW~=" starting from the third character
        for y in range(2, len(EdgeDir)):
            if EdgeDir[y] in "NSEW~=":
                finalIndex = y - 2
                break
        # Determine the gangshi index
        gangshi = 2 + finalIndex if finalIndex != -1 else len(EdgeDir)
        # Extract the substring from the third character to the gangshi index
        temp1 = EdgeDir[2:gangshi]
    else:
        eArgs = "~"
        # Find the index of the first occurrence of "NSEW~=" starting from the second character
        for y in range(1, len(EdgeDir)):
            if EdgeDir[y] in "NSEW~=":
                finalIndex = y - 1
                break
        # Determine the gangshi index
        gangshi = finalIndex + 1 if finalIndex != -1 else len(EdgeDir)
        # Extract the substring from the second character to the gangshi index
        temp1 = EdgeDir[1:gangshi]
    return eArgs, temp1


def edgeDirective(graph, EdgeDir):
    if len(EdgeDir) > 1 and EdgeDir[1] in "!+*~@":
        eArgs, finalIndex = EdgeDir[1], -1
        finalIndex = next((y - 2 for y in range(2, len(EdgeDir)) if EdgeDir[y] in "NSEW~="), None)
        if finalIndex != -1:
            gangshi = 2 + finalIndex
        else:
            gangshi = len(EdgeDir)
        if True:
            temp1 = EdgeDir[2:gangshi]
    else:
        eArgs, finalIndex = "~", -1
        finalIndex = next((y - 1 for y in range(1, len(EdgeDir)) if EdgeDir[y] in "NSEW~="), None)

        if finalIndex != -1:
            gangshi = finalIndex + 1
        else:
            gangshi = len(EdgeDir)

        if True:
            temp1 = EdgeDir[1:gangshi]

    startIndex = -1
    for y in range(2, len(EdgeDir)): 
        if EdgeDir[y] in "RT": startIndex = y - 2; break

    v2_end = startIndex + 2 if startIndex != -1 else None
    temp2 = EdgeDir[gangshi + 1:v2_end] if not any(ch in "NSEW" for ch in EdgeDir[1:]) else None

    doubled = "=" in EdgeDir

    if temp2: temp2start = slice_parse(graph['size'], temp1); temp1start = slice_parse(graph['size'], temp2); edges = list(zip(temp2start, temp1start))
    else: temp2start = slice_parse(graph['size'], temp1); edges = generateEdges(graph, temp2start, EdgeDir[1:])

    edges_set = set()
    for (temp1, temp2) in edges: 
        edges_set.update({(temp1, temp2), (temp2, temp1)} if doubled else {(temp1, temp2)})


    edges = edges_set

    return edgeProcess(graph, edges, eArgs, EdgeDir)

def generateEdges(graph, vertices, edge_direction):
    edges = []
    # Adding intercardinal directions to the compass
    directions = re.findall(r'[NSEW]{1,2}', edge_direction[1:])[0]
    width = graph['width']
    size = graph['size']

    for vertex in vertices:
        for direction in directions:
            north, south, east, west = range(4)
            # Determine the offset for each direction
            offsets = {
                'N': -width,
                'S': width,
                'E': 1,
                'W': -1,
                'NE': -width + 1,
                'NW': -width - 1,
                'SE': width + 1,
                'SW': width - 1,
            }
            # Check if the vertex is on the appropriate edge for the direction
            checks = {
                'N': vertex >= width,
                'S': vertex < size - width,
                'E': vertex % width < width - 1,
                'W': vertex % width > 0,
                'NE': vertex >= width and vertex % width < width - 1,
                'NW': vertex >= width and vertex % width > 0,
                'SE': vertex < size - width and vertex % width < width - 1,
                'SW': vertex < size - width and vertex % width > 0,
            }
            # Add edge if the direction is valid for the current vertex
            if direction in checks and checks[direction]:
                edges.append((vertex, vertex + offsets[direction]))
    return edges

def maxVals(graph, valuation, vertex):
    edgeSet, maxVal, optimal = set(), float('-inf'), set()
    # Replace the comprehension with a for loop to populate edgeSet
    for edge in graph['edges']:
        if edge[0] == vertex:
            edgeSet.add(edge[1])

    # Check if edgeSet is empty and return an empty set if true
    if not edgeSet:
        return set()
    for edge in edgeSet:
        if (vertex, edge) in graph['eProps']:
            if graph['eProps'][(vertex, edge)] > maxVal:
                if graph['eProps'][(vertex,edge)] != -1:
                    optimal = {edge}
                    maxVal = graph['eProps'][(vertex,edge)]
        if valuation[edge]:
            if valuation[edge] > maxVal:
                if graph['eProps'][(vertex, edge)] == -1:
                    optimal = {edge}
                    maxVal = valuation[edge]
        if valuation[edge]:
            if valuation[edge] == maxVal:
                optimal.add(edge)
        if (vertex, edge) in graph['eProps']:
            if graph['eProps'][(vertex, edge)] == maxVal:
                if graph['eProps'][(vertex,edge)] != -1:
                    optimal.add(edge)
    return optimal

def initialize_evals_and_endState(graph):
    vals, final = [], []
    for x in range(graph['size']):
        value = graph['vProps'][x]
        if value != 'none':
            vals.append(value)
            final.append(x)
        else:
            vals.append('')
    return vals, final

def replace_empty_with_zeros(vals):
    for x in range(len(vals)):
        if vals[x] == '':
            vals[x] = 0
    return vals

def calculate_val(vertex, edgeSet, graph, prevValuation, gamma):
    grad = 0
    if edgeSet:
        for edge in edgeSet:
            grad += graph['eProps'][(vertex, edge)] if (vertex, edge) in graph['eProps'] and graph['eProps'][(vertex, edge)] != -1 else prevValuation[edge]
        grad /= len(edgeSet)
        grad = grad * gamma if gamma > 0.5 else grad - gamma

    return grad

def replace_zeros_with_empty(vals):
    for x in range(len(vals)):
        if vals[x] == 0:
            vals[x] = ''
    return vals

def grfValuePolicy(graph, plcy, gamma):
    vals, final = initialize_evals_and_endState(graph)
    if graph['width'] == 0:
        return vals
    vals = replace_empty_with_zeros(vals)
    delta = 100000
    while(delta > .001):
        prevValuation = vals.copy()
        for vertex in range(len(plcy)):
            if vertex not in final:
                vals[vertex] = calculate_val(vertex, plcy[vertex], graph, prevValuation, gamma)
        delta = max(abs(vals[x] - prevValuation[x]) for x in range(len(vals)))
    vals = replace_zeros_with_empty(vals)
    return vals

# def grfPolicyFromValuation(graph, vals):
def grfPolicyFromValuation(graph, vals):
    plcy = []
    for x in range(graph['size']):
        plcy.append(set() if graph['vProps'][x] != 'none' else maxVals(graph, vals, x))
    return plcy


def grfFindOptimalPolicy(graph):
    # Initialize the vals list
    graph_size = graph['size']
    lastplcy, vals = [],[]
    
    for x in range(graph_size):
        value = graph['vProps'][x]
        vals.append(value if value != 'none' else 0)
    if graph['width'] == 0:
        result = []
        for _ in range(graph['size']):
            result.append(set())
        return result
    plcy = grfPolicyFromValuation(graph, vals)

    # Initialize the plcy and previous plcy lists
    for gabor in range(graph_size):
        temp_dict = {'temp'}
        lastplcy.append(temp_dict)
    # Iterate until the plcy and previous plcy are equal
    while True:
        vals = grfValuePolicy(graph, plcy, .01)
        new_plcy = grfPolicyFromValuation(graph, vals)
        if new_plcy == plcy:
            break
        lastplcy, plcy = plcy, new_plcy


    # Round the non-empty values in the vals list to 4 decimal places
    for x in range(len(vals)):
        if vals[x] != '':
            if vals[x]:
                vals[x] = round(vals[x], 4)
    return grfPolicyFromValuation(graph, vals)



def get_evals(graph):
    vals = list()
    for x in range(graph['size']):
        vals.append(graph['vProps'][x] if graph['vProps'][x] != 'none' else 0)
    return vals

def format_evals(vals):
    return [f" {'00'}" if vert in [0, ''] else f" {float(vert):.5g}"[-6:] for vert in vals]


def get_policy(graph, vals):
    plcy = grfPolicyFromValuation(graph, vals)
    pastPlcy = [{'temp'} for x in range(graph['size'])]
    while(plcy != pastPlcy):
        vals = grfValuePolicy(graph, (pastPlcy := plcy), .01)
        plcy = grfPolicyFromValuation(graph, vals)
    return vals

def format_toRet(temp, graph):
    width = graph['width']
    return '\n'.join(''.join(temp[x][1:] if x % width == 0 else temp[x] if x % width == width - 1 else temp[x] for x in range(len(temp))).splitlines())


def getValuation(graph):
    if graph['width'] == 0:
        vals = get_evals(graph)
        temp = format_evals(vals)
        return "".join(temp)[1:]
    else:
        vals = get_evals(graph)
        vals = get_policy(graph, vals)
        temp = format_evals(vals)
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
    return {x:set() for x in range(graph['size'])}

def update_allMoves(tempDirs, graph, width):
    for tup in graph['edges']:
        k = tup[0]
        x = tup[1]
        if findDir(width, k, x): tempDirs[k].add(findDir(width, k, x))
    return tempDirs

def add_default_direction(tempDirs):
    for x in tempDirs:
        if not tempDirs[x]:
            tempDirs[x].add('.')
    return tempDirs

def get_moveStr(tempDirs, dirdict, width):
    nxtStr = "".join(dirdict.get("".join(sorted(dirs))) for dirs in tempDirs.values())
    newDir = [nxtStr[x:x+width] for x in range(0, len(nxtStr), width)]
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
        'ESW':'x',
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
    temp1 = "".join(dirsDict.get("".join(sorted(dirs)), "") for dirs in maxDirs.values())
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
    # args = 'GG44 V4::11B V20R34 V18R25'.split(' ')
    # args = 'GN4W23 V3R'.split(' ')
    # args = 'GG13 E~10~9 V8R21 V1R33'.split(' ')
    graph = grfParse(args)
    # print(grfNbrs(graph, 10))
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