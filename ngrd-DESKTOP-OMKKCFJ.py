import sys; args = sys.argv[1:]
import re, math

def calcWidth(size):
    for x in range(1, size + 1):
        if size % x == 0 and x >= math.sqrt(size):
            return x
    return size

def parseDirectives(directives):
    graph = {
        'size': None,
        'width': None,
        'rwd': 12,
        'rewards': None,
        'blocks': set(),
        'goalType': 'G1'
    }

    graph['size'] = int(directives[0])
    graph['width'] = int(directives[1]) if len(directives) > 1 and directives[1].isdigit() else calcWidth(graph['size'])

    graph['rewards'] = [-float('inf')] * graph['size']

    for directive in directives[1:]:
        if directive.startswith('R') or directive.startswith('r'):
            if ':' in directive:
                cur = directive[1:].split(':')
                if cur[0] == '':
                    graph['rwd'] = int(cur[1])
                else:
                    cell, reward = map(int, cur)
                    graph['rewards'][cell] = reward
            elif directive[1:].isdigit():
                cell = int(directive[1:])
                graph['rewards'][cell] = graph['rwd']
            else:
                graph['rwd'] = int(directive[1:])
        elif directive.startswith('B'):
            match = re.match(r'^B(\d*)([NSEW]*)$', directive)
            if match:
                cell_str, directions_str = match.groups()
                if len(directions_str):
                    cell = int(cell_str)
                    for d in directions_str:
                        if d == 'N' and cell // graph['width']:
                            if ((cell, cell - graph['width']) in graph['blocks']):
                                graph['blocks'].remove((cell, cell - graph['width']))
                                graph['blocks'].remove((cell - graph['width'], cell))
                            else:
                                graph['blocks'].add((cell, cell - graph['width']))
                                graph['blocks'].add((cell - graph['width'], cell))
                        elif d == 'S' and cell // graph['width'] < (graph['size'] // graph['width'] - 1):
                            if ((cell, cell + graph['width']) in graph['blocks']):
                                graph['blocks'].remove((cell, cell + graph['width']))
                                graph['blocks'].remove((cell + graph['width'], cell))
                            else:
                                graph['blocks'].add((cell, cell + graph['width']))
                                graph['blocks'].add((cell + graph['width'], cell))
                        elif d == 'E' and cell % graph['width'] < (graph['width'] - 1):
                            if ((cell, cell + 1) in graph['blocks']):
                                graph['blocks'].remove((cell, cell + 1))
                                graph['blocks'].remove((cell + 1, cell))
                            else:
                                graph['blocks'].add((cell, cell + 1))
                                graph['blocks'].add((cell + 1, cell))
                        elif d == 'W' and cell % graph['width']:
                            if ((cell, cell - 1) in graph['blocks']):
                                graph['blocks'].remove((cell, cell - 1))
                                graph['blocks'].remove((cell - 1, cell))
                            else:
                                graph['blocks'].add((cell, cell - 1))
                                graph['blocks'].add((cell - 1, cell))
                else:
                    cell = int(directive[1:])
                    width = graph['width']
                    if cell // width:
                        if ((cell, cell - graph['width']) in graph['blocks']):
                            graph['blocks'].remove((cell, cell - graph['width']))
                            graph['blocks'].remove((cell - graph['width'], cell))
                        else:
                            graph['blocks'].add((cell, cell - graph['width']))
                            graph['blocks'].add((cell - graph['width'], cell))
                    if cell // width < (graph['size'] // width - 1):
                        if ((cell, cell + graph['width']) in graph['blocks']):
                            graph['blocks'].remove((cell, cell + graph['width']))
                            graph['blocks'].remove((cell + graph['width'], cell))
                        else:
                            graph['blocks'].add((cell, cell + graph['width']))
                            graph['blocks'].add((cell + graph['width'], cell))
                    if cell % width < (width - 1):
                        if ((cell, cell + 1) in graph['blocks']):
                            graph['blocks'].remove((cell, cell + 1))
                            graph['blocks'].remove((cell + 1, cell))
                        else:
                            graph['blocks'].add((cell, cell + 1))
                            graph['blocks'].add((cell + 1, cell))
                    if cell % width:
                        if ((cell, cell - 1) in graph['blocks']):
                            graph['blocks'].remove((cell, cell - 1))
                            graph['blocks'].remove((cell - 1, cell))
                        else:
                            graph['blocks'].add((cell, cell - 1))
                            graph['blocks'].add((cell - 1, cell))
        elif directive == 'G0':
            graph['goalType'] = 'G0'
        elif directive == 'G1':
            graph['goalType'] = 'G1'

    return graph

def getNeighbors(cell, graph):
    neighbors = []
    size = graph['size']
    width = graph['width']
    if cell >= width and (cell, cell - width) not in graph['blocks']:
        neighbors.append(cell - width)
    if cell % width < width - 1 and (cell, cell + 1) not in graph['blocks']:
        neighbors.append(cell + 1)
    if cell < size - width and (cell, cell + width) not in graph['blocks']:
        neighbors.append(cell + width)
    if cell % width > 0 and (cell, cell - 1) not in graph['blocks']:
        neighbors.append(cell - 1)
    return neighbors

def getDirection(src, dest, width):
    if dest == src - width:
        return 'U'
    elif dest == src + 1:
        return 'R'
    elif dest == src + width:
        return 'D'
    elif dest == src - 1:
        return 'L'
    return ''

def calculateOptimalDirections(graph):
    size = graph['size']
    width = graph['width']
    rewards = graph['rewards']
    goalType = graph['goalType']
    inf = float('inf')
    directions = [[[]] * width for _ in range(size // width)]
    queue = []
    rew = []
    for i in range(size):
        if rewards[i] != -inf: 
            queue.append((i, 0, rewards[i]))
            rew.append(i)
    if graph['goalType'] == 'G0':
        maxRewards = [[-inf] * width for _ in range(size // width)]
        steps = [[inf] * width for _ in range(size // width)]
        for x in rew: 
            maxRewards[x // width][x % width] = rewards[x] 
            steps[x // width][x % width] = 0
        visited = {}
        while queue:
            top = queue.pop(0)
            neighbors = getNeighbors(top[0], graph)
            for neighbor in neighbors:
                x, y = neighbor // width, neighbor % width
                if ((neighbor, top[2]) in visited and visited[(neighbor, top[2])]) or neighbor in rew: continue
                visited[(neighbor, top[2])] = True
                if top[2] > maxRewards[x][y]:
                    maxRewards[x][y] = max(maxRewards[x][y], top[2])
                    steps[x][y] = top[1] + 1
                elif top[2] == maxRewards[x][y]:
                    steps[x][y] = min(steps[x][y], top[1] + 1)
                queue.append((neighbor, top[1] + 1, top[2]))
        for i in range(size):
            x1, y1 = i // width, i % width
            neighbors = getNeighbors(i, graph)
            dirs = ''
            for neighbor in neighbors:
                x2, y2 = neighbor // width, neighbor % width
                if (maxRewards[x1][y1] == maxRewards[x2][y2]) and (steps[x2][y2] + 1 == steps[x1][y1]) and maxRewards[x1][y1] != -inf:
                    dirs += getDirection(i, neighbor, width)
            directions[x1][y1] = dirs
    else:
        maxRatio = [[-inf] * width for _ in range(size // width)]
        ok = [[] for _ in range(size)]
        steps = [[inf] * width for _ in range(size // width)]
        for x in rew: 
            steps[x // width][x % width] = 0
        visited = {}
        while queue:
            top = queue.pop(0)
            neighbors = getNeighbors(top[0], graph)
            for neighbor in neighbors:
                x, y = neighbor // width, neighbor % width
                ratio = top[2] / (top[1] + 1)
                if ratio > maxRatio[x][y]:
                    maxRatio[x][y] = ratio
                    ok[neighbor].clear()
                    ok[neighbor].append(top[0])
                    steps[x][y] = top[1] + 1
                elif ratio == maxRatio[x][y]:
                    ok[neighbor].append(top[0])
                if ((neighbor, top[2]) in visited and visited[(neighbor, top[2])]) or neighbor in rew: continue
                visited[(neighbor, top[2])] = True
                queue.append((neighbor, top[1] + 1, top[2]))
        for i in range(size):
            x1, y1 = i // width, i % width
            neighbors = getNeighbors(i, graph)
            dirs = ''
            for neighbor in neighbors:
                if maxRatio[x1][y1] != -inf and neighbor in ok[i]:
                    dirs += getDirection(i, neighbor, width)
            directions[x1][y1] = dirs
    return directions

def directionSymbols(directions):
    symbols = {
        '': '.', 'U': 'U', 'R': 'R', 'D': 'D', 'L': 'L',
        'RU': 'V', 'DRU': 'W', 'DR': 'S', 'DLR': 'T', 'DL': 'E',
        'DLU': 'F', 'LU': 'M', 'LRU': 'N', 'DU': '|', 'LR': '-',
        'DLRU': '+'
    }
    return symbols[''.join(sorted(directions))]

def displayGrid(graph, optimalDirections):
    size = graph['size']
    width = graph['width']
    result = ''
    for i in range(size):
        if graph['rewards'][i] > 0:
            result += '*'
        else:
            result += directionSymbols(optimalDirections[i // width][i % width])
    for i in range(0, size, width):
        print(result[i:i+width])

def main():
    graph = parseDirectives(args)
    optimalDirections = calculateOptimalDirections(graph)
    displayGrid(graph, optimalDirections)

if __name__ == "__main__":
    main()

# Om Gole, 6, 2024