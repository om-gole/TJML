import sys; args = sys.argv[1:]
import re
import math
#pythonu rlsetup.py 16 <- SIZE u? <- Width: you are either solving G0 or G1.

#G0

#G0 is the same thing with gridworld, but trying to find the maximum reward

#Default to G0 or G1

#3 reward directives: R6 (default RWD) R 6:u (vertex and RWD) R:5 (sets default reward), if none, default is 12
# B# is a blocking vertex || B#SE (Vertex number and direction) -> toggle edge.

def grfParse(args):
    #default props
    graph = {}
    nbrs, nbrsStr, vprops, eprops, jumps, width, dRwd, size, type, gametype = [], [], [], {}, [], 0, 12, 0, "N", "G1"
    size = int(args[0])   
    width = getWidth(size)
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

    for arg in args[1:]:
        gdir = None
        rdir = None
        bdir = None
        if "G" in arg:
            gdir = arg
        if "R" in arg:
            rdir = arg
        if "B" in arg:
            bdir = arg
        
        if gdir != None:
            if arg == "G1":
                gametype = "G1"
            gametype = "G0"
        
        if rdir != None:
            rwdcase, strval, rwdval = distinguish_string(rdir)
            if rwdcase == "R:#": #Sets the implied reward to the number indicated (default is 12)
                dRwd = int(strval)
            
            if rwdcase == "R#": #Sets the reward at the cell indicated by number equal to the implied reward
                vprops[int(strval)] = {"rwd":dRwd}
            if rwdcase == "R#:#": #Sets the reward for the cell at the first # to be equal to the 2nd #
                vprops[int(strval)] = {"rwd":int(rwdval)}
        if bdir != None:
            bcase, vertex, dirs = distinguish_bstring(bdir)
            if bcase == "B#":
                nbrsStr = []
                eprops = {}                
                nbrs = remove_sets_with_number(nbrs, vertex)
                for v in range(0, size):

                    temp_nbrs = set()
                    if vertex - 1 >= 0 and (vertex - 1)//width == vertex//width:
                        temp_nbrs.add(vertex - 1)
                    if vertex + 1 < size and (vertex + 1)//width == vertex//width:
                        temp_nbrs.add(vertex + 1)
                    if vertex + width < size and (vertex + width)%width == vertex%width:
                        temp_nbrs.add(vertex + width)
                    if vertex - width >= 0 and (vertex - width)%width == vertex%width:
                        temp_nbrs.add(vertex - width)




                    for n in nbrs:
                        eprops[str((v, n))] = {"rwd": 0}
                    if temp_nbrs in nbrs:
                        print(temp_nbrs)
                        nbrsStr.append(getChar(v, width, temp_nbrs))
                
            if bcase == "B#[NSEW]+":
                bing = 1
            

    


    graph = {"props":{"rwd":dRwd, "width": width}, "nbrs":nbrs, "nbrsStr":nbrsStr, "size":size, "vProps":vprops, "eProps": eprops, "jumps": jumps, "game": gametype}
    return graph

def remove_sets_with_number(sets_list, number):
    # Use list comprehension to create a new list that only includes sets that do not contain the given number
    new_sets_list = [s for s in sets_list if number not in s]
    return new_sets_list


def distinguish_bstring(s):
    # Regular expression pattern for strings of the form B# and B#[NSEW]+
    pattern = r'^B(\d+)([NSEW]*)$'
    
    # Search for the pattern in the input string
    match = re.search(pattern, s)
    
    if match:
        # If the string matches the pattern, extract the number and the letters
        number = int(match.group(1))
        letters = match.group(2)
        
        # Determine the type of the string
        if letters:
            string_type = 'B#[NSEW]+'
        else:
            string_type = 'B#'
        
        return string_type, number, letters
    
    else:
        # If the string does not match the pattern, return None
        return None

def distinguish_string(s):
    # Define the regex patterns
    pattern1 = r"^R:(\d+)$"
    pattern2 = r"^R(\d+)$"
    pattern3 = r"^R(\d+):(\d+)$"
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

def getWidth(size):
    begin = math.ceil(math.sqrt(size))
    # Iterate from begin to size (inclusive)
    for x in range(begin, size + 1):
        # If size is divisible by i, return i
        if size % x == 0:
            return x
        
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
    return neighbors_string

def main():
    graph = grfParse(['4', 'G0', 'R0', 'B1'])

    # graph = grfParse(args)
    print(graph)
     
    # policy_str = calculate_policy(graph)
    # showPolicy(policy, graph)
    # size = grfSize(graph)
    # print(grfNbrs(graph, 0))
    # print(grfVProps(graph, 0))
    edgesStr = grfStrEdges(graph) 
    # propsStr = grfStrProps(graph)
    # # output edgesStr
    # # print(grfNbrs(graph, 7))
    # # print(graph["nbrs"])
    # # print(edgesStr)
    # print(propsStr)
if __name__ == '__main__':main()

#Om Gole, 6, 2024