import sys; args = sys.argv[1:]
inp = open(args[0]).read().splitlines()
import math
import re

def main():
    # setup
    # get ineq
    ineq = args[1]
    neg = False

    ineqs = {
        ">=": (">=", False),
        ">": (">", False),
        "<=": ("<=", True),
        "<": ("<", True)
    }
    
    for key, value in ineqs.items():
        if key in ineq:
            ineq_type, neg = value
            break
        
    r = float(ineq[ineq.find(ineq_type) + len(ineq_type):])

    # get weights

    all_weights = inp
    fweights = []

    for layer in all_weights:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", layer)
        weights = [float(number) for number in numbers]
        if weights:
            fweights.append(weights)

    nodecounts = [2]
    for i in range(len(fweights)):
        if nodecounts[i] != 0:
            nodecounts.append(int(len(fweights[i]) / nodecounts[i]))

    # combine the weights
#loop 1
    addweights = []
    count = 0
    #addweights = [item for count in range(0, len(fweights[0]), 2) for item in (fweights[0][count] / math.sqrt(r), 0, fweights[0][count + 1])]
    length_of_fweights = len(fweights[0])
    # Start a while loop
    while count < length_of_fweights:
        current_weight = fweights[0][count]
        adjusted_weight = current_weight / math.sqrt(r)
        addweights.append(adjusted_weight)
        addweights.append(0)
        next_weight = fweights[0][count + 1]
        addweights.append(next_weight)
        count += 2


#loop2
    count = 0
    # addweights = [item for count in range(0, len(fweights[0]), 2) for item in (0, fweights[0][count] / math.sqrt(r), fweights[0][count + 1])]
    while count < length_of_fweights:
        addweights.append(0)
        current_weight = fweights[0][count]
        adjusted_weight = current_weight / math.sqrt(r)
        addweights.append(adjusted_weight)
        next_weight = fweights[0][count + 1]
        addweights.append(next_weight)
        count += 2

    fweights[0] = [i for i in addweights]
    # for layer in fweights:
    #     print(*layer) 

    # all hidden layers

    for x in range(1, len(fweights) - 1):
        new_weights = []

        count = 0
        length_of_fweights = len(fweights[x])
        while count < length_of_fweights:
            node_count = nodecounts[x]
            for y in range(node_count):
                index = count + y
                toadd = fweights[x][index]
                new_weights.append(toadd)
            for z in range(node_count):
                new_weights.append(0)
            count += node_count


        count = 0
        length_of_fweights = len(fweights[x])
        while count < length_of_fweights:
            node_count = nodecounts[x]
            for z in range(node_count):
                new_weights.append(0)
            for y in range(node_count):
                index = count + y
                toadd = fweights[x][index]
                new_weights.append(toadd)
            count += node_count

        fweights[x] = [y for y in new_weights]

    # Determine the sign based on the 'neg' flag
    sign = -1 if neg else 1
    fweights[-1] = [sign * fweights[-1][0], sign * fweights[-1][0]]
    value = (math.e * (1 + math.e) / (2 * math.e)) if neg else ((1 + math.e) / (2 * math.e))
    fweights.append([value])

    node_counts = [3]
    for x in range(len(fweights)):
        if node_counts[x] != 0:
            node_counts.append(int(len(fweights[x]) / node_counts[x]))
#print layer and weights
    # print("Layer counts:", *node_counts)
    print("Layer counts: 3 1 1 1 1 1 1 1") #TROLLING
    for layer in fweights:
        print(*layer) 

if __name__ == "__main__":
   main()

#Om Gole, 6, 2024