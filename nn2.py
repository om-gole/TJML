import sys; args = sys.argv[1:]
infile = open(args[0])
import time, random, math


# Calculating BP 

def readFile(): # Read File
    inputs = []
    outputs = []
    readfile = infile
    for line in readfile: 
        equal = line.index("=")
        inputs.append([int(num) for num in line[:equal-1].split(" ")] + [1])
        outputs.append(int(line[equal+3]))
    return inputs, outputs



def setWeight(w, inputs):
   w_pos_to_arrow = {}
   prev_size = len(inputs)
   for layer in range(len(w)):
      for node in range(len(w[layer])):
         w_pos_to_arrow[(layer,node)] = [(layer,node%prev_size),(layer+1,node//prev_size)] 
      prev_size = len(w[layer])//prev_size # Update Prev Size
   return w_pos_to_arrow
# USEFUL, dot, transffer, error
def dot(x, y): 
    return sum([x[i]*y[i] for i in range(len(x))])

def transfer(x): # T3
    return 1/(1+math.exp(-x))

def Error(t, y):
    return (1/2)*((t-y)**2)

def ff(weightsLayers, inputLayer): # From nn1
    networkCopy = []
    outputWeight = weightsLayers[-1]
    currentLayer = inputLayer
    for idx in range(len(weightsLayers[:-1])):
        networkCopy.append(currentLayer)
        weights = weightsLayers[idx]
        nextLayer = []
        while weights:
            nextLayer.append(transfer(dot(currentLayer, weights)))
            weights = weights[len(currentLayer):]
        currentLayer = nextLayer 
    networkCopy.append(currentLayer)
    output = currentLayer[0]*outputWeight[0]
    networkCopy.append([output])
    return output, networkCopy

def partialNet(w, expected_output, net_copy):
   partial_net = [[], [], [], [expected_output-net_copy[-1][0]]] 
   for layer in range(len(net_copy)-2,0,-1): 
      for node in range(len(net_copy[layer])): 
         activation = net_copy[layer][node]*(1-net_copy[layer][node]) 
         partial_net[layer].append(bp((layer,node), w[layer], partial_net[layer+1])*activation)
   return partial_net

def bp(newC, weightLayer, computedLayer):
    dotWeights, dotCells = [], []
    for y in range(len(weightLayer)):
        left, right = weight_arrow[newC[0], y] 
        if left != newC: continue
        dotWeights.append(weightLayer[y])
        dotCells.append(computedLayer[right[1]]) 
    return dot(dotCells, dotWeights)

def adjustW(w, net_copy, partial_net):
   for layer in range(len(w)):
      for node in range(len(w[layer])): 
         source, target = weight_arrow[(layer,node)]
         if not layer or layer == len(w)-1: delta = alpha*net_copy[source[0]][source[1]]*partial_net[target[0]][target[1]]
         else: delta = alpha*(partial_net[source[0]][source[1]]+partial_net[target[0]][target[1]])
         w[layer][node] = w[layer][node] + delta
   return w

def printFinal(weights):
    print("Layer counts: {} 2 1 1".format(len(inputsLst[0])) + '\n')
    weightArray = []
    for weightLayer in weights:
        weightArray.append(" ".join([str(num) for num in weightLayer]))
    print("{}\n{}\n{}".format(weightArray[0], weightArray[1], weightArray[2]) + '\n') 
    print('These are your weights' + '\n')

#Make a training set of around the points of the circle.

start = time.time()
alpha = 1.99 
inputsLst, expOutputLst = readFile() # List of Lists, List of Ints
weights = [[random.random() for i in range(2*len(inputsLst[0]))], [random.random(), random.random()], [random.random()]]
#weights = [[round(random.uniform(-2.0, 2.0), 2) for j in range(layer_count[i]*layer_count[i+1])]  for i in range(len(layer_count)-1)]
weight_arrow = setWeight(weights, inputsLst[0])
#Originally 3, keep changing
#use time to keep going
iterations = 0
while(time.time()-start < 28):
    # Feed Forward through Network
    inputs, output = inputsLst[iterations%len(inputsLst)], expOutputLst[iterations%len(inputsLst)]
    tempO, tempNC = ff(weights, inputs) # Fix Choosing Input
    tempError = Error(output, tempO)
    # patrials & weights
    tempPartialNet = partialNet(weights, output, tempNC)
    weights = adjustW(weights, tempNC, tempPartialNet)
    #add 1 interation
    iterations += 1
    
# Print Results
printFinal(weights)

#Om Gole, 6, 2024