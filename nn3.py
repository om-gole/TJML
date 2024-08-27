import sys; args = sys.argv[1:]
import math, random, time

#python nn3.py x*x+y*y<=1.42265634705309

#timetest 0 last time
t = time.time()

#PRELIMINARY CALCULATIONS/INIT

inputs = []
outputs = []


inequality = args[0][7:9] if '=' in args[0] else args[0][7]
# print(inequality)
radius = float(args[0][9:]) if '=' in args[0] else float(args[0][8:])

layerAMT = [3,6,3,2,1,1] 
struct = [[0 for i in range(layerAMT[x])] for x in range(len(layerAMT))]

#print(struct)
#Initialize Weights 
weightsList = []
for i in range(1,len(struct)-1):
        weightsList.append([random.random() for __ in range(len(struct[i])*len(struct[i-1]))])
weightsList.append([random.random() for i in range(len(struct[-1]))])

numLayers = len(struct)
alpha = 0.1

def initialize():
    for __ in range(1000):
        temp1 = calcRandom(3)
        temp2 = calcRandom(3)
        answer = 0#equation(temp2,temp1)
        if inequality[0] == '>':
            if temp2*temp2+temp1*temp1 > radius: 
                answer = 1
            else: 
                answer = 0
        else:
            if temp2*temp2+temp1*temp1 < radius: 
                answer = 1
            else: 
                answer = 0
        
        #OUTPUTS
        outputs.append([answer])

        #INPUTS
        inputs.append([temp2,temp1,1.0])

def calcRandom(x):
     return random.random()*x - 1.5

def make_gradient(errorList, structure):
    gradientList = []
    for layer in range(len(structure)-2):
        gradient = []
        for error in range(len(errorList[layer])):
            for prev in range(len(structure[layer])):
                gradient.append(errorList[layer][error] * structure[layer][prev])
        gradientList.append(gradient)

    # Last Layer
    gradient = []
    for error in range(len(errorList[-1])):
        gradient.append(structure[-2][error] * errorList[-1][error])
    gradientList.append(gradient)

    return gradientList

def d(x):
    return x*(1-x)

def calculate_errors(structure, weightsList, outputIdx):
    errors = []
    last_layer_error = [outputs[outputIdx][i] - structure[-1][i] for i in range(len(outputs[outputIdx]))]
    errors.append(last_layer_error)
 
    second_last_layer_error = [errors[0][i] * d(structure[-2][i]) * weightsList[-1][i] for i in range(len(structure[-2]))]
    errors.insert(0, second_last_layer_error)

    for idxLyr in range(len(structure[0:-3]), 0, -1):
        current_layer_error = []
        for i in range(len(structure[idxLyr])):
            loop_step = len(structure[idxLyr])
            sum_of_errors = sum(errors[0][iter_step] * weightsList[idxLyr][wIdx] for iter_step, wIdx in enumerate(range(i, len(weightsList[idxLyr]), loop_step)))
            current_layer_error.append(sum_of_errors * d(structure[idxLyr][i]))
        errors.insert(0, current_layer_error)

    return errors

#1.42265634705309

def calculate_output(struct):
    output_layer = struct[-1]
    second_last_layer = struct[-2]
    last_weights = weightsList[-1]
    
    for i in range(len(output_layer)):
        output_layer[i] = second_last_layer[i] * last_weights[i]

def bp(network, weight, output):
    
    error = calculate_errors(network, weight,output)
    #print(error)
    # SPAM PRINT print(f'Error: {error}')
    gradient = make_gradient(error, network)
    # SPAM PPRINT print(f'Gradient: {gradient}')
    for x in range(len(weight)):
        for y in range(len(weight[x])):
            weight[x][y] = weight[x][y] + (gradient[x][y] * alpha)

def transfer(x): #T3 from nn2
    return 1/(1+math.exp(-x)) 

def adjustW():
    adjacent()

def adjacent():
    global alpha
    for x in range(800):
        for i in range(len(inputs)):
            adjust(i)            
        alpha *= 0.995

def adjust(x):
    temp = struct.copy()
    temp[0] = inputs[x]
    #ff then bp
    temp = ff(temp)
    bp(temp, weightsList,x)



def ff(struct):
    # from nn1
    for layer_index in range(len(weightsList) - 1):
        layer_weights = weightsList[layer_index]
        node_count = len(layer_weights) // len(struct[layer_index])
        step_size = len(struct[layer_index])
        start_index = -1 * step_size
        for node_index in range(node_count):
            start_index += step_size
            node_value = 0
            for weight_index in range(start_index, start_index + step_size):
                node_value += struct[layer_index][weight_index % step_size] * layer_weights[weight_index]
            node_value = transfer(node_value)
            struct[layer_index + 1][node_index] = node_value
    calculate_output(struct)

    return struct

def main():
    initialize()

    print(f'Layer counts: {" ".join(str(len(s)) for s in struct).rstrip()}')
    
    adjustW() #change weights

    # print('Weights!')
    for w in weightsList:
        print(w)

    print(f'Neural Network: {struct}')

    #t2 = time.time()
    #print(f'{round(t2-t)} seconds to run the program')
    #print(struct)

#x*x+y*y>=1.3972055888292676

if __name__ == '__main__':
    main()
#Om Gole, 6, 2024