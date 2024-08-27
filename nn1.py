import sys; args = sys.argv[1:]
file = open(args[0])
import math
# sample input: weights.txt T3 5 2 3 1 4
# sample output: 

def transfer(t_funct, input_val):
   temp = sum(input_val)
   if t_funct == 'T1':
      return temp
   if t_funct == 'T2' and temp <= 0:
      return 0
   if t_funct == 'T2':
      return temp
   if t_funct == 'T3':
      return 1 / (1 + math.e ** -temp)
   if t_funct == 'T4':
      return -1 + 2/(1+math.e**-temp)

def dot_product(input_vals, weights, layer):
   return [[a * b for a, b in zip(input_vals, weights[layer][cell_num])] for cell_num in range(len(weights[layer]))] 

def evaluate(file, input_vals, t_funct):
   raw_data = [[float(weight) for weight in layer.split()] for layer in file.read().split('\n')]
   data = [line for line in raw_data if line]

   network = []

   for index, line in enumerate(data):
      num_cells = len(input_vals) if index == 0 else len(network[index - 1])
      cell_weights = [line[cell * num_cells : (cell + 1) * num_cells] for cell in range(len(line) // num_cells)]
      network.append(cell_weights)
   
   level = 0
   while (level < len(network) - 1):
      weighted_input = dot_product(input_vals, network, level)
      input_vals = [transfer(t_funct, input_val) for input_val in weighted_input]
      level += 1

   toRet = [network[level][0][i] * input_vals[i] for i in range(len(network[level][0]))]

   return toRet
     
def main():
   inputs, t_funct, transfer_found = [], 'T1', False
   for arg in args[1:]:
      if not transfer_found:
         t_funct, transfer_found = arg, True
      else:
         inputs.append(float(arg))
   li =(evaluate(file, inputs, t_funct)) #ff
   for x in li:
      print (x, end=' ') # final outputs
      
if __name__ == '__main__': main()

# Om Gole, Period 6, 2024