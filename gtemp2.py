import sys; args = sys.argv[1:]
def grfParse(lstArgs): 
   #returns graph constructed from lstArgs
   adjacents = dict()
   graphType = 'G'
   width = None
   reward = 12
   size = 0
   vslc = []
   all_v_list = []
   for direct in lstArgs:
      if direct[0] == 'G':
         for ind, arg in enumerate(direct):
            if arg == 'N':
               graphType = 'N'
            elif arg == 'W':
               temp_ind = ind + 1
               while temp_ind < len(direct) and direct[temp_ind].isdigit():
                  temp_ind += 1
               width = int(direct[ind + 1: temp_ind])
            elif direct[ind - 1] == 'R':
               temp_ind = ind
               while temp_ind < len(direct) and direct[temp_ind].isdigit():
                  temp_ind += 1
               reward = int(direct[ind: temp_ind])
            elif ind > 0 and direct[ind - 1] in 'GN' and direct[ind].isdigit():
               temp_ind = ind
               while temp_ind < len(direct) and direct[temp_ind].isdigit():
                  temp_ind += 1
               size = int(direct[ind: temp_ind])
               
         if width is None:
            width = int((size) ** 0.5)
            if not (size) ** 0.5 == width:
               width += 1
            while size % width != 0:
               width += 1
         if width == 0:
            graphType = 'N2'
            
      elif direct[0] == 'V':
         all_v = set()
         v_reward = None
         blocking = False
         index = 1
         while index < len(direct) and (direct[index].isdigit() or direct[index] in ['-',':',',']):
            index += 1
         vslc = direct[1:index]
         vslc = vslc.split(',')
         for ind, vlists in enumerate(vslc):
            if ':' in vlists:
               x = list(range(size))
               sli = vlists.split(":")
               if len(sli) > 2:
                  step = 1
                  if sli[2]:
                     step = int(sli[2])
                  if sli[0] == '' and sli[1] != '':
                     all_v = all_v.union(set(x[ : int(sli[1]) : step]))
                  elif sli[0] != '' and sli[1] == '':
                     all_v = all_v.union(set(x[ int(sli[0]) :: step]))
                  elif sli[0] != '' and sli[1] != '':
                     all_v = all_v.union(set(x[ int(sli[0]) : int(sli[1]): step]))
                  else:
                     all_v = all_v.union(x[::step])
               else:
                  if sli[0] == '' and sli[1] != '':
                     all_v = all_v.union(set(x[ : int(sli[1]) :]))
                  elif sli[0] != '' and sli[1] == '':
                     all_v = all_v.union(set(x[ int(sli[0]) :: ]))
                  elif sli[0] != '' and sli[1] != '':
                     all_v = all_v.union(set(x[ int(sli[0]) : int(sli[1]):]))
                  else:
                     all_v = all_v.union(x[::])
            else:
               ord_list = list(range(size))
               all_v.add(ord_list[int(vlists)])
            
            print(all_v)
         while index < len(direct):
            if direct[index] == 'R':
               temp_ind = index + 1
               while temp_ind < len(direct) and direct[temp_ind].isdigit():
                  temp_ind += 1
               if temp_ind != index + 1:
                  v_reward = int(direct[index + 1: temp_ind])
               else:
                  v_reward = reward
            if direct[index] == 'B':
               blocking = True
            index += 1
         all_v_list.append(['V', all_v, v_reward, blocking, '_', False, '_'])
      
      elif direct[0] == 'E':
         mngmnt = None
         preset = False
         if direct[1] in '!+*~@':
            mngmnt = direct[1] 
         e_reward = None
         blocking = False
         start_index = 1
         if mngmnt:
            start_index = 2
         index = start_index
         all_v1, index = add_verts(direct, start_index, index, size)
         if direct[index] in '=~':
            direction = direct[index]
            index += 1
            start_index = index
            while index < len(direct) and (direct[index].isdigit() or direct[index] in ['-',':',',']):
               index += 1
            all_v2, index = add_verts(direct, start_index, index, size)
            if not mngmnt:
               mngmnt = '~'
            preset = True
         else:
            start_index = index
            while direct[index] in 'ENSW':
               index += 1
            dir_list = direct[start_index : index]
            direction = direct[index]
            check_valids = {'N' : -1 * width, 'S' : width, 'E' : 1, 'W' : -1}
            all_v2 = []
            for val in all_v1:
               tups = []
               for news in dir_list:
                  if news == 'N':
                     if val - width >= 0:
                        tups.append(val - width)
                     else:
                        tups.append('_')
                  elif news == 'E':
                     if (val + 1) % width != 0:
                        tups.append(val + 1)
                     else:
                        tups.append('_')
                  elif news == 'W':
                     if (val - 1) % width != width - 1:
                        tups.append(val - 1)
                     else:
                        tups.append('_')
                  elif news == 'S':
                     if val + width < size:
                        tups.append(val + width)
                     else:
                        tups.append('_')
               all_v2.append(tups)
         while index < len(direct):
            if direct[index] == 'R':
               temp_ind = index + 1
               while temp_ind < len(direct) and direct[temp_ind].isdigit():
                  temp_ind += 1
               if temp_ind != index + 1:
                  e_reward = int(direct[index + 1: temp_ind])
               else:
                  e_reward = reward
            index += 1
         all_v_list.append(['E', mngmnt, all_v1, all_v2, direction, preset, e_reward])
   # print(all_v_list)    
   for i in range(size):
      adjacents[i] = [set(), None]
   if width != 0:  
      for i in range(size):
         if i % width != width - 1: 
            adjacents[i][0].add(i + 1)
            adjacents[i + 1][0].add(i)
            
         if i + width < size: 
            adjacents[i][0].add(i + width)
            adjacents[i + width][0].add(i)
   if graphType == 'G':
      edges = {i : [{val for val in adjacents[i][0]}, adjacents[i][1]] for i in adjacents} 
   else:
      edges = dict()
      for i in range(size):
         edges[i] = [set(), None]
   copy_edges = {i : [{val for val in edges[i][0]}, edges[i][1]] for i in edges}
   # print(copy_edges, '\n')
   edge_rewards = dict()
   for option, vert_set, rew, block, direction, regs, e_rew in all_v_list:
      if option == 'V':
         if rew != None:
            for vert in vert_set:
               copy_edges[vert][1] = rew
         if block:
            for vert in vert_set:
               adj_nbrs = {i for i in adjacents[vert][0] if i not in vert_set}
               nbrs = edges[vert][0].union(adj_nbrs)
               for nbr in nbrs:
                  if nbr not in vert_set:
                     if nbr in copy_edges[vert][0]:
                        copy_edges[vert][0].remove(nbr)
                        if (vert, nbr) in edge_rewards:
                           edge_rewards[(vert, nbr)] = None
                     else:
                        copy_edges[vert][0].add(nbr)
                        
                     if vert in copy_edges[nbr][0]:
                        copy_edges[nbr][0].remove(vert)
                        if (nbr, vert) in edge_rewards:
                           edge_rewards[(nbr, vert)] = None
                     else:
                        if nbr in adjacents[vert][0]:
                           copy_edges[nbr][0].add(vert)
               for v in [value for value in edges if value not in nbrs]:
                  if v not in vert_set and vert in copy_edges[v][0]:
                     copy_edges[v][0].remove(vert)
                     
                  # if nbr in copy_edges[vert][0]:
                  #    if nbr not in vert_set:
                  #       copy_edges[vert][0].remove(nbr)
                  #       copy_edges[nbr][0].remove(vert)
                  # else:
                  #    copy_edges[vert][0].add(nbr)
                  #    copy_edges[nbr][0].add(vert)
               edges = {i : [{val for val in copy_edges[i][0]}, copy_edges[i][1]] for i in copy_edges}
      elif option == 'E':
         edge_set = set()
         mngmnt, all_v1, all_v2 = vert_set, rew, block
         if not regs:
            new_all_v1 = []
            new_all_v2 = []
            for indicie, val in enumerate(all_v1):
               others = all_v2[indicie]
               for o_e in others:
                  if o_e != '_':
                     new_all_v1.append(val)
                     new_all_v2.append(o_e)
            all_v1 = new_all_v1
            all_v2 = new_all_v2
         if mngmnt == None:
            mngmnt = '~'
         copy_edges, edges = edge_case(mngmnt, edges, copy_edges, all_v1, all_v2, edge_set, edge_rewards)
         if direction == '=':
            copy_edges, edges  = edge_case(mngmnt, edges, copy_edges, all_v2, all_v1, edge_set, edge_rewards)
         for edge in [(x_val, y_val) for (x_val, y_val) in zip(all_v1, all_v2)]:
            if edge not in edge_rewards:
               edge_rewards[edge] = None
            if edge_rewards[edge] == True:
               edge_rewards[edge] = e_rew
         if direction == '=':
            for edge in [(x_val, y_val) for (x_val, y_val) in zip(all_v2, all_v1)]:
               if edge not in edge_rewards:
                  edge_rewards[edge] = None
               if edge_rewards[edge] == True:
                  edge_rewards[edge] = e_rew
      # print(copy_edges, '\n')
   return (copy_edges, reward, width, graphType, edge_rewards)

def edge_case(mngmnt, edges, copy_edges, all_v1, all_v2, edge_set, edge_rewards):
   for v_ind, vNum in enumerate(all_v1):
      if (vNum, all_v2[v_ind]) not in edge_set:
         if mngmnt == '~':
            if all_v2[v_ind] in edges[vNum][0]:
               copy_edges[vNum][0].remove(all_v2[v_ind])
               edge_rewards[(vNum, all_v2[v_ind])] = False
            else:
               copy_edges[vNum][0].add(all_v2[v_ind])
               edge_rewards[(vNum, all_v2[v_ind])] = True
         elif mngmnt == '!':
            if all_v2[v_ind] in edges[vNum][0]:
               copy_edges[vNum][0].remove(all_v2[v_ind])
               edge_rewards[(vNum, all_v2[v_ind])] = False
         elif mngmnt == '+':
            if all_v2[v_ind] not in edges[vNum][0]:
               copy_edges[vNum][0].add(all_v2[v_ind])
               edge_rewards[(vNum, all_v2[v_ind])] = True
            else:
               edge_rewards[(vNum, all_v2[v_ind])] = False
         elif mngmnt == '@':
            if all_v2[v_ind] in edges[vNum][0]:
              edge_rewards[(vNum, all_v2[v_ind])] = True
         elif mngmnt == '*':
            copy_edges[vNum][0].add(all_v2[v_ind])
            edge_rewards[(vNum, all_v2[v_ind])] = True
         edges = {i : [{val for val in copy_edges[i][0]}, copy_edges[i][1]] for i in copy_edges}
         edge_set.add((vNum, all_v2[v_ind]))
      else:
         edge_set.add((vNum, all_v2[v_ind]))
   return copy_edges, edges
   
def add_verts(direct, start_index, index, size):
   all_v = []
   while index < len(direct) and (direct[index].isdigit() or direct[index] in ['-',':',',']):
      index += 1
   vslc = direct[start_index:index]
   vslc = vslc.split(',')
   for ind, vlists in enumerate(vslc):
      if ':' in vlists:
         x = list(range(size))
         sli = vlists.split(":")
         if len(sli) > 2:
            step = 1
            if sli[2]:
               step = int(sli[2])
            if sli[0] == '' and sli[1] != '':
               all_v = all_v + x[ : int(sli[1]) : step]
            elif sli[0] != '' and sli[1] == '':
               all_v = all_v + x[ int(sli[0]) :: step]
            elif sli[0] != '' and sli[1] != '':
               all_v = all_v + x[ int(sli[0]) : int(sli[1]): step]
            else:
               all_v = all_v + x[::step]
         else:
            if sli[0] == '' and sli[1] != '':
               all_v = all_v + x[ : int(sli[1]) :]
            elif sli[0] != '' and sli[1] == '':
               all_v = all_v + x[ int(sli[0]) :: ]
            elif sli[0] != '' and sli[1] != '':
               all_v = all_v + x[ int(sli[0]) : int(sli[1]):]
            else:
               all_v = all_v + x[::]
      else:
         ord_list = list(range(size))
         all_v = all_v + [ord_list[int(vlists)]]
   return (all_v, index)

def grfSize(graph): 
   #returns the number of vertices the graph has
   return len(graph[0])

def grfNbrs(graph, v): 
   #returns a set or list of the neighbors (ie. ints) of vertex v
   if graph[0]:
      return graph[0][v][0]

def grfGProps(graph): 
   #returns the dictionary of properties for the graph including width and rwd
   props = dict
   if graph[3] == 'G' or graph[3] == 'N2':
      props = {'rwd': graph[1], 'width': graph[2]}
   else:
      props = {'rwd': graph[1]}
   return props

def grfVProps(graph, v): 
   #returns the dictionary of properties of vertex v
   if graph[0][v][1] != None:
      return {'rwd': graph[0][v][1]}
   return {}

def grfEProps(graph, v1, v2): 
   #returns the dictionary of properties of edge (v1, v2)
   if (v1, v2) in graph[4] and graph[4][(v1, v2)] and graph[4][(v1, v2)] != False:
      return {'rwd': graph[4][(v1, v2)]}
   return {}

def grfStrEdges(graph): 
   #returns a string representation of the graph edges
   direction = {graph[2]: 'S', -1: 'W', 1: 'E', -1 * graph[2]: 'N'}
   reps = {'E': 'E', 'N': 'N', 'S': 'S', 'W': 'W',
            'EN': 'L', 'ES': 'r', 'NW': 'J', 'SW': '7', 'EW': '-', 'NS': '|',
            'ENS': '>', 'ENW': '^', 'ESW': 'v', 'NSW': '<',
            'ENSW': '+', '.': '.'}
   total_rep = ''
   jumps = set()
   for vert in graph[0]:
      string_rep = ''
      if graph[0][vert][0]:
         for nbr in graph[0][vert][0]:
            num = nbr - vert
            if num == -1 and (vert % graph[2] != 0):
               string_rep += direction[num]
            elif num == 1 and (nbr % graph[2] != 0):
               string_rep += direction[num]
            elif abs(num) == graph[2]:
               string_rep += direction[num]
            else:
               jumps.add((vert, nbr))
         string_rep = ''.join(sorted(string_rep))
         if not string_rep:
            string_rep = '.'
      else:
         string_rep = '.'
      total_rep += reps[string_rep]
      # if graph[2] > 0 and (vert + 1) % graph[2] == 0:
      #    total_rep += '\n'
   if jumps:
      jum_str = ''
      jum_set = set()
      for one, two in jumps:
         if (one, two) not in jum_set:
            if (two, one) in jumps:
               jum_set.add((one, two))
               jum_set.add((two, one))
               jum_str += f'{one}={two};'
            else:
               jum_set.add((one, two))
               jum_str += f'{one}~{two};'
      if jum_str:
         total_rep = f'{total_rep}\n{jum_str[:-1]}'  
      else:
         total_rep = f'{total_rep}\n{jum_str}' 
   if set(total_rep) == {'.'}:
      return ''
   # print(total_rep)
   return total_rep

def grfStrProps(graph): 
   var = grfGProps(graph)
   if graph[3] == 'G' or graph[3] == 'N2':
      var =  f"rwd: {var['rwd']}, width: {var['width']}"
   else:
      var = f"rwd: {var['rwd']}"
   for vert in graph[0]:
      if graph[0][vert][1] != None:
         var += '\n' + f'{vert}: rwd: {graph[0][vert][1]}'
   for edge in graph[4]:
      if graph[4][edge] and graph[4][edge] != False:
         var += '\n' + f'{edge}: rwd: {graph[4][edge]}'
   return var

def main():
   graph = grfParse(args)
   # graph = grfParse(['GN12W6R25', 'V0R'])
   edgesStr = grfStrEdges(graph)
   propsStr = grfStrProps(graph)
   # print(grfVProps(graph, 0))
   print(edgesStr)
   print('\n')
   print(propsStr, '\n')
   # print(grfNbrs(graph, 1))
   twoDstring = edgesStr.split('\n')
   output = ''
   if twoDstring[0]:
      output += '\n'.join([twoDstring[0][i : i + graph[2]] for i in range(0, len(twoDstring[0]), graph[2])])
      if len(twoDstring) > 1:
         output += f'\nJumps: {twoDstring[1]}'
      output += '\n' + propsStr
      # for vert in graph[0]:
      #    if graph[0][vert][1] != None:
      #       output += '\n' + f'{vert}: rwd: {graph[0][vert][1]}'
      # for edge in graph[4]:
      #    if graph[4][edge] and graph[4][edge] != False:
      #       output += '\n' + f'{edge}: rwd: {graph[4][edge]}'
      print(output)
   else:
      output = propsStr
      print(output)
   # print(args)
   # graph = grfParse(args)
   # print(graph)
   # print(grfSize(graph))
   # print(grfGProps(graph))
   # print(grfVProps(graph, 4))
   # print(grfNbrs(graph, 1))
   # print(grfEProps(graph, 0, 26))
   # print(grfStrEdges(graph))
   # print(grfStrProps(graph))
   
if __name__ == '__main__': main()
#Om Gole, 6, 2024