import random
# Talk to the grader to get a small example with threads. What's the minimum you can say to the grader to get this?

#For the example, determine the size (# of vertices), visit(tLimit, strandCtLimit). Port those and especially the strands as a triple quoted string into your main(). and convert the stands as a 
#triple quoted string into your main() and convert the strands as a string to a list of lists of tuple. Print these. Non-string strands to show you've done it.

def monteCarlo(sz, actionReport):
    global size, strands, policy
    if sz>0: 
        size = sz
        policy = [set() for v in range(sz)]
        visitCtLimit = actionReport[0][1]
        strandCtLimit = actionReport[1][1]
        
        augmentCt = -1
        dist = 0
        while augmentCt:
            augmentCt = 0; dist -=1
        
        return []


    
    if len(actionReport) > 1 and actionReport[-1][1]>0:
        policy[actionReport[-2][0]].add(actionReport[-1][0])
    if sz<0: return []
    if sz == 0: return policy

def main():
    # Hard code the test case
    vertexCt = 10
    visitCtLim = 450
    strandCtLim = 18
    strands = """
    Strand 1: 2 R20
    Strand 2: 5 0 6 7 3 3 5 6 5 0 5 0 5 0 2 R20
    Strand 3: 9 2 R20
    Strand 4: 7 5 6 7 6 1 2 R20
    Strand 5: 8 1 2 R20
    Strand 6: 9 8 6 1 6 5 6 1 3 7 3 8 3 9 4 2 R20
    Strand 7: 6 5 6 5 0 5 0 2 R20
    Strand 8: 7 8 6 5 0 6 7 8 4 0 5 6 7 2 R20
    Strand 9: 6 5 0 6 7 8 9 4 0 2 R20
    Strand 10: 4 9 8 1 0 2 R20
    Strand 11: 1 4 9 4 2 R20
    Strand 12: 7 5 0 2 R20
    Strand 13: 1 4 7 5 6 1 2 R20
    Strand 14: 5 6 7 3 2 R20
    Strand 15: 8 6 7 3 8 1 2 R20
    Strand 16: 6 1 0 2 R20
    Strand 17: 6 1 0 6 7 3 2 R20
    Strand 18: 4 2 R20
    """.strip().splitlines()

    assembled_strands = []
    for strand in strands:
        parts = strand.split()[2:]  # Split the strand and ignore the initial 'Strand X:'
        assembled_strand = []
        for i in range(len(parts)):
            if 'R' in parts[i]:
                # If 'R' is found, create a tuple with the previous number and the reward value
                reward = int(parts[i][1:])
                assembled_strand.append((int(parts[i-1]), reward))
                break
            elif i < len(parts) - 1 and 'R' not in parts[i+1]:
                # Pair the current number with the next number if it's not a reward string
                assembled_strand.append((int(parts[i]), int(parts[i+1])))
        print(assembled_strand)
        assembled_strands.append(assembled_strand)
    # Call monteCarlo function
    
    initRes = monteCarlo(vertexCt, [(-4,visitCtLim), (-5,strandCtLim)])
    print(initRes)
    for i, strnd in enumerate(assembled_strands):
        res = monteCarlo(-len(assembled_strands)+i, strnd)

    print(f"Policy {res=}")  # res stands for result
    
    
if __name__ == "__main__":
    main()

#Om Gole, 6, 2024