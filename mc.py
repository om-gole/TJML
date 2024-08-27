import random

def monteCarlo(sz, lstStrand):
    print(lstStrand)
    if sz > 0:
        # Set the maximum length of a Monte Carlo strand and the maximum number of times a vertex will be visited
        strandLimitLen = 100
        sameVertexLimit = 10
        return [(-2, strandLimitLen), (-3, sameVertexLimit)]
    elif sz < 0:
        # Calculate probabilities for the Monte Carlo simulator
        probabilities = []
        for i in range(abs(sz)):
            if i < len(lstStrand):  # Check if i is within the range of lstStrand
                vtxFrom = lstStrand[i][0]
                vtxTo = lstStrand[i+1][0] if i+1 < len(lstStrand) else lstStrand[i][0]
                p = random.uniform(0, 1)
                probabilities.append((p, (vtxFrom, vtxTo)))
        return probabilities
    else:
        # Return a policy
        policy = []
        for i in range(len(lstStrand)):
            policy.append([lstStrand[i][0]])
        return policy
    
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

    # Assemble the strands
    assembled_strands = []
    for strand in strands:
        strand = strand.split()[2:]
        assembled_strand = [(int(strand[i]), int(strand[i+1])) for i in range(0, len(strand)-1, 2)]
        if 'R' in strand[-1]:
            assembled_strand[-1] = (assembled_strand[-1][0], int(strand[-1][1:]))
        assembled_strands.append(assembled_strand)

    # Call monteCarlo function
    initRes = monteCarlo(vertexCt, [(-4,visitCtLim), (-5,strandCtLim)])
    print(initRes)
    for i, strnd in enumerate(assembled_strands):
        res = monteCarlo(1+i-len(assembled_strands), strnd)

    print(f"Policy {res=}")  # res stands for result

if __name__ == "__main__":
    main()

#Om Gole, 6, 2024