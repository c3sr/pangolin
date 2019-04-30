from __future__ import print_function

A = [0,0,1,2,4,4,5,5,5]
B = [0,0,1,1,1,3,3,3,4]

def binary_search(d, A, B, bounds):

    assert d < len(A) + len(B)

    if d > len(A):
        aTop = len(A)
        bBot = d-len(A)
    else:
        aTop = d
        bBot = 0

    if d > len(B):
        bTop = len(B)
        aBot = d - len(B)
    else:
        bTop = d
        aBot = 0

    # print("diag %d: A[%d:%d]=%s B[%d:%d]=%s" % (d, aBot, aTop, str(A[aBot:aTop]), bBot, bTop, str(B[bBot:bTop])))

    iter = 0
    begin = max(0, d - len(B))
    end = min(d, len(A))
    while begin < end:
        mid = (begin + end) // 2
        aKey = A[mid]
        bKey = B[d - 1 - mid]

        if bounds == "upper":
            pred = aKey < bKey
        else:
            pred = bKey >= aKey
        if pred:
            begin = mid + 1
        else:
            end = mid
    return begin


C = 4
for c in range(0, len(A) + len(B), C):
    ub = binary_search(c, A, B, "upper")
    lb = binary_search(c, A, B, "lower")
    print("merge path at diag", c, (lb, c - lb))
