
x = [ [15, 25], [40, 60] ]
y = [ [10, 20], [50, 70], [80, 90] ]


# def merge(intervals):
#     result = []
#     intervals = sorted(intervals)
#     s1, e1 = intervals[0]
#
#     for s2, e2 in intervals[1:]:
#         if (s2 <= e1):
#             e1 = max(e1, e2)
#         elif s2 > e1:
#             result.append([s1, e1])
#             s1, e1 = s2, e2
#     result.append([s1, e1])
#     return result

def merge(a, b):
    intervals = a + b
    intervals.sort()
    res = [intervals[0]]
    for i in intervals[1:]:
        if res[-1][1] >= i[0]:
            res[-1][1] = max(i[1], res[-1][1])
        else:
            res.append(i)
    return res
if __name__ == '__main__':
    print(merge(x, y))