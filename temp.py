import re

# print(33 // 2)

def process_vslcs(vslcs_str, graph_size):
    vertices = []
    vslcs_list = vslcs_str.split(',')
    for vslc in vslcs_list:
        if '::' in vslc:
            start, step = map(int, vslc.split('::'))
            if start < 0:
                start += graph_size
            vertices.extend(list(range(start, graph_size, step)))
        elif ':' in vslc:
            parts = vslc.split(':')
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else graph_size
            if start < 0:
                start += graph_size
            if end < 0:
                end += graph_size
            vertices.extend(list(range(start, end)))
        else:
            vertex = int(vslc)
            if vertex < 0:
                vertex += graph_size
            vertices.append(vertex)
    return vertices


# # Test the function with your examples
# print(process_vslcs("-3", 12))  # Output: [9]
# print(process_vslcs(":", 12))  # Output: [9]
# print(process_vslcs("1::2", 12))  # Output: [1, 3, 5, 7, 9, 11]
# print(process_vslcs("-2::-3,3::4", 12))  # Output: [10, 7, 4, 1, 3, 7, 11]
# print(process_vslcs("1,-2,3", 12))  # Output: [1, 10, 3]
# print(process_vslcs("1:4,6:9,-2:", 12))  # Output: [1, 2, 3, 6, 7, 8, 10, 11]
# print(process_vslcs("1::3,5::3,-2:", 12))  # Output: [1, 4, 7, 10, 5, 8, 11, 10, 11]


