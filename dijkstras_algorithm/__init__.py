import heapq

def edge_costs(
    feature_function, vertex, constant=0, heuristic=0, feature_function_index=2
):
    return (
        constant
        + 1 * feature_function[vertex][0]
        + heuristic * feature_function[vertex][feature_function_index]
    )


class Vertex:
    def __init__(self):
        self.d = float("Inf")
        self.parent = None
        self.finished = False
        self.coordinate = None


def dijkstra_bwd(
    feature_function,
    vertices,
    source,
    neighbors,
    max_len=625,
    constant=0,
    heuristic=0,
    heuristic_index=2,
):
    nodes = {}
    for node in vertices:
        nodes[node] = Vertex()
    nodes[source].d = 0
    queue = [(0, source)]  # priority queue
    while queue:
        d, node = heapq.heappop(queue)
        if nodes[node].finished:
            continue
        nodes[node].finished = True
        neighbor_nodes = neighbors[node]
        for neighbor in neighbor_nodes:
            neighbor = int(neighbor)
            if neighbor == max_len:
                continue
            if nodes[neighbor].finished:
                continue
            new_d = d + edge_costs(
                feature_function, neighbor, constant, heuristic, heuristic_index
            )
            if new_d < nodes[neighbor].d:
                nodes[neighbor].d = new_d
                nodes[neighbor].parent = node
                heapq.heappush(queue, (new_d, neighbor))
    return nodes


def dijkstra_fwd(feature_function, vertices, source, node_f, neighbors, max_len=-1):
    nodes = {}
    for node in vertices:
        nodes[node] = Vertex()
    nodes[source].d = 0
    nodes[source].coordinate = 0
    queue = [(0, source)]  # priority queue
    while queue:
        d, node = heapq.heappop(queue)
        if nodes[node].finished:
            continue
        nodes[node].finished = True
        if node == node_f:
            break
        neighbor_nodes = neighbors[node]
        for neighbor in neighbor_nodes:
            neighbor = int(neighbor)
            if neighbor == max_len:
                continue
            if nodes[neighbor].finished:
                continue
            new_d = d + edge_costs(feature_function, neighbor)
            if new_d < nodes[neighbor].d:
                nodes[neighbor].d = new_d
                nodes[neighbor].parent = node
                nodes[neighbor].coordinate = nodes[node].coordinate + 1
                heapq.heappush(queue, (new_d, neighbor))
    return nodes
