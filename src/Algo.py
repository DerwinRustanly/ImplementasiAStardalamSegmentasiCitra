import heapq
import numpy as np

# Fungsi Heuristik (contoh sederhana, dapat disesuaikan dengan kasus nyata)
def heuristic(node, goal):
    # Misalkan fungsi heuristik adalah jarak Euclidean
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

# Uniform Cost Search
def uniform_cost_search(start, goal, graph):
    visited = set()
    priority_queue = [(0, start)]
    while priority_queue:
        cost, node = heapq.heappop(priority_queue)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return cost
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (cost + weight, neighbor))
    return float('inf')

# Greedy Best First Search
def greedy_best_first_search(start, goal, graph):
    visited = set()
    priority_queue = [(heuristic(start, goal), start)]
    while priority_queue:
        _, node = heapq.heappop(priority_queue)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return True
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (heuristic(neighbor, goal), neighbor))
    return False

# A* Search
def a_star_search(start, goal, graph):
    visited = set()
    priority_queue = [(0 + heuristic(start, goal), 0, start)]
    while priority_queue:
        _, cost, node = heapq.heappop(priority_queue)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return cost
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(priority_queue, (cost + weight + heuristic(neighbor, goal), cost + weight, neighbor))
    return float('inf')
