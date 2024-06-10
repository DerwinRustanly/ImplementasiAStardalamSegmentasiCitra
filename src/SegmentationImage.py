import cv2
import numpy as np
from matplotlib import pyplot as plt
import heapq

def kmeans_segmentation(image_path, k):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return image, segmented_image, labels.reshape(image.shape[:2]), centers

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def path_finding_algorithm(start, goal, labels, algorithm):
    rows, cols = labels.shape
    open_set = []
    heapq.heappush(open_set, (0, start))  # Priority, current
    came_from = {}
    cost_so_far = {start: 0}  # A* and UCS will use this

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return labels[goal[0], goal[1]]

        neighbors = [(current[0] + dr, current[1] + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        neighbors = [(r, c) for r, c in neighbors if 0 <= r < rows and 0 <= c < cols]

        for neighbor in neighbors:
            if algorithm in ['A*', 'UCS']:
                new_cost = cost_so_far[current] + 1  # Assume uniform cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    if algorithm == 'A*':
                        priority += heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
            elif algorithm == 'GBFS':
                # For GBFS, use heuristic only for the priority queue
                if neighbor not in came_from:
                    priority = heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current

    return labels[start[0], start[1]]  # fallback if no path found


def refine_labels(segmented_image, labels, algorithm):
    rows, cols = labels.shape
    refined_labels = labels.copy()
    for r in range(rows):
        for c in range(cols):
            if r > 0 and labels[r, c] != labels[r-1, c]:
                refined_labels[r, c] = path_finding_algorithm((r, c), (r-1, c), labels, algorithm)
            if c > 0 and labels[r, c] != labels[r, c-1]:
                refined_labels[r, c] = path_finding_algorithm((r, c), (r, c-1), labels, algorithm)
    refined_image = np.zeros_like(segmented_image)
    for r in range(rows):
        for c in range(cols):
            refined_image[r, c] = segmented_image[r, c]
    return refined_image, refined_labels

def calculate_metrics(original_labels, refined_labels):
    intersection = np.sum(original_labels == refined_labels)
    union = np.prod(original_labels.shape)
    return 2 * intersection / (union + intersection), intersection / union

# Path to image and parameters
image_path = 'peppers512warna.bmp'
k = 3

# Process image
original_image, segmented_image, segmented_labels, centers = kmeans_segmentation(image_path, k)

# Refine segmentation
refined_image_a, refined_labels_a = refine_labels(segmented_image, segmented_labels, 'A*')

# Calculate metrics
dice_a, jaccard_a = calculate_metrics(segmented_labels, refined_labels_a)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
axes[0].imshow(original_image), axes[0].set_title('Original Image')
axes[1].imshow(refined_image_a), axes[1].set_title(f'A* Refinement\nDice: {dice_a:.4f}, Jaccard: {jaccard_a:.4f}')
plt.show()
