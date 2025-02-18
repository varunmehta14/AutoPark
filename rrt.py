import pygame
import heapq
import math
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car with A* Pathfinding and Visualization")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # Non-drivable areas
GREEN = (0, 255, 0)  # Empty slots
RED = (255, 0, 0)  # Occupied slots
YELLOW = (255, 255, 0)  # Target slot
LIGHT_BLUE = (173, 216, 230)  # Drivable areas

# Parking lot parameters
PARKING_SPOT_WIDTH = 50
PARKING_SPOT_HEIGHT = 80
PARKING_START_X = 200
PARKING_START_Y = 100
PARKING_SPACING_X = 10
PARKING_SPACING_Y = 90  # Vertical spacing

# Generate random occupied spots
NUM_SPOTS_PER_ROW = 10
NUM_ROWS = 4
NUM_OCCUPIED_SPOTS = 30

occupied_spots = set()
while len(occupied_spots) < NUM_OCCUPIED_SPOTS:
    row = random.randint(0, NUM_ROWS - 1)
    col = random.randint(0, NUM_SPOTS_PER_ROW - 1)
    occupied_spots.add((row, col))

# Load car image
car_image = pygame.image.load("car.png")  # Replace with your car image file
car_width, car_height = 30, 60  # Resize dimensions
car_image = pygame.transform.scale(car_image, (car_width, car_height))

# Car parameters
car = pygame.Rect(WIDTH // 2, HEIGHT - 100, car_width, car_height)  # Initial position
car_speed = 2
car_angle = 0  # Initial angle
target_slot = None  # Initially no target

# Parking spots and non-drivable areas
parking_spots = []
non_drivable_areas = []

for row in range(NUM_ROWS):
    y = PARKING_START_Y + row * (PARKING_SPOT_HEIGHT + PARKING_SPACING_Y)
    for col in range(NUM_SPOTS_PER_ROW):
        x = PARKING_START_X + col * (PARKING_SPOT_WIDTH + PARKING_SPACING_X)
        rect = pygame.Rect(x, y, PARKING_SPOT_WIDTH, PARKING_SPOT_HEIGHT)
        parking_spots.append({"rect": rect, "occupied": (row, col) in occupied_spots})
        if col < NUM_SPOTS_PER_ROW - 1:
            gap_x = x + PARKING_SPOT_WIDTH
            non_drivable_areas.append(pygame.Rect(gap_x, y, PARKING_SPACING_X, PARKING_SPOT_HEIGHT))

# A* Algorithm
def a_star(start, goal, grid_size=10):
    def heuristic(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = {start: 0}
    came_from = {}
    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for dx, dy in [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size),
                       (-grid_size, -grid_size), (-grid_size, grid_size), (grid_size, -grid_size), (grid_size, grid_size)]:
            neighbor = (current[0] + dx, current[1] + dy)

            neighbor_rect = pygame.Rect(neighbor[0], neighbor[1], grid_size, grid_size)
            collision = False
            for spot in parking_spots:
                if spot["occupied"] and neighbor_rect.colliderect(spot["rect"]):
                    collision = True
                    break
            for area in non_drivable_areas:
                if neighbor_rect.colliderect(area):
                    collision = True
                    break

            if collision or neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current

    return []

# Find all paths and choose the closest
def find_all_paths_and_choose_closest():
    global target_slot, all_paths

    car_center = car.center
    all_paths = []  # Store all paths for visualization
    min_distance = float("inf")
    closest_path = None
    closest_slot = None

    # Generate paths to all empty slots
    for spot in parking_spots:
        if not spot["occupied"]:  # Only consider empty slots
            goal = spot["rect"].center
            path = a_star(car_center, goal)

            if path:
                # Calculate path cost (distance)
                path_cost = sum(
                    math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    for i in range(len(path) - 1)
                )
                all_paths.append((spot["rect"], path, path_cost))

                # Check if this is the closest path
                if path_cost < min_distance:
                    min_distance = path_cost
                    closest_path = path
                    closest_slot = spot["rect"]

    # Return the closest slot and path
    return closest_slot, closest_path

# Move car along the path
def move_car_along_path(waypoints):
    global car

    if not waypoints:
        return False

    # Get the next waypoint
    next_waypoint = waypoints[0]
    car_center = car.center

    # Calculate the distance to the next waypoint
    dx = next_waypoint[0] - car_center[0]
    dy = next_waypoint[1] - car_center[1]
    distance = math.hypot(dx, dy)

    # If close enough to the waypoint, move to the next one
    if distance < 5:
        waypoints.pop(0)
        return False

    # Move the car towards the waypoint
    car.x += car_speed * dx / distance
    car.y += car_speed * dy / distance

    return False  # Car is still moving

# Main loop
running = True
clock = pygame.time.Clock()
car_parked = False
waypoints = []
all_paths = []  # To store all paths for persistent visualization

while running:
    screen.fill(LIGHT_BLUE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw parking spots
    for spot in parking_spots:
        color = RED if spot["occupied"] else GREEN
        pygame.draw.rect(screen, color, spot["rect"])
        pygame.draw.rect(screen, BLACK, spot["rect"], 2)

    # Draw non-drivable areas
    for area in non_drivable_areas:
        pygame.draw.rect(screen, BLACK, area)

    # Find paths to all slots and choose the closest
    if target_slot is None:
        target_slot, waypoints = find_all_paths_and_choose_closest()

    # Draw all paths (light gray) for visualization
    for slot, path, _ in all_paths:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, (0, 0, 0), path[i], path[i + 1], 2)  # Light gray paths

    # Move the car along the waypoints
    if waypoints and not car_parked:
        car_parked = move_car_along_path(waypoints)

    # Draw the chosen path (green) and highlight the chosen slot (yellow)
    if target_slot:
        pygame.draw.rect(screen, YELLOW, target_slot, 4)  # Highlight the target slot
        for i in range(len(waypoints) - 1):
            pygame.draw.line(screen, (0, 255, 0), waypoints[i], waypoints[i + 1], 4)  # Highlight the chosen path in green

    # Draw the car
    rotated_car = pygame.transform.rotate(car_image, 0)  # No rotation for simplicity
    rotated_rect = rotated_car.get_rect(center=car.center)
    screen.blit(rotated_car, rotated_rect.topleft)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
