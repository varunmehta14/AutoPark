import pygame
import heapq #used for A*
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
NUM_OCCUPIED_SPOTS = 39

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

rotated_car = pygame.transform.rotate(car_image, car_angle)

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
            
            
# Generate random obstacles
#Add Obstacles
NUM_OBSTACLES = 10
obstacles = []

# Define the bounds for obstacle generation
OBSTACLE_PADDING = 50  # Distance from the grid
OBSTACLE_AREA_X_MIN = PARKING_START_X - OBSTACLE_PADDING
OBSTACLE_AREA_X_MAX = PARKING_START_X + NUM_SPOTS_PER_ROW * (PARKING_SPOT_WIDTH + PARKING_SPACING_X) + OBSTACLE_PADDING
OBSTACLE_AREA_Y_MIN = PARKING_START_Y - OBSTACLE_PADDING
OBSTACLE_AREA_Y_MAX = PARKING_START_Y + NUM_ROWS * (PARKING_SPOT_HEIGHT + PARKING_SPACING_Y) + OBSTACLE_PADDING

# Generate obstacles near the parking grid
while len(obstacles) < NUM_OBSTACLES:
    obstacle_x = random.randint(OBSTACLE_AREA_X_MIN, OBSTACLE_AREA_X_MAX - 100)
    obstacle_y = random.randint(OBSTACLE_AREA_Y_MIN, OBSTACLE_AREA_Y_MAX - 100)
    obstacle_width = random.randint(30, 60)  # Smaller obstacles for tighter spaces
    obstacle_height = random.randint(30, 60)
    obstacle = pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)

    # Ensure obstacles do not overlap with parking spots or car's starting area
    collision = any(obstacle.colliderect(spot["rect"]) for spot in parking_spots) or obstacle.colliderect(car)
    if not collision:
        obstacles.append(obstacle)

# Dynamic moving cars
NUM_DYNAMIC_CARS = 5
dynamic_cars = []
for _ in range(NUM_DYNAMIC_CARS):
    x = random.randint(PARKING_START_X, WIDTH - car_width)
    y = random.randint(PARKING_START_Y, HEIGHT // 2)
    rect = pygame.Rect(x, y, car_width, car_height)
    speed_x = random.choice([-2, 2])
    speed_y = random.choice([-1, 1])
    dynamic_cars.append({"rect": rect, "speed": (speed_x, speed_y)})

# Function to move dynamic cars
def move_dynamic_cars():
    for car in dynamic_cars:
        car["rect"].x += car["speed"][0]
        car["rect"].y += car["speed"][1]

        # Bounce off walls
        if car["rect"].left < 0 or car["rect"].right > WIDTH:
            car["speed"] = (-car["speed"][0], car["speed"][1])
        if car["rect"].top < 0 or car["rect"].bottom > HEIGHT:
            car["speed"] = (car["speed"][0], -car["speed"][1])

def a_star_with_obstacles(start, goal, detected_obstacles, grid_size=10):
    """
    A* pathfinding algorithm that avoids obstacles, occupied parking slots, and non-drivable areas.
    """
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

            # Check collision with dynamically detected obstacles
            for obstacle in detected_obstacles:
                if neighbor_rect.colliderect(obstacle):
                    collision = True
                    break

            # Check collision with occupied parking spots
            for spot in parking_spots:
                if spot["occupied"] and neighbor_rect.colliderect(spot["rect"]):
                    collision = True
                    break

            # Check collision with non-drivable areas
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
    
def find_all_paths_and_choose_closest_with_lidar():
    """
    Finds paths to all unoccupied parking slots dynamically based on obstacles
    detected by the LiDAR sensor.
    """
    global target_slot, all_paths

    car_center = car.center
    all_paths = []  # Store all paths for visualization
    min_distance = float("inf")
    closest_path = None
    closest_slot = None

    # Get detected obstacles dynamically from the LiDAR sensor
    detected_obstacles = [
        pygame.Rect(point[0] - 5, point[1] - 5, 10, 10)  # Approximate point as a small rectangle
        for obj_type, _, point in lidar_sensor(car, parking_spots, obstacles)
        if obj_type == "Obstacle"
    ]

    # Generate paths to all empty slots
    for spot in parking_spots:
        if not spot["occupied"]:  # Only consider empty slots
            goal = spot["rect"].center

            # Update A* to avoid only detected obstacles
            path = a_star_with_obstacles(car_center, goal, detected_obstacles)

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


def move_car_along_path(waypoints):
    """
    Moves the car along the given waypoints.
    Returns True if the car is parked, otherwise False.
    """
    global car  # Ensure we modify the car's position globally

    if not waypoints:
        return True  # If there are no waypoints, the car is parked

    # Get the next waypoint
    next_waypoint = waypoints[0]
    car_center = car.center

    # Calculate the distance to the next waypoint
    dx = next_waypoint[0] - car_center[0]
    dy = next_waypoint[1] - car_center[1]
    distance = math.hypot(dx, dy)

    # If close enough to the waypoint, move to the next one
    if distance < 5:  # Threshold for reaching the waypoint
        waypoints.pop(0)
        return len(waypoints) == 0  # Return True if all waypoints are completed

    # Move the car towards the waypoint
    car.x += car_speed * dx / distance
    car.y += car_speed * dy / distance

    return False  # The car is still moving


# Main loop
running = True
clock = pygame.time.Clock()
car_parked = False
waypoints = []
all_paths = []  # To store all paths for persistent visualization


def lidar_sensor(car, parking_spots, obstacles, num_rays=36, max_range=200):
    """
    Simulated LiDAR sensor for the car.
    Casts rays around the car to detect objects.

    Args:
        car: The car's Rect object.
        parking_spots: List of parking spots.
        obstacles: List of obstacles.
        num_rays: Number of rays to cast.
        max_range: Maximum range of the sensor.

    Returns:
        List of detected objects with their types and distances.
    """
    car_center = car.center
    detected_objects = []

    # Cast rays in a 360-degree circle
    for i in range(num_rays):
        angle = math.radians(i * (360 / num_rays))
        ray_x = car_center[0] + max_range * math.cos(angle)
        ray_y = car_center[1] + max_range * math.sin(angle)
        ray_end = (ray_x, ray_y)

        # Check intersections with obstacles and parking spots
        min_distance = max_range
        detected_object = None

        for obstacle in obstacles:
            if obstacle.clipline(car_center, ray_end):
                intersection = obstacle.clipline(car_center, ray_end)[0]
                distance = math.hypot(intersection[0] - car_center[0], intersection[1] - car_center[1])
                if distance < min_distance:
                    min_distance = distance
                    detected_object = ("Obstacle", intersection)

        for spot in parking_spots:
            if spot["rect"].clipline(car_center, ray_end):
                intersection = spot["rect"].clipline(car_center, ray_end)[0]
                distance = math.hypot(intersection[0] - car_center[0], intersection[1] - car_center[1])
                if distance < min_distance:
                    min_distance = distance
                    detected_object = ("Parking Spot", intersection)

        # Record the closest detected object in this ray's direction
        if detected_object:
            detected_objects.append((detected_object[0], min_distance, detected_object[1]))

    return detected_objects

def log_detected_obstacles():
    detected_obstacles = [
        {"type": obj_type, "distance": distance, "location": point}
        for obj_type, distance, point in lidar_sensor(car, parking_spots, obstacles)
        if obj_type == "Obstacle"
    ]
    print("\nDetected Obstacles:")
    for obstacle in detected_obstacles:
        print(f"Type: {obstacle['type']}, Distance: {obstacle['distance']:.2f}, Location: {obstacle['location']}")
    return detected_obstacles

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     screen.fill(WHITE)

#     # Draw parking spots
#     for spot in parking_spots:
#         color = RED if spot["occupied"] else GREEN
#         pygame.draw.rect(screen, color, spot["rect"])

#     # Draw obstacles
#     for obstacle in obstacles:
#         pygame.draw.rect(screen, BLACK, obstacle)

#     # Draw dynamic cars
#     move_dynamic_cars()
#     for dcar in dynamic_cars:
#         pygame.draw.rect(screen, YELLOW, dcar["rect"])

#     # Main car logic
#     if not car_parked:
#         if not waypoints:
#             target_slot, waypoints = find_all_paths_and_choose_closest_with_lidar()
#         else:
#             car_parked = move_car_along_path(waypoints)

#     # Draw the main car
#     screen.blit(rotated_car, car)

#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()

while running:
    screen.fill(LIGHT_BLUE)
    print("\nDetected Obstacles:")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("QUIT event detected, exiting...", flush=True)
            log_detected_obstacles()  # Log obstacles before quitting
            running = False

    # Draw parking spots
    for spot in parking_spots:
        color = RED if spot["occupied"] else GREEN
        pygame.draw.rect(screen, color, spot["rect"])
        pygame.draw.rect(screen, BLACK, spot["rect"], 2)

    # Draw non-drivable areas
    for area in non_drivable_areas:
        pygame.draw.rect(screen, BLACK, area)

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, (128, 0, 128), obstacle)  # Purple for obstacles

    # Visualize LiDAR rays
    detected_objects = lidar_sensor(car, parking_spots, obstacles)
    for obj_type, distance, point in detected_objects:
        pygame.draw.line(screen, (255, 255, 0), car.center, point, 1)  # Yellow rays
        pygame.draw.circle(screen, (255, 0, 0), (int(point[0]), int(point[1])), 5)  # Red dots

    # Pathfinding
    if target_slot is None:
        target_slot, waypoints = find_all_paths_and_choose_closest_with_lidar()
        print(f"Target slot selected: {target_slot}, Waypoints: {waypoints}", flush=True)

    # Draw all paths (light gray) for visualization
    for slot, path, _ in all_paths:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, (0, 0, 0), path[i], path[i + 1], 2)

    # Draw dynamic cars
    move_dynamic_cars()
    for dcar in dynamic_cars:
        pygame.draw.rect(screen, YELLOW, dcar["rect"])

#     # Main car logic
#     if not car_parked:
#         if not waypoints:
#             target_slot, waypoints = find_all_paths_and_choose_closest_with_lidar()
#         else:
#             car_parked = move_car_along_path(waypoints)

    
    # Move the car along the waypoints
    if waypoints and not car_parked:
        car_parked = move_car_along_path(waypoints)

    # Check if the car has stopped moving
    if car_parked and not waypoints:
        print("The car has stopped moving and is parked!", flush=True)
        detected_obstacles = log_detected_obstacles()
        break

    # Draw the chosen path and target slot
    if target_slot:
        pygame.draw.rect(screen, YELLOW, target_slot, 4)
        for i in range(len(waypoints) - 1):
            pygame.draw.line(screen, (0, 255, 0), waypoints[i], waypoints[i + 1], 4)

    # Draw the car
    rotated_car = pygame.transform.rotate(car_image, 0)
    rotated_rect = rotated_car.get_rect(center=car.center)
    screen.blit(rotated_car, rotated_rect.topleft)

    pygame.display.flip()
    clock.tick(60)

log_detected_obstacles()  # Log obstacles in case parking condition fails
pygame.quit()