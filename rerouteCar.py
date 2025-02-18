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


# Generate random obstacles with names
NUM_OBSTACLES = 20
obstacles = []

# Define the bounds for obstacle generation
OBSTACLE_PADDING = 50  # Distance from the grid
OBSTACLE_AREA_X_MIN = PARKING_START_X - OBSTACLE_PADDING
OBSTACLE_AREA_X_MAX = PARKING_START_X + NUM_SPOTS_PER_ROW * (PARKING_SPOT_WIDTH + PARKING_SPACING_X) + OBSTACLE_PADDING
OBSTACLE_AREA_Y_MIN = PARKING_START_Y - OBSTACLE_PADDING
OBSTACLE_AREA_Y_MAX = PARKING_START_Y + NUM_ROWS * (PARKING_SPOT_HEIGHT + PARKING_SPACING_Y) + OBSTACLE_PADDING

# Generate obstacles near the parking grid
obstacle_count = 1  # Counter for naming obstacles
while len(obstacles) < NUM_OBSTACLES:
    obstacle_x = random.randint(OBSTACLE_AREA_X_MIN, OBSTACLE_AREA_X_MAX - 100)
    obstacle_y = random.randint(OBSTACLE_AREA_Y_MIN, OBSTACLE_AREA_Y_MAX - 100)
    obstacle_width = random.randint(30, 60)  # Smaller obstacles for tighter spaces
    obstacle_height = random.randint(30, 60)
    obstacle_rect = pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)

    # Ensure obstacles do not overlap with parking spots or car's starting area
    collision = any(obstacle_rect.colliderect(spot["rect"]) for spot in parking_spots) or obstacle_rect.colliderect(car)
    if not collision:
        # Create a named obstacle and add it to the list
        obstacle = {
            "name": f"Obstacle {obstacle_count}",
            "rect": obstacle_rect
        }
        obstacles.append(obstacle)
        obstacle_count += 1
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

# def a_star_with_obstacles(start, goal, detected_obstacles, grid_size=10):
#     """
#     A* pathfinding algorithm that avoids obstacles, occupied parking slots, and non-drivable areas.
#     """
#     def heuristic(pos1, pos2):
#         # Manhattan distance heuristic
#         return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     g_score = {start: 0}
#     f_score = {start: heuristic(start, goal)}
#     came_from = {}
#     closed_set = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if current == goal:
#             # Reconstruct the path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]

#         closed_set.add(current)

#         for dx, dy in [
#             (-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size),
#             (-grid_size, -grid_size), (-grid_size, grid_size), (grid_size, -grid_size), (grid_size, grid_size)
#         ]:
#             neighbor = (current[0] + dx, current[1] + dy)

#             # Define the rectangle for the neighbor to check collisions
#             neighbor_rect = pygame.Rect(neighbor[0], neighbor[1], grid_size, grid_size)
#             collision = False

#             # Check collision with dynamically detected obstacles
#             for obstacle in detected_obstacles:
#                 if neighbor_rect.colliderect(obstacle):
#                     collision = True
#                     break

#             # Check collision with occupied parking spots
#             for spot in parking_spots:
#                 if spot["occupied"] and neighbor_rect.colliderect(spot["rect"]):
#                     collision = True
#                     break

#             # Check collision with non-drivable areas
#             for area in non_drivable_areas:
#                 if neighbor_rect.colliderect(area):
#                     collision = True
#                     break

#             if collision or neighbor in closed_set:
#                 continue

#             tentative_g_score = g_score[current] + math.hypot(dx, dy)

#             # Add a penalty for proximity to obstacles (encourage avoiding obstacles)
#             for obstacle in detected_obstacles:
#                 if neighbor_rect.colliderect(obstacle.inflate(20, 20)):  # Inflate to create a "buffer"
#                     tentative_g_score += 50  # Add a high penalty for proximity

#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))
#                 came_from[neighbor] = current

#     return []

def a_star_with_dynamic_rerouting(car_rect, goal, detected_obstacles, grid_size=10):
    """
    A* pathfinding with real-time rerouting based on nearby obstacle detection.
    """
    def heuristic(pos1, pos2):
        # Manhattan distance heuristic
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(pos):
        # Generate 8 potential neighbors
        for dx, dy in [
            (-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size),
            (-grid_size, -grid_size), (-grid_size, grid_size), (grid_size, -grid_size), (grid_size, grid_size)
        ]:
            yield pos[0] + dx, pos[1] + dy

    open_set = []
    start = (car_rect.centerx, car_rect.centery)
    heapq.heappush(open_set, (0, start))
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors(current):
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

            tentative_g_score = g_score[current] + math.hypot(neighbor[0] - current[0], neighbor[1] - current[1])

            # Add a penalty for proximity to obstacles
            for obstacle in detected_obstacles:
                if neighbor_rect.colliderect(obstacle.inflate(20, 20)):  # Inflate to create a "buffer"
                    tentative_g_score += 50  # Add a high penalty for proximity

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                came_from[neighbor] = current

    return []



move_dynamic_cars()

# Generate paths to all empty slots
def find_all_paths_and_choose_closest_with_lidar():
    global target_slot, all_paths

    car_center = car.center
    all_paths = []  # Store all paths for visualization
    min_distance = float("inf")
    closest_path = None
    closest_slot = None

    # Exclude obstacles detected by LiDAR
    detected_obstacles = [obstacle["rect"] for obstacle in obstacles]

    # Generate paths to all empty slots
    for spot in parking_spots:
        if not spot["occupied"]:  # Only consider empty slots
            goal = spot["rect"].center
            car_rect = pygame.Rect(car.centerx - car_width // 2, car.centery - car_height // 2, car_width, car_height)
            path = a_star_with_dynamic_rerouting(car_rect, goal, detected_obstacles)

            if path:
                # Calculate path cost (distance)
                path_cost = sum(
                    math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    for i in range(len(path) - 1)
                )
                all_paths.append((spot["rect"], path, path_cost))  # Store the path for visualization

                # Check if this is the closest path
                if path_cost < min_distance:
                    min_distance = path_cost
                    closest_path = path
                    closest_slot = spot["rect"]

    # Debugging: Print all paths and their costs
    print("\nAll Paths:")
    for slot, path, cost in all_paths:
        print(f"To Slot {slot.center}: Path={path}, Cost={cost:.2f}")

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

# Function to calculate Euclidean distance
def calculate_distance(location1, location2):
    return math.sqrt((location1.x - location2.x)**2 + 
                     (location1.y - location2.y)**2 + 
                     (location1.z - location2.z)**2)

def should_reroute(car_rect, obstacle_rect, car_speed, obstacle_speed, threshold_distance=20.0):
    car_future_position = pygame.Rect(
        car_rect.x + car_speed[0], car_rect.y + car_speed[1],
        car_rect.width, car_rect.height
    )
    obstacle_future_position = pygame.Rect(
        obstacle_rect.x + obstacle_speed[0], obstacle_rect.y + obstacle_speed[1],
        obstacle_rect.width, obstacle_rect.height
    )

    current_distance = math.hypot(car_rect.centerx - obstacle_rect.centerx,
                                  car_rect.centery - obstacle_rect.centery)
    future_distance = math.hypot(car_future_position.centerx - obstacle_future_position.centerx,
                                 car_future_position.centery - obstacle_future_position.centery)

    return current_distance < threshold_distance or future_distance < threshold_distance


def check_and_reroute(car, waypoints, detected_objects, car_speed):
    global target_slot

    for obj in detected_objects:
        obstacle_rect = pygame.Rect(obj.get("location", (0, 0))[0], obj.get("location", (0, 0))[1], 1, 1)
        obstacle_speed = (0, 0) if obj.get("type") == "Obstacle" else obj.get("speed", (0, 0))
        if should_reroute(car, obstacle_rect, car_speed, obstacle_speed):
            print(f"Potential collision with {obj.get('name', 'Unknown')}! Rerouting...", flush=True)
            target_slot, new_waypoints = find_all_paths_and_choose_closest_with_lidar()
            return new_waypoints
# Main loop
running = True
clock = pygame.time.Clock()
car_parked = False
waypoints = []
all_paths = []  # To store all paths for persistent visualization

def lidar_sensor(car, parking_spots, obstacles, dynamic_cars, num_rays=36, max_range=200):
    car_center = car.center
    detected_objects = []

    for i in range(num_rays):
        angle = math.radians(i * (360 / num_rays))
        ray_x = car_center[0] + max_range * math.cos(angle)
        ray_y = car_center[1] + max_range * math.sin(angle)
        ray_end = (ray_x, ray_y)

        min_distance = max_range
        detected_object = None

        # Check intersections with obstacles
        for obstacle in obstacles:
            rect = obstacle["rect"]
            if rect.clipline(car_center, ray_end):
                intersection = rect.clipline(car_center, ray_end)[0]
                distance = math.hypot(intersection[0] - car_center[0], intersection[1] - car_center[1])
                if distance < min_distance:
                    min_distance = distance
                    detected_object = {
                        "name": obstacle.get("name", "Unknown"),
                        "type": "Obstacle",
                        "distance": distance,
                        "location": intersection,
                    }

                    

        # Check intersections with dynamic cars
        for idx, dcar in enumerate(dynamic_cars):
            rect = dcar["rect"]
            if rect.clipline(car_center, ray_end):
                intersection = rect.clipline(car_center, ray_end)[0]
                distance = math.hypot(intersection[0] - car_center[0], intersection[1] - car_center[1])
                if distance < min_distance:
                    min_distance = distance
                    # For dynamic cars
                    detected_object = {
                        "name": f"DynamicCar{idx}",
                        "type": "DynamicCar",
                        "distance": distance,
                        "location": intersection,
                        "speed": dcar.get("speed", (0, 0))
                    }

        if detected_object:
            detected_objects.append(detected_object)

    return detected_objects


#***********to  detect unique obs*****************************8
def log_detected_obstacles(detected_objects):
    """
    Processes and returns a list of unique detected obstacles.
    Args:
        detected_objects: List of all detected objects from the LiDAR sensor.
    """
    unique_obstacles = {}

    for obj in detected_objects:
        # Use the obstacle name as a unique identifier
        if obj["name"] not in unique_obstacles:
            unique_obstacles[obj["name"]] = {
                "type": obj["type"],
                "distance": obj["distance"],
                "location": obj["location"],
                "name": obj["name"],
                "speed": obj.get("speed", (0, 0))  # Add speed, default to (0, 0) if not present
            }

    # Print the unique obstacles with their speeds
    for obstacle in unique_obstacles.values():
        print(f"Name: {obstacle['name']}, Type: {obstacle['type']}, Distance: {obstacle['distance']:.2f}, Location: {obstacle['location']}, Speed: {obstacle['speed']}")

    return list(unique_obstacles.values())


simulation_running = True  # Flag to keep the simulation running after parking
unique_detected_objects = []  # List to accumulate detected obstacles

def pid_speed_control(current_speed, target_speed, obstacle_distance, obstacle_speed, Kp=1.0, Ki=0.1, Kd=0.05):
    error = target_speed - current_speed
    integral = 0  # You need to maintain this between function calls
    derivative = obstacle_speed  # Rate of change of obstacle distance

    output = Kp * error + Ki * integral + Kd * derivative

    # Adjust speed based on obstacle distance
    if obstacle_distance < 10:  # Threshold distance
        output = min(output, obstacle_distance / 2)  # Reduce speed as we get closer

    return max(0, min(output, target_speed))  # Ensure speed is between 0 and target_speed

while running:
    screen.fill(LIGHT_BLUE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("QUIT event detected, exiting...", flush=True)
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
        pygame.draw.rect(screen, (128, 0, 128), obstacle["rect"])  # Purple for obstacles

    # Draw dynamic cars
    move_dynamic_cars()
    for dcar in dynamic_cars:
        pygame.draw.rect(screen, YELLOW, dcar["rect"])

    # Visualize LiDAR rays and accumulate detected objects
    detected_objects = lidar_sensor(car, parking_spots, obstacles, dynamic_cars, num_rays=72, max_range=300)
    unique_detected_objects.extend(detected_objects)  # Add new detected objects

    for obj in detected_objects:
        pygame.draw.line(screen, (255, 255, 0), car.center, obj["location"], 1)  # Yellow rays
        pygame.draw.circle(screen, (255, 0, 0), (int(obj["location"][0]), int(obj["location"][1])), 5)  # Red dots

    # Pathfinding and movement
    if simulation_running:
        if target_slot is None:
            target_slot, waypoints = find_all_paths_and_choose_closest_with_lidar()
            print(f"Target slot selected: {target_slot}, Waypoints: {waypoints}", flush=True)

    
    # Draw all candidate paths (light gray)
        for slot, path, _ in all_paths:
            for i in range(len(path) - 1):
                pygame.draw.line(screen, (200, 200, 200), path[i], path[i + 1], 2)  # Light gray paths

        # Highlight the chosen path (green) and target slot (yellow)
        if target_slot:
            pygame.draw.rect(screen, YELLOW, target_slot, 4)  # Highlight the target slot
            for i in range(len(waypoints) - 1):
                pygame.draw.line(screen, (0, 0, 0), waypoints[i], waypoints[i + 1], 4)  # Green path
        if waypoints and not car_parked:
            next_waypoint = waypoints[0]
            current_speed = car_speed
            obstacle_distance = min([obj['distance'] for obj in detected_objects]) if detected_objects else float('inf')
            obstacle_speed = next((obj.get('speed', (0, 0)) for obj in detected_objects if obj['distance'] == obstacle_distance), (0, 0))
            
            target_speed = 2  # Default target speed
            
            # Implement PID control
            if obstacle_distance <= 10:  # Object detected at 10 meters or closer
                adjusted_speed = pid_speed_control(current_speed, target_speed, obstacle_distance, math.hypot(*obstacle_speed))
                print(f"Object detected at {obstacle_distance:.2f} meters. Adjusting speed to {adjusted_speed:.2f}")
            else:
                adjusted_speed = target_speed
            
            # Use adjusted_speed for car movement
            dx = next_waypoint[0] - car.centerx
            dy = next_waypoint[1] - car.centery
            distance = math.hypot(dx, dy)
            
            if distance < 5:  # Threshold for reaching the waypoint
                waypoints.pop(0)
            else:
                car.x += adjusted_speed * dx / distance
                car.y += adjusted_speed * dy / distance
    
    # Use adjusted_speed for car movement
    # ...
        # Move the car along the waypoints
        if waypoints and not car_parked:
            car_parked = move_car_along_path(waypoints)

        # Check if the car has stopped moving
        if car_parked and not waypoints:
            print("The car has stopped moving and is parked!", flush=True)

            # Filter unique obstacles and print them
            unique_detected_objects = log_detected_obstacles(unique_detected_objects)

            print("\nFinal List of Unique Detected Obstacles:")
            for obstacle in unique_detected_objects:
                print(obstacle)

            simulation_running = False  # Stop the simulation logic, but keep the screen open

    # Draw the car
    rotated_car = pygame.transform.rotate(car_image, 0)
    rotated_rect = rotated_car.get_rect(center=car.center)
    screen.blit(rotated_car, rotated_rect.topleft)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()


while running:
    screen.fill(LIGHT_BLUE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("QUIT event detected, exiting...", flush=True)
            running = False

    # Draw parking spots, non-drivable areas, obstacles, and dynamic cars
    # ... (existing drawing code)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("QUIT event detected, exiting...", flush=True)
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
        pygame.draw.rect(screen, (128, 0, 128), obstacle["rect"])  # Purple for obstacles

    # Draw dynamic cars
    move_dynamic_cars()
    for dcar in dynamic_cars:
        pygame.draw.rect(screen, YELLOW, dcar["rect"])

    
    # Visualize LiDAR rays and detect objects
    detected_objects = lidar_sensor(car, parking_spots, obstacles, dynamic_cars, num_rays=72, max_range=300)
    unique_detected_objects.extend(detected_objects)

    for obj in detected_objects:
        pygame.draw.line(screen, (255, 255, 0), car.center, obj["location"], 1)
        pygame.draw.circle(screen, (255, 0, 0), (int(obj["location"][0]), int(obj["location"][1])), 5)

    if simulation_running:
        if target_slot is None:
            target_slot, waypoints = find_all_paths_and_choose_closest_with_lidar()
            print(f"Target slot selected: {target_slot}, Waypoints: {waypoints}", flush=True)

        # Draw paths and highlight chosen path
        # ... (existing drawing code)
        for slot, path, _ in all_paths:
            for i in range(len(path) - 1):
                pygame.draw.line(screen, (200, 200, 200), path[i], path[i + 1], 2)  # Light gray paths

        # Highlight the chosen path (green) and target slot (yellow)
        if target_slot:
            pygame.draw.rect(screen, YELLOW, target_slot, 4)  # Highlight the target slot
            for i in range(len(waypoints) - 1):
                pygame.draw.line(screen, (0, 0, 0), waypoints[i], waypoints[i + 1], 4)  # Green path


        # Move the car along the waypoints
        if waypoints and not car_parked:
            # Check for potential collisions and reroute if necessary
            car_speed = (waypoints[0][0] - car.centerx, waypoints[0][1] - car.centery)
            waypoints = check_and_reroute(car, waypoints, detected_objects, car_speed)
            car_parked = move_car_along_path(waypoints)

        # Check if the car has stopped moving
        if car_parked and not waypoints:
            print("The car has stopped moving and is parked!", flush=True)
            simulation_running = False

    # Draw the car
    rotated_car = pygame.transform.rotate(car_image, 0)
    rotated_rect = rotated_car.get_rect(center=car.center)
    screen.blit(rotated_car, rotated_rect.topleft)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()