# static other cars and finds path to all other unoccupied slots and goes over there 

# import pygame
# import math
# import heapq
# import random

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 1200, 800
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Parking with LiDAR Sensors and Pathfinding")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)  # Non-drivable areas
# GREEN = (0, 255, 0)  # Empty slots
# RED = (255, 0, 0)  # Occupied slots
# YELLOW = (255, 255, 0)  # Target slot
# LIGHT_BLUE = (173, 216, 230)  # Drivable areas
# LIGHT_GRAY = (211, 211, 211)  # Path visualization

# # Parking lot parameters
# PARKING_SPOT_WIDTH = 50
# PARKING_SPOT_HEIGHT = 80
# PARKING_START_X = 200
# PARKING_START_Y = 100
# PARKING_SPACING_X = 10
# PARKING_SPACING_Y = 90  # Vertical spacing

# # Generate random parked cars
# NUM_SPOTS_PER_ROW = 10
# NUM_ROWS = 4
# NUM_RANDOM_PARKED_CARS = 50

# # Load car images
# parked_car_image = pygame.image.load("sally.png")
# parked_car_image = pygame.transform.scale(parked_car_image, (40, 70))
# main_car_image = pygame.image.load("mcqueen.png")
# main_car_width, main_car_height = 30, 60
# main_car_image = pygame.transform.scale(main_car_image, (main_car_width, main_car_height))

# # Initialize cars and slots
# main_car = pygame.Rect(WIDTH // 2, HEIGHT - 100, main_car_width, main_car_height)
# car_speed = 2

# def generate_random_car_positions(num_cars, width, height):
#     parked_cars = []
#     while len(parked_cars) < num_cars:
#         car_x = random.randint(100, WIDTH - 100)
#         car_y = random.randint(100, HEIGHT - 100)
#         new_car = pygame.Rect(car_x, car_y, width, height)
#         if not any(new_car.colliderect(car) for car in parked_cars):
#             parked_cars.append(new_car)
#     return parked_cars

# random_parked_cars = generate_random_car_positions(NUM_RANDOM_PARKED_CARS, 40, 70)

# # Parking slots
# parking_spots = []
# for row in range(NUM_ROWS):
#     y = PARKING_START_Y + row * (PARKING_SPOT_HEIGHT + PARKING_SPACING_Y)
#     for col in range(NUM_SPOTS_PER_ROW):
#         x = PARKING_START_X + col * (PARKING_SPOT_WIDTH + PARKING_SPACING_X)
#         rect = pygame.Rect(x, y, PARKING_SPOT_WIDTH, PARKING_SPOT_HEIGHT)
#         parking_spots.append({"rect": rect, "occupied": False})

# def a_star(start, goal, grid_size=10):
#     def heuristic(pos1, pos2):
#         return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     g_score = {start: 0}
#     came_from = {}
#     closed_set = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if current == goal:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]

#         closed_set.add(current)

#         for dx, dy in [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size),
#                        (-grid_size, -grid_size), (-grid_size, grid_size),
#                        (grid_size, -grid_size), (grid_size, grid_size)]:
#             neighbor = (current[0] + dx, current[1] + dy)
#             neighbor_rect = pygame.Rect(neighbor[0], neighbor[1], grid_size, grid_size)

#             collision = False
#             for spot in parking_spots:
#                 if spot["occupied"] and neighbor_rect.colliderect(spot["rect"]):
#                     collision = True
#                     break
#             for car in random_parked_cars:
#                 if neighbor_rect.colliderect(car):
#                     collision = True
#                     break

#             if collision or neighbor in closed_set:
#                 continue

#             tentative_g_score = g_score[current] + heuristic(current, neighbor)
#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 g_score[neighbor] = tentative_g_score
#                 f_score = tentative_g_score + heuristic(neighbor, goal)
#                 heapq.heappush(open_set, (f_score, neighbor))
#                 came_from[neighbor] = current

#     return []

# def update_parking_slot_occupancy():
#     for spot in parking_spots:
#         spot_rect = spot["rect"]
#         spot["occupied"] = any(spot_rect.colliderect(car) for car in random_parked_cars)

# def find_all_paths_and_choose_closest():
#     car_center = main_car.center
#     all_paths = []
#     min_distance = float("inf")
#     closest_path = None
#     closest_slot = None

#     for spot in parking_spots:
#         if not spot["occupied"]:
#             goal = spot["rect"].center
#             path = a_star(car_center, goal)
#             if path:
#                 path_cost = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
#                                 for i in range(len(path) - 1))
#                 all_paths.append((spot["rect"], path, path_cost))

#                 if path_cost < min_distance:
#                     min_distance = path_cost
#                     closest_path = path
#                     closest_slot = spot["rect"]

#     return closest_slot, closest_path, all_paths

# def move_car_along_path(waypoints):
#     global main_car
#     if not waypoints:
#         return True

#     next_waypoint = waypoints[0]
#     car_center = main_car.center
#     dx = next_waypoint[0] - car_center[0]
#     dy = next_waypoint[1] - car_center[1]
#     distance = math.hypot(dx, dy)

#     if distance < 5:
#         waypoints.pop(0)
#         return False

#     main_car.x += car_speed * dx / distance
#     main_car.y += car_speed * dy / distance
#     return False

# # Main loop
# running = True
# clock = pygame.time.Clock()
# car_parked = False
# waypoints = []
# all_paths = []
# target_slot = None

# while running:
#     screen.fill(LIGHT_BLUE)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     update_parking_slot_occupancy()

#     for spot in parking_spots:
#         color = RED if spot["occupied"] else GREEN
#         pygame.draw.rect(screen, color, spot["rect"])
#         pygame.draw.rect(screen, BLACK, spot["rect"], 2)

#     for car in random_parked_cars:
#         screen.blit(parked_car_image, car.topleft)

#     if target_slot is None:
#         target_slot, waypoints, all_paths = find_all_paths_and_choose_closest()

#     for slot, path, _ in all_paths:
#         for i in range(len(path) - 1):
#             pygame.draw.line(screen, LIGHT_GRAY, path[i], path[i + 1], 2)

#     if waypoints:
#         for i in range(len(waypoints) - 1):
#             pygame.draw.line(screen, GREEN, waypoints[i], waypoints[i + 1], 3)

#     if target_slot:
#         pygame.draw.rect(screen, YELLOW, target_slot, 4)

#     if not car_parked:
#         car_parked = move_car_along_path(waypoints)

#     rotated_car = pygame.transform.rotate(main_car_image, 0)
#     rotated_rect = rotated_car.get_rect(center=main_car.center)
#     screen.blit(rotated_car, rotated_rect.topleft)

#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()



#code for dynamic other cars
import pygame
import math
import heapq
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamic Parking with LiDAR and Pathfinding")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GRAY = (211, 211, 211)

# Parking lot parameters
PARKING_SPOT_WIDTH = 50
PARKING_SPOT_HEIGHT = 80
PARKING_START_X = 200
PARKING_START_Y = 100
PARKING_SPACING_X = 10
PARKING_SPACING_Y = 90

# Initialize car images
parked_car_image = pygame.image.load("sally.png")
parked_car_image = pygame.transform.scale(parked_car_image, (40, 70))
main_car_image = pygame.image.load("mcqueen.png")
main_car_width, main_car_height = 30, 60
main_car_image = pygame.transform.scale(main_car_image, (main_car_width, main_car_height))

# Initialize main car
main_car = pygame.Rect(WIDTH // 2, HEIGHT - 100, main_car_width, main_car_height)
car_speed = 2

# Parking slots
NUM_SPOTS_PER_ROW = 10
NUM_ROWS = 4
parking_spots = []
for row in range(NUM_ROWS):
    y = PARKING_START_Y + row * (PARKING_SPOT_HEIGHT + PARKING_SPACING_Y)
    for col in range(NUM_SPOTS_PER_ROW):
        x = PARKING_START_X + col * (PARKING_SPOT_WIDTH + PARKING_SPACING_X)
        rect = pygame.Rect(x, y, PARKING_SPOT_WIDTH, PARKING_SPOT_HEIGHT)
        parking_spots.append({"rect": rect, "occupied": False})

# Generate random parked cars
NUM_RANDOM_PARKED_CARS = 20
random_parked_cars = []
for _ in range(NUM_RANDOM_PARKED_CARS):
    car_x = random.randint(100, WIDTH - 100)
    car_y = random.randint(100, HEIGHT - 100)
    random_parked_cars.append({
        "rect": pygame.Rect(car_x, car_y, 40, 70),
        "speed": [random.choice([-1, 1]), random.choice([-1, 1])],
    })

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

        for dx, dy in [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in closed_set:
                continue
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current
    return []

def update_parking_slot_occupancy():
    for spot in parking_spots:
        spot["occupied"] = any(car["rect"].colliderect(spot["rect"]) for car in random_parked_cars)

def find_closest_parking_slot():
    car_center = main_car.center
    closest_path = None
    closest_slot = None
    min_distance = float('inf')

    for spot in parking_spots:
        if not spot["occupied"]:
            goal = spot["rect"].center
            path = a_star(car_center, goal)
            if path:
                distance = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
                if distance < min_distance:
                    min_distance = distance
                    closest_path = path
                    closest_slot = spot["rect"]
    return closest_slot, closest_path

def move_car_along_path(waypoints):
    global main_car
    if not waypoints:
        return True

    next_waypoint = waypoints[0]
    dx = next_waypoint[0] - main_car.centerx
    dy = next_waypoint[1] - main_car.centery
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance < 5:
        waypoints.pop(0)
        return False

    main_car.x += car_speed * dx / distance
    main_car.y += car_speed * dy / distance
    return False

def move_random_cars():
    for car in random_parked_cars:
        car["rect"].x += car["speed"][0]
        car["rect"].y += car["speed"][1]

        # Check for collisions with walls
        if car["rect"].left <= 0 or car["rect"].right >= WIDTH:
            car["speed"][0] *= -1
        if car["rect"].top <= 0 or car["rect"].bottom >= HEIGHT:
            car["speed"][1] *= -1

        # Avoid collisions with other parked cars
        for other_car in random_parked_cars:
            if car != other_car and car["rect"].colliderect(other_car["rect"]):
                car["speed"][0] *= -1
                car["speed"][1] *= -1

# # Function to check if any part of the path is blocked by another car
# def is_path_obstructed(waypoints):
#     """Check if any waypoint along the path is obstructed by a moving car."""
#     for wp in waypoints:
#         for car in random_parked_cars:
#             if car["rect"].collidepoint(wp):
#                 return True
#     return False

# # Main loop
# running = True
# clock = pygame.time.Clock()
# car_parked = False
# waypoints = []
# target_slot = None

# while running:
#     screen.fill(LIGHT_BLUE)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     move_random_cars()
#     update_parking_slot_occupancy()

#     # Draw parking slots
#     for spot in parking_spots:
#         color = RED if spot["occupied"] else GREEN
#         pygame.draw.rect(screen, color, spot["rect"])
#         pygame.draw.rect(screen, BLACK, spot["rect"], 2)

#     # Draw random parked cars
#     for car in random_parked_cars:
#         screen.blit(parked_car_image, car["rect"].topleft)

#     # Recompute target and path if needed
#     if target_slot is None or car_parked:
#         target_slot, waypoints = find_closest_parking_slot()

#     # Check if the path is obstructed
#     if waypoints and is_path_obstructed(waypoints):
#         # Recompute the path if obstructed
#         target_slot, waypoints = find_closest_parking_slot()

#     # Visualize the path
#     if waypoints:
#         for i in range(len(waypoints) - 1):
#             pygame.draw.line(screen, LIGHT_GRAY, waypoints[i], waypoints[i + 1], 2)

#     # Highlight the target slot
#     if target_slot:
#         pygame.draw.rect(screen, YELLOW, target_slot, 4)

#     # Move the car along the path
#     if not car_parked:
#         car_parked = move_car_along_path(waypoints)

#     # Draw the main car
#     rotated_car = pygame.transform.rotate(main_car_image, 0)
#     rotated_rect = rotated_car.get_rect(center=main_car.center)
#     screen.blit(rotated_car, rotated_rect.topleft)

#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()

def is_path_obstructed(waypoints):
    """Check if any waypoint along the path is obstructed by a moving car."""
    for wp in waypoints:
        for car in random_parked_cars:
            if car["rect"].collidepoint(wp):
                return True
    return False

def check_collision():
    """Check if the main car collides with any random parked car."""
    for car in random_parked_cars:
        if main_car.colliderect(car["rect"]):
            return True
    return False

def avoid_collision():
    """Move the main car away from the collision."""
    for car in random_parked_cars:
        if main_car.colliderect(car["rect"]):
            dx = main_car.centerx - car["rect"].centerx
            dy = main_car.centery - car["rect"].centery
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 0:
                main_car.x += (dx / distance) * 10
                main_car.y += (dy / distance) * 10

# Main loop
running = True
clock = pygame.time.Clock()
car_parked = False
waypoints = []
target_slot = None
collision_cooldown = 0

while running:
    screen.fill(LIGHT_BLUE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    move_random_cars()
    update_parking_slot_occupancy()

    # Draw parking slots
    for spot in parking_spots:
        color = RED if spot["occupied"] else GREEN
        pygame.draw.rect(screen, color, spot["rect"])
        pygame.draw.rect(screen, BLACK, spot["rect"], 2)

    # Draw random parked cars
    for car in random_parked_cars:
        screen.blit(parked_car_image, car["rect"].topleft)

    # Check for collisions
    if check_collision():
        avoid_collision()
        collision_cooldown = 60  # Set cooldown for 1 second (60 frames)
        waypoints = []  # Clear the current path
        target_slot = None  # Reset the target slot

    # Decrease collision cooldown
    if collision_cooldown > 0:
        collision_cooldown -= 1

    # Recompute target and path if needed
    if (target_slot is None or car_parked or collision_cooldown == 0) and not waypoints:
        target_slot, waypoints = find_closest_parking_slot()

    # Check if the path is obstructed
    if waypoints and is_path_obstructed(waypoints):
        # Recompute the path if obstructed
        target_slot, waypoints = find_closest_parking_slot()

    # Visualize the path
    if waypoints:
        for i in range(len(waypoints) - 1):
            pygame.draw.line(screen, LIGHT_GRAY, waypoints[i], waypoints[i + 1], 2)

    # Highlight the target slot
    if target_slot:
        pygame.draw.rect(screen, YELLOW, target_slot, 4)

    # Move the car along the path
    if not car_parked and collision_cooldown == 0:
        car_parked = move_car_along_path(waypoints)

    # Draw the main car
    rotated_car = pygame.transform.rotate(main_car_image, 0)
    rotated_rect = rotated_car.get_rect(center=main_car.center)
    screen.blit(rotated_car, rotated_rect.topleft)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()