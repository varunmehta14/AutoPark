import pygame
import math
import heapq
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Parking with LiDAR Sensors and Dynamic Pathfinding")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # Non-drivable areas
GREEN = (0, 255, 0)  # Empty slots
RED = (255, 0, 0)  # Occupied slots
YELLOW = (255, 255, 0)  # Target slot
LIGHT_BLUE = (173, 216, 230)  # Drivable areas
LIGHT_GRAY = (211, 211, 211)  # Path visualization

# Parking lot parameters
PARKING_SPOT_WIDTH = 50
PARKING_SPOT_HEIGHT = 80
PARKING_START_X = 200
PARKING_START_Y = 100
PARKING_SPACING_X = 10
PARKING_SPACING_Y = 90  # Vertical spacing

# Generate random parked cars
NUM_SPOTS_PER_ROW = 10
NUM_ROWS = 4
NUM_RANDOM_PARKED_CARS = 20

# Load car images
parked_car_image = pygame.image.load("sally.png")
parked_car_image = pygame.transform.scale(parked_car_image, (40, 70))
main_car_image = pygame.image.load("mcqueen.png")
main_car_width, main_car_height = 30, 60
main_car_image = pygame.transform.scale(main_car_image, (main_car_width, main_car_height))

# Initialize cars and slots
car_speed = 2
num_cars_to_park = 1
cars_to_park = [{"rect": pygame.Rect(WIDTH // 2 + i * 40, HEIGHT - 100, main_car_width, main_car_height),
                 "waypoints": [],
                 "target_slot": None,
                 "parked": False} for i in range(num_cars_to_park)]


def generate_random_car_positions(num_cars, width, height):
    parked_cars = []
    while len(parked_cars) < num_cars:
        car_x = random.randint(100, WIDTH - 100)
        car_y = random.randint(100, HEIGHT - 100)
        new_car = pygame.Rect(car_x, car_y, width, height)
        if not any(new_car.colliderect(car) for car in parked_cars):
            parked_cars.append(new_car)
    return parked_cars


random_parked_cars = generate_random_car_positions(NUM_RANDOM_PARKED_CARS, 40, 70)

# Parking slots
parking_spots = []
for row in range(NUM_ROWS):
    y = PARKING_START_Y + row * (PARKING_SPOT_HEIGHT + PARKING_SPACING_Y)
    for col in range(NUM_SPOTS_PER_ROW):
        x = PARKING_START_X + col * (PARKING_SPOT_WIDTH + PARKING_SPACING_X)
        rect = pygame.Rect(x, y, PARKING_SPOT_WIDTH, PARKING_SPOT_HEIGHT)
        parking_spots.append({"rect": rect, "occupied": False})


def lidar_scan(slot):
    """Simulates a LiDAR scan for a given parking slot."""
    slot_rect = slot["rect"]
    for car in random_parked_cars + [c["rect"] for c in cars_to_park if not c["parked"]]:
        if slot_rect.colliderect(car):
            return True  # Slot is occupied
    return False  # Slot is empty


def update_parking_slot_occupancy():
    """Updates parking slot occupancy using LiDAR scan."""
    for spot in parking_spots:
        spot["occupied"] = lidar_scan(spot)


def dynamic_a_star(start, goal, grid_size=10):
    """A* pathfinding considering dynamic obstacles."""
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
                       (-grid_size, -grid_size), (-grid_size, grid_size),
                       (grid_size, -grid_size), (grid_size, grid_size)]:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_rect = pygame.Rect(neighbor[0], neighbor[1], grid_size, grid_size)

            # Check collisions with static obstacles
            collision = any(neighbor_rect.colliderect(spot["rect"]) for spot in parking_spots if spot["occupied"])

            # Check collisions with dynamic cars
            collision = collision or any(neighbor_rect.colliderect(car["rect"]) for car in cars_to_park if not car["parked"])

            if collision or neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current

    return []  # No path found


def find_path_for_car(car):
    """Assigns the closest empty slot to the car and calculates the path."""
    car_center = car['rect'].center
    available_spots = [spot for spot in parking_spots if not spot['occupied']]
    
    # Sort slots by proximity to the car
    available_spots.sort(key=lambda spot: math.hypot(spot['rect'].centerx - car_center[0], 
                                                     spot['rect'].centery - car_center[1]))
    
    for spot in available_spots:
        goal = spot['rect'].center
        path = dynamic_a_star(car_center, goal)
        if path:
            car['target_slot'] = spot
            car['waypoints'] = path
            spot['occupied'] = True  # Reserve the slot
            return


def move_car_along_path(car):
    """Moves the car along its path and recalculates waypoints if blocked."""
    if car["parked"]:
        return

    # Check if the target slot is still valid
    if car["target_slot"] and lidar_scan(car["target_slot"]):
        car["target_slot"] = None
        car["waypoints"] = []
        find_path_for_car(car)
        return

    if not car["waypoints"]:
        return

    next_waypoint = car["waypoints"][0]
    car_center = car["rect"].center
    dx = next_waypoint[0] - car_center[0]
    dy = next_waypoint[1] - car_center[1]
    distance = math.hypot(dx, dy)

    if distance < 5:
        car["waypoints"].pop(0)
        if not car["waypoints"]:
            car["parked"] = True
            if car["target_slot"]:
                car["target_slot"]["occupied"] = True  # Update slot occupancy
            update_parking_slot_occupancy()  # Update all parking slots
        return

    car["rect"].x += car_speed * dx / distance
    car["rect"].y += car_speed * dy / distance


# Main loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(LIGHT_BLUE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update parking slots with LiDAR simulation
    update_parking_slot_occupancy()

    # Draw parking slots
    for spot in parking_spots:
        color = RED if spot["occupied"] else GREEN
        pygame.draw.rect(screen, color, spot["rect"])
        pygame.draw.rect(screen, BLACK, spot["rect"], 2)

    # Draw random parked cars
    for car in random_parked_cars:
        screen.blit(parked_car_image, car.topleft)

    # Move main cars and find paths dynamically
    for car in cars_to_park:
        if car["target_slot"] is None:
            find_path_for_car(car)
        else:
            move_car_along_path(car)

        rotated_car = pygame.transform.rotate(main_car_image, 0)
        rotated_rect = rotated_car.get_rect(center=car["rect"].center)
        screen.blit(rotated_car, rotated_rect.topleft)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
