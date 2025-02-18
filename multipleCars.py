import carla
import pygame
import time
import keyboard
import heapq  # For priority queue
import math
import random

# Initialize pygame for HUD display
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Parking Lot Simulation")
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

# Parking lot and slot variables
parking_lot_boundary = []  # Parking lot outer boundary (4 points)
slot_boundaries = []  # Parking slot boundary (4 points for one slot)
slot_dimensions = []  # Store slot dimensions (length and width)
slot_status = []  # Slot status: True if occupied, False if empty
slot_sensors = []  # Store obstacle sensors for slots
parking_slots = []  # Store all parking slot locations

# Vehicle initialization
vehicle = None
goal_slot = None
scale_factor = 0.005

class PIDController:
    def __init__(self, kp, ki, kd, max_output, min_output):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.max_output = max_output
        self.min_output = min_output
        self.prev_error = 0
        self.integral = 0

    def control(self, error):
        """PID controller to calculate the control output."""
        # Proportional term
        p = self.kp * error
        # Integral term
        self.integral += error
        i = self.ki * self.integral
        # Derivative term
        d = self.kd * (error - self.prev_error)

        # Calculate total control signal
        control_signal = p + i + d
        # Save current error for next iteration
        self.prev_error = error

        # Clamp the control signal to within the specified limits
        return max(self.min_output, min(self.max_output, control_signal))


def capture_coordinates():
    """Captures coordinates from the spectator's current location."""
    spectator = world.get_spectator()
    location = spectator.get_transform().location
    return carla.Location(location.x, location.y, location.z)

def calculate_slots():
    global parking_lot_boundary, slot_boundaries, parking_slots
    min_x = min([coord.x for coord in parking_lot_boundary])
    max_x = max([coord.x for coord in parking_lot_boundary])
    min_y = min([coord.y for coord in parking_lot_boundary])
    max_y = max([coord.y for coord in parking_lot_boundary])
    slot_length = slot_dimensions[0]
    slot_width = slot_dimensions[1]
    slots_per_row = int((max_x - min_x) // slot_length)
    rows = int((max_y - min_y) // slot_width)
    parking_slots = []
    for row in range(rows):
        for col in range(slots_per_row):
            slot_x = min_x + col * slot_length
            slot_y = min_y + row * slot_width
            parking_slots.append(carla.Location(x=slot_x, y=slot_y))
    print(f"Generated {len(parking_slots)} parking slots.")
    return parking_slots

def process_obstacle_data(slot_index, event):
    if event:
        detected_object = event.actor
        # print(f"Detected object: {detected_object.type_id}, Location: {detected_object.get_location()}")
        slot_status[slot_index] = True

def use_predefined_coordinates():
    # Predefined coordinates for outer boundary and slot boundary
    predefined_outer_boundary = [
        carla.Location( x=-29.2, y=-44, z=0),
        carla.Location( x=-24.2, y=-44, z=0),
        carla.Location(x=-24.2, y=-14, z=0),
        carla.Location(x=-29.2, y=-14, z=0)
    ] 
    predefined_slot_boundary = [
        carla.Location(x=-31.612957611083984,  y=-25.633522033691406, z=0),
        carla.Location(x=-26.612957611083984,  y=-25.633522033691406, z=0),
        carla.Location(x=-26.612957611083984,  y=-28.633522033691406, z=0),
        carla.Location(x=-31.612957611083984,  y=-28.633522033691406, z=0)
    ]
    return predefined_outer_boundary, predefined_slot_boundary

def initialize_parking_lot(use_predefined=False):
    global parking_lot_boundary, slot_boundaries, slot_dimensions, slot_status, slot_sensors

    if use_predefined:
        parking_lot_boundary, slot_boundary = use_predefined_coordinates()
        print("Using predefined coordinates for parking lot and slot boundaries.")
    else:
        print("Press SPACE 4 times to capture the parking lot boundary coordinates.")
        while len(parking_lot_boundary) < 4:
            if keyboard.is_pressed('space'):
                parking_lot_boundary.append(capture_coordinates())
                print(f"Parking Lot Boundary coordinate {len(parking_lot_boundary)} captured.")
                time.sleep(0.5)

        print("Press SPACE 4 times to capture the parking slot boundary coordinates.")
        slot_boundary = []
        while len(slot_boundary) < 4:
            if keyboard.is_pressed('space'):
                slot_boundary.append(capture_coordinates())
                print(f"Parking Slot Boundary coordinate {len(slot_boundary)} captured.")
                time.sleep(0.5)

    slot_boundaries.append(slot_boundary)
    min_x = min([coord.x for coord in slot_boundary])
    max_x = max([coord.x for coord in slot_boundary])
    min_y = min([coord.y for coord in slot_boundary])
    max_y = max([coord.y for coord in slot_boundary])
    slot_length = max_x - min_x 
    slot_width = max_y - min_y -0.18
    slot_dimensions.extend([slot_length, slot_width])
    print(f"Slot length: {slot_length}, Slot width: {slot_width}")

    parking_slots = calculate_slots()

    for idx, slot in enumerate(parking_slots):
        center_x = slot.x
        center_y = slot.y
        center_location = carla.Location(center_x, center_y, 2)
        obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
        print(obstacle_bp)
        obstacle_bp.set_attribute('sensor_tick', '0.1')
        sensor = world.spawn_actor(obstacle_bp, carla.Transform(center_location))
        sensor.listen(lambda data, idx=idx: process_obstacle_data(idx, data))
        slot_sensors.append(sensor)
        slot_status.append(False)


def visualize_parking_slots():
    for idx, slot in enumerate(parking_slots):
        # Define the midpoint of the slot and the box extent
        midpoint = carla.Location(slot.x, slot.y, 0.5)  # Adjust z for better visualization
        box_extent = carla.Vector3D(slot_dimensions[0] / 2, slot_dimensions[1] / 2, 0.1)
        
        # Create the bounding box
        bounding_box = carla.BoundingBox(midpoint, box_extent)
        
        # Set color based on occupancy
        color = carla.Color(0, 255, 0) if not slot_status[idx] else carla.Color(255, 0, 0)  # Green if unoccupied, Red if occupied
        
        # Visualize the box in the CARLA environment
        world.debug.draw_box(
            bounding_box,
            carla.Rotation(),
            thickness=0.1,
            color=color,
            # life_time=2.0  # Box will be visible for 2 seconds
        )


def render_parking_lot():
    screen.fill((255, 255, 255))
    if len(parking_lot_boundary) == 4:
        min_x = min([coord.x for coord in parking_lot_boundary])
        max_x = max([coord.x for coord in parking_lot_boundary])
        min_y = min([coord.y for coord in parking_lot_boundary])
        max_y = max([coord.y for coord in parking_lot_boundary])
        x1, y1 = min_x * scale_factor, min_y * scale_factor
        x2, y2 = max_x * scale_factor, min_y * scale_factor
        x3, y3 = max_x * scale_factor, max_y * scale_factor
        x4, y4 = min_x * scale_factor, max_y * scale_factor
        pygame.draw.polygon(screen, (0, 0, 0), [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], 3)

    if slot_boundaries:
        for idx, slot in enumerate(slot_boundaries):
            min_x = min([coord.x for coord in slot])
            max_x = max([coord.x for coord in slot])
            min_y = min([coord.y for coord in slot])
            max_y = max([coord.y for coord in slot])
            x1, y1 = min_x * scale_factor, min_y * scale_factor
            x2, y2 = max_x * scale_factor, min_y * scale_factor
            x3, y3 = max_x * scale_factor, max_y * scale_factor
            x4, y4 = min_x * scale_factor, max_y * scale_factor
            color = (0, 255, 0) if not slot_status[idx] else (255, 0, 0)
            pygame.draw.polygon(screen, color, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], 0)
    pygame.display.flip()

def calculate_distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

def find_closest_unoccupied_slot(vehicle_location):
    closest_slot = None
    min_distance = float('inf')
    
    for idx, slot in enumerate(parking_slots):
        if not slot_status[idx]:
            distance = calculate_distance(vehicle_location, slot)
            if distance < min_distance:
                min_distance = distance
                closest_slot = slot
    
    return closest_slot


# def a_star(start, goal, grid_size=1.0):
#     """
#     Implements the A* algorithm to find the shortest path in CARLA's environment.
#     Ensures the path is on roads or parking lots and avoids obstacles.
#     """
#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: calculate_distance(start, goal)}

#     def is_location_free(location):
#         """Check if a location is free of obstacles."""
#         radius = 1.0  # Collision radius
#         actors = world.get_actors()
#         # Include static obstacles, vehicles, walkers, and street barriers
#         obstacles = list(actors.filter('static.*')) + list(actors.filter('vehicle.*')) + \
#                     list(actors.filter('walker.*')) + list(actors.filter('static.prop.streetbarrier'))

#         for obj in obstacles:
#             if calculate_distance(location, obj.get_location()) < radius:
#                 print(f"Obstacle detected at {location} (near {obj.type_id})")
#                 return False
#         return True

#     def neighbors(location):
#         """Generate neighbors only on valid roads or parking lots."""
#         offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
#         potential_neighbors = [
#             carla.Location(location.x + dx, location.y + dy, location.z)
#             for dx, dy in offsets
#         ]
#         return [
#             loc for loc in potential_neighbors
#             if world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any) and is_location_free(loc)
#         ]

#     closed_set = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if calculate_distance(current, goal) < grid_size * 1.5:
#             # Path found; reconstruct path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]  # Reverse path

#         closed_set.add(current)

#         for neighbor in neighbors(current):
#             if neighbor in closed_set:
#                 continue

#             tentative_g_score = g_score[current] + calculate_distance(current, neighbor)

#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = tentative_g_score + calculate_distance(neighbor, goal)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))

#     print("No path found. Check obstacles or grid resolution.")
#     return []  # Return empty path if no path is found

#### considers rerouting 
# def a_star(start, goal, grid_size=1.0):
#     """
#     Implements the A* algorithm to find the shortest path in CARLA's environment.
#     Ensures the path is on roads or parking lots and avoids obstacles.
#     """
#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: calculate_distance(start, goal)}

#     def is_location_free(location):
#         """Check if a location is free of obstacles."""
#         radius = 1.0  # Collision radius
#         actors = world.get_actors()
#         # Include static obstacles, vehicles, walkers, and street barriers
#         obstacles = list(actors.filter('static.*')) + list(actors.filter('vehicle.*')) + \
#                     list(actors.filter('walker.*')) + list(actors.filter('static.prop.streetbarrier'))

#         for obj in obstacles:
#             # Increase radius if obstacle is detected
#             if calculate_distance(location, obj.get_location()) < radius:
#                 print(f"Obstacle detected at {location} (near {obj.type_id})")
#                 return False
#         return True

#     def neighbors(location):
#         """Generate neighbors only on valid roads or parking lots."""
#         offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
#         potential_neighbors = [
#             carla.Location(location.x + dx, location.y + dy, location.z)
#             for dx, dy in offsets
#         ]
        
#         valid_neighbors = []
#         for loc in potential_neighbors:
#             if world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any):
#                 if is_location_free(loc):  # Ensure the location is free of obstacles
#                     valid_neighbors.append(loc)
#                 else:
#                     print(f"Blocked neighbor at {loc} due to an obstacle.")
#         return valid_neighbors

#     closed_set = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         # Check if we reached the goal
#         if calculate_distance(current, goal) < grid_size * 1.5:
#             # Path found; reconstruct the path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]  # Reverse path

#         closed_set.add(current)

#         # Explore neighbors and calculate scores
#         for neighbor in neighbors(current):
#             if neighbor in closed_set:
#                 continue

#             tentative_g_score = g_score[current] + calculate_distance(current, neighbor)

#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = tentative_g_score + calculate_distance(neighbor, goal)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))

#         # If the open set is exhausted but no path is found, we are stuck
#         if not open_set:
#             print("No valid path found. The car is stuck or blocked.")
#             return []

#     print("No path found. Check obstacles or grid resolution.")
#     return []  # Return empty path if no path is found

# smoothing done but does not consider car's dimensions
def calculate_angle(current_transform, next_location):
    """
    Calculate the angle between the car's current direction (yaw) and the direction to the next location.

    Args:
    - current_transform (carla.Transform): Current transform of the vehicle (including position and rotation).
    - next_location (carla.Location): Next location to which the vehicle is moving.

    Returns:
    - float: The angle in radians between the current direction (yaw) and the direction to the next location.
    """
    # Get the direction vector from current_location to next_location
    delta_x = next_location.x - current_transform.location.x
    delta_y = next_location.y - current_transform.location.y
    
    # Calculate the angle to the next location
    target_angle = math.atan2(delta_y, delta_x)
    
    # Get the current heading of the vehicle (yaw) from its transform
    current_angle = current_transform.rotation.yaw * math.pi / 180  # Convert degrees to radians
    
    # Calculate the angle difference, ensuring it is within [-pi, pi] range
    angle_diff = target_angle - current_angle
    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    return angle_diff

    

def a_star(start, goal, grid_size=1.0, car_length=4.69, car_width=1.85, turning_radius=6):
    """
    Implements the A* algorithm to find the shortest path in CARLA's environment.
    Ensures the path is on roads or parking lots and avoids obstacles, considering the car's dimensions.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: calculate_distance(start, goal)}

    def is_location_free(location):
        """Check if a location is free of obstacles, considering the car's size and turning radius."""
        radius = 1.0  # Collision radius for obstacles
        actors = world.get_actors()
        obstacles = list(actors.filter('static.*')) + list(actors.filter('vehicle.*')) + \
                    list(actors.filter('walker.*')) + list(actors.filter('static.prop.streetbarrier'))

        # Check for obstacles considering the car's dimensions
        for obj in obstacles:
            distance = calculate_distance(location, obj.get_location())
            if distance < radius:
                print(f"Obstacle detected at {location} (near {obj.type_id})")
                return False
        return True

    def neighbors(location):
        """Generate neighbors only on valid roads or parking lots, considering the car's dimensions and turning radius."""
        offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
        potential_neighbors = [
            carla.Location(location.x + dx, location.y + dy, location.z)
            for dx, dy in offsets
        ]
        
        valid_neighbors = []
        for loc in potential_neighbors:
            # Check if the neighbor is on a valid road or parking lot
            if world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any):
                if is_location_free(loc):  # Ensure the location is free of obstacles
                    # Consider the car's turning radius: avoid sharp turns
                    if is_turn_possible(location, loc):
                        valid_neighbors.append(loc)
                    else:
                        print(f"Turn not possible at {loc} due to turning radius.")
                else:
                    print(f"Blocked neighbor at {loc} due to an obstacle.")
        return valid_neighbors

    def is_turn_possible(current_location, next_location):
        """Check if a turn is possible based on the car's turning radius."""
        # Calculate angle between current direction and the direction to the next location
        current_transform = vehicle.get_transform()  # This is a carla.Transform
        angle_to_turn = calculate_angle(current_transform, next_location)  # next_location is a carla.Location


        
        # Ensure that the turn is within the car's turning radius
        if abs(angle_to_turn) <= turning_radius:
            return True
        return False

    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if calculate_distance(current, goal) < grid_size * 1.5:
            # Path found; reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            # Smooth the path before returning it
            smoothed_path = smooth_path(path[::-1])  # Reverse path and smooth it
            return smoothed_path
            # return path[::-1]  # Reverse path

        closed_set.add(current)

        # Explore neighbors and calculate scores
        for neighbor in neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + calculate_distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + calculate_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path found. Check obstacles or grid resolution.")
    return []  # Return empty path if no path is found


def a_star(start, goal, grid_size=1.0, car_length=4.69, car_width=1.85, turning_radius=6):
    """
    A* algorithm to find the shortest path in CARLA's environment.
    Considers static barriers, vehicles, and walkers as obstacles.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: calculate_distance(start, goal)}

    def is_location_free(location):
        """Check if a location is free of obstacles, considering barriers, vehicles, and walkers."""
        radius = 1.0  # Collision radius
        actors = world.get_actors()
        obstacles = (
            list(actors.filter('vehicle.*')) +
            list(actors.filter('walker.*')) +
            list(actors.filter('static.prop.streetbarrier'))  # Include barriers
        )

        for obj in obstacles:
            if calculate_distance(location, obj.get_location()) < radius:
                print(f"Obstacle detected at {location} (near {obj.type_id})")
                return False
        return True

    def neighbors(location):
        """Generate neighbors, ensuring they are valid and free of obstacles."""
        offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
        valid_neighbors = []
        for dx, dy in offsets:
            neighbor = carla.Location(location.x + dx, location.y + dy, location.z)
            waypoint = world.get_map().get_waypoint(neighbor, project_to_road=True, lane_type=carla.LaneType.Any)
            if waypoint and is_location_free(neighbor) and is_turn_possible(location, neighbor):
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def is_turn_possible(current_location, next_location):
        """Check if a turn is possible based on the car's turning radius."""
        angle_to_turn = calculate_angle(vehicle.get_transform(), next_location)
        return abs(angle_to_turn) <= turning_radius

    closed_set = set()
    while open_set:
        _, current = heapq.heappop(open_set)

        if calculate_distance(current, goal) < grid_size * 1.5:
            # Path found
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return smooth_path(path[::-1])

        closed_set.add(current)

        for neighbor in neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + calculate_distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + calculate_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path found. Check obstacles or grid resolution.")
    return []

def a_star(start, goal, grid_size=1.0, car_length=4.69, car_width=1.85, turning_radius=6):
    """
    Further optimized A* algorithm to find the shortest path in CARLA's environment.
    Considers static barriers, vehicles, and walkers as obstacles.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: calculate_distance(start, goal)}

    # Cache actors and obstacle locations for efficiency
    actors = world.get_actors()
    obstacles = (
        list(actors.filter('vehicle.*')) +
        list(actors.filter('walker.*')) +
        list(actors.filter('static.prop.streetbarrier'))  # Include barriers
    )
    obstacle_locations = {obj.get_location(): obj for obj in obstacles}  # Cache obstacle locations in a dictionary

    def is_location_free(location):
        """Efficiently check if a location is free of obstacles."""
        radius_squared = 1.0 ** 2  # Square of the collision radius
        for obj_loc in obstacle_locations.keys():
            if (location.x - obj_loc.x) ** 2 + (location.y - obj_loc.y) ** 2 < radius_squared:
                return False
        return True

    def neighbors(location):
        """Generate neighbors, ensuring they are valid and free of obstacles."""
        offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
        valid_neighbors = []
        for dx, dy in offsets:
            neighbor = carla.Location(location.x + dx, location.y + dy, location.z)
            waypoint = world.get_map().get_waypoint(neighbor, project_to_road=True, lane_type=carla.LaneType.Any)
            if waypoint and is_location_free(neighbor) and is_turn_possible(location, neighbor):
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def is_turn_possible(current_location, next_location):
        """Check if a turn is possible based on the car's turning radius."""
        angle_to_turn = calculate_angle(vehicle.get_transform(), next_location)
        return abs(angle_to_turn) <= turning_radius

    closed_set = set()

    # Cache distance calculations to avoid redundant computations
    distance_cache = {}


    def get_cached_distance(loc1, loc2):
        """Retrieve or calculate the distance between two locations."""
        key = tuple(sorted(((loc1.x, loc1.y, loc1.z), (loc2.x, loc2.y, loc2.z))))
        if key not in distance_cache:
            distance_cache[key] = calculate_distance(loc1, loc2)
        return distance_cache[key]


    while open_set:
        _, current = heapq.heappop(open_set)

        if get_cached_distance(current, goal) < grid_size * 1.5:
            # Path found
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return smooth_path(path[::-1])

        closed_set.add(current)

        for neighbor in neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + get_cached_distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + get_cached_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path found. Check obstacles or grid resolution.")
    return []




#### considers the cars dimensions and smooth it but car doesnt move properly and does not answer
# def calculate_angle(current_transform, next_location):
#     """
#     Calculate the angle between the car's current direction (yaw) and the direction to the next location.

#     Args:
#     - current_transform (carla.Transform): Current transform of the vehicle (including position and rotation).
#     - next_location (carla.Location): Next location to which the vehicle is moving.

#     Returns:
#     - float: The angle in radians between the current direction (yaw) and the direction to the next location.
#     """
#     # Get the direction vector from current_location to next_location
#     delta_x = next_location.x - current_transform.location.x
#     delta_y = next_location.y - current_transform.location.y
    
#     # Calculate the angle to the next location
#     target_angle = math.atan2(delta_y, delta_x)
    
#     # Get the current heading of the vehicle (yaw) from its transform
#     current_angle = current_transform.rotation.yaw * math.pi / 180  # Convert degrees to radians
    
#     # Calculate the angle difference, ensuring it is within [-pi, pi] range
#     angle_diff = target_angle - current_angle
#     if angle_diff > math.pi:
#         angle_diff -= 2 * math.pi
#     elif angle_diff < -math.pi:
#         angle_diff += 2 * math.pi
    
#     return angle_diff

    

# def a_star(start, goal, grid_size=1.0, car_length=2.1, car_width=0.7, turning_radius=3):
#     """
#     Implements the A* algorithm to find the shortest path in CARLA's environment.
#     Ensures the path is on roads or parking lots and avoids obstacles, considering the car's dimensions.
#     """
#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: calculate_distance(start, goal)}

#     def is_location_free(location):
#         """
#         Check if a location is free of obstacles, considering the car's dimensions and turning radius.
#         """
#         radius = 1 # Collision radius for obstacles
#         actors = world.get_actors()
#         obstacles = list(actors.filter('static.*')) + list(actors.filter('vehicle.*')) + \
#                     list(actors.filter('walker.*')) + list(actors.filter('static.prop.streetbarrier'))

#         # Define corners of the car based on its dimensions and orientation
#         def get_car_corners(loc, yaw):
#             half_length = car_length / 2
#             half_width = car_width / 2
#             corners = [
#                 (loc.x + half_length * math.cos(yaw) - half_width * math.sin(yaw),
#                 loc.y + half_length * math.sin(yaw) + half_width * math.cos(yaw)),
#                 (loc.x + half_length * math.cos(yaw) + half_width * math.sin(yaw),
#                 loc.y + half_length * math.sin(yaw) - half_width * math.cos(yaw)),
#                 (loc.x - half_length * math.cos(yaw) - half_width * math.sin(yaw),
#                 loc.y - half_length * math.sin(yaw) + half_width * math.cos(yaw)),
#                 (loc.x - half_length * math.cos(yaw) + half_width * math.sin(yaw),
#                 loc.y - half_length * math.sin(yaw) - half_width * math.cos(yaw)),
#             ]
#             return [carla.Location(x=corner[0], y=corner[1], z=loc.z) for corner in corners]

#         # Get car's corners for the current location
#         current_yaw = vehicle.get_transform().rotation.yaw * math.pi / 180  # Convert yaw to radians
#         car_corners = get_car_corners(location, current_yaw)

#         for obj in obstacles:
#             obj_location = obj.get_location()
#             for corner in car_corners:
#                 if calculate_distance(corner, obj_location) < radius:
#                     print(f"Obstacle detected at {corner} (near {obj.type_id})")
#                     return False
#         return True


#     def neighbors(location):
#         """
#         Generate neighbors only on valid roads or parking lots, considering the car's dimensions and turning radius.
#         """
#         offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
#         potential_neighbors = [
#             carla.Location(location.x + dx, location.y + dy, location.z)
#             for dx, dy in offsets
#         ]

#         valid_neighbors = []
#         for loc in potential_neighbors:
#             # Check if the neighbor is on a valid road or parking lot
#             if world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any):
#                 # Ensure the car's full footprint is free of obstacles
#                 if is_location_free(loc):
#                     # Consider the car's turning radius: avoid sharp turns
#                     if is_turn_possible(location, loc):
#                         valid_neighbors.append(loc)
#                     else:
#                         print(f"Turn not possible at {loc} due to turning radius.")
#                 else:
#                     print(f"Blocked neighbor at {loc} due to an obstacle.")
#         return valid_neighbors


#     def is_turn_possible(current_location, next_location):
#         """
#         Check if a turn is possible based on the car's turning radius and dimensions.
#         """
#         # Calculate angle between current direction and the direction to the next location
#         current_transform = vehicle.get_transform()  # This is a carla.Transform
#         angle_to_turn = calculate_angle(current_transform, next_location)  # next_location is a carla.Location

#         # Ensure the turn is within the car's turning radius
#         if abs(angle_to_turn) > turning_radius:
#             return False

#         # Ensure car's dimensions remain valid during the turn
#         num_samples = 5  # Number of intermediate points to check during the turn
#         for t in range(1, num_samples + 1):
#             interpolated_location = carla.Location(
#                 current_location.x + (next_location.x - current_location.x) * t / num_samples,
#                 current_location.y + (next_location.y - current_location.y) * t / num_samples,
#                 current_location.z
#             )
#             if not is_location_free(interpolated_location):
#                 print(f"Turn not possible at {interpolated_location} due to obstacles.")
#                 return False
#         return True


#     closed_set = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if calculate_distance(current, goal) < grid_size * 1.5:
#             # Path found; reconstruct the path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             # Smooth the path before returning it
#             smoothed_path = smooth_path(path[::-1])  # Reverse path and smooth it
#             return smoothed_path
#             # return path[::-1]  # Reverse path

#         closed_set.add(current)

#         # Explore neighbors and calculate scores
#         for neighbor in neighbors(current):
#             if neighbor in closed_set:
#                 continue

#             tentative_g_score = g_score[current] + calculate_distance(current, neighbor)

#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = tentative_g_score + calculate_distance(neighbor, goal)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))

#     print("No path found. Check obstacles or grid resolution.")
#     return []  # Return empty path if no path is found

def smooth_path(path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
    """
    Smooth a given path using gradient descent.

    Args:
    - path (list[carla.Location]): The path to be smoothed.
    - weight_data (float): Weight for data points to retain their position.
    - weight_smooth (float): Weight for smoothing the path.
    - tolerance (float): Tolerance for stopping the smoothing.

    Returns:
    - list[carla.Location]: The smoothed path.
    """
    new_path = [carla.Location(p.x, p.y, p.z) for p in path]  # Deep copy of the path
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):  # Skip the first and last points
            for coord in ['x', 'y']:
                old_value = getattr(new_path[i], coord)
                # Apply smoothing formula
                setattr(new_path[i], coord,
                        old_value + weight_data * (getattr(path[i], coord) - old_value) +
                        weight_smooth * (getattr(new_path[i - 1], coord) + getattr(new_path[i + 1], coord) - 2 * old_value))
                change += abs(old_value - getattr(new_path[i], coord))
    return new_path

def visualize_path(path):
    if len(path) < 2:
        print("Path must contain at least two points to draw a line.")
        return

    for i in range(len(path) - 1):
        start_point = carla.Location(path[i].x, path[i].y, path[i].z + 0.5)  # Adjust z for visibility
        end_point = carla.Location(path[i + 1].x, path[i + 1].y, path[i + 1].z + 0.5)
        
        # Correctly call the debug draw_line method
        world.debug.draw_line(
            begin=start_point,                # Start point of the line
            end=end_point,                    # End point of the line
            thickness=0.1,                    # Line thickness
            color=carla.Color(0, 0, 255),     # Line color
            # life_time=5.0                     # Duration for which the line is visible
        )



def drive_to_slot(vehicle, target_location):
    """
    Navigate the vehicle to the specified parking slot using A* and stop upon reaching the target.
    Implements PID control for throttle and steering with slower driving and smoother turns.
    :param vehicle: The vehicle actor
    :param target_location: carla.Location of the target parking slot
    """
    # Initialize PID controllers for steering and speed with tuned values for smoother driving
    steering_pid = PIDController(kp=0.4, ki=0.05, kd=0.1, max_output=0.5, min_output=-0.5)  # Reduced steering sensitivity
    speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, max_output=0.3, min_output=0.0)  # Slow speed control

    vehicle_location = vehicle.get_location()
    path = a_star(vehicle_location, target_location)

    if path:
        print(f"Path found: {len(path)} steps")
        visualize_path(path)

        for waypoint in path:
            while calculate_distance(vehicle.get_location(), waypoint) > 1.0:
                current_location = vehicle.get_location()
                current_heading = vehicle.get_transform().rotation.yaw
                target_heading = math.degrees(math.atan2(waypoint.y - current_location.y, waypoint.x - current_location.x))

                # Calculate the heading error (difference between current and target heading)
                heading_error = target_heading - current_heading
                if heading_error > 180:
                    heading_error -= 360
                elif heading_error < -180:
                    heading_error += 360

                # PID control for steering with smoother adjustments
                steering_control = steering_pid.control(heading_error)

                # PID control for throttle (speed), where the speed decreases as the car approaches the target
                distance_to_target = calculate_distance(current_location, waypoint)
                speed_control = speed_pid.control(distance_to_target)

                # Cap the maximum steering angle to avoid sharp turns
                steering_control = max(-1.0, min(1.0, steering_control))

                # Create vehicle control
                control = carla.VehicleControl()
                control.throttle = speed_control  # Adjust speed based on PID output (slower)
                control.steer = steering_control  # Adjust steering based on PID output (smoother turns)

                vehicle.apply_control(control)

                # Adding a small delay to prevent the car from overreacting
                time.sleep(0.1)

        # Stop the vehicle at the final location
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        print("Parked Properly!")
    else:
        print("No path found. Check obstacles or grid resolution.")

def drive_to_slot(vehicle, target_location):
    """
    Navigate the vehicle to the specified parking slot using A* and stop upon reaching the target.
    Implements PID control for throttle and steering with slower driving and smoother turns.
    :param vehicle: The vehicle actor
    :param target_location: carla.Location of the target parking slot
    """
    # Initialize PID controllers for steering and speed with tuned values for smoother driving
    steering_pid = PIDController(kp=0.4, ki=0.05, kd=0.1, max_output=0.5, min_output=-0.5)  # Reduced steering sensitivity
    speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, max_output=0.3, min_output=0.0)  # Slow speed control

    vehicle_location = vehicle.get_location()
    path = a_star(vehicle_location, target_location)

    if path:
        print(f"Path found: {len(path)} steps")
        visualize_path(path)

        for waypoint in path:
            while calculate_distance(vehicle.get_location(), waypoint) > 1.0:
                current_location = vehicle.get_location()
                current_heading = vehicle.get_transform().rotation.yaw

                # Calculate target heading, considering waypoint offset
                delta_x = waypoint.x - current_location.x
                delta_y = waypoint.y - current_location.y
                target_heading = math.degrees(math.atan2(delta_y, delta_x))

                # Calculate the heading error (difference between current and target heading)
                heading_error = target_heading - current_heading
                if heading_error > 180:
                    heading_error -= 360
                elif heading_error < -180:
                    heading_error += 360

                # PID control for steering with smoother adjustments
                steering_control = steering_pid.control(heading_error)

                # PID control for throttle (speed), where the speed decreases as the car approaches the target
                distance_to_target = calculate_distance(current_location, waypoint)
                speed_control = speed_pid.control(distance_to_target)

                # Limit steering to avoid sharp turns
                steering_control = max(-1.0, min(1.0, steering_control))

                # Adjust throttle based on steering: slower speed for sharper turns
                if abs(steering_control) > 0.3:
                    speed_control *= 0.5  # Reduce speed for sharp turns

                # Create vehicle control
                control = carla.VehicleControl()
                control.throttle = speed_control  # Adjust speed based on PID output (slower)
                control.steer = steering_control  # Adjust steering based on PID output (smoother turns)

                # Apply control to the vehicle
                vehicle.apply_control(control)

                # Adding a small delay to prevent the car from overreacting
                time.sleep(0.05)

        # Stop the vehicle at the final location
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        print("Parked Properly!")
    else:
        print("No path found. Check obstacles or grid resolution.")


def create_zigzag_barriers(blueprint_library, world, start_x, start_y, z, zigzag_offset, barrier_length, angle, num_barriers):
    """
    Create a zigzag pattern of obstacles using street barriers.

    Args:
    - blueprint_library (carla.BlueprintLibrary): CARLA blueprint library.
    - world (carla.World): CARLA world instance.
    - start_x (float): Starting x-coordinate of the zigzag pattern.
    - start_y (float): Starting y-coordinate of the zigzag pattern.
    - z (float): Z-coordinate for barrier placement.
    - zigzag_offset (float): Distance offset for the zigzag pattern along the y-axis.
    - barrier_length (float): Approximate length of a single barrier.
    - angle (float): Angle (in degrees) for barrier orientation in the zigzag.
    - num_barriers (int): Total number of barriers to spawn in the zigzag pattern.

    Returns:
    - list[carla.Actor]: List of spawned barrier actors.
    """
    # Define the blueprint for the street barrier
    streetbarrier_bp = blueprint_library.find('static.prop.streetbarrier')

    # List to store barrier locations and spawned actors
    barriers = []

    # Generate zigzag barrier locations and spawn them
    for i in range(num_barriers):
        if i % 2 == 0:
            # Barriers on the positive y-offset
            location = carla.Transform(
                carla.Location(x=start_x + i * (barrier_length + 1), y=start_y + zigzag_offset, z=z),
                carla.Rotation(pitch=0, yaw=angle, roll=0)
            )
        else:
            # Barriers on the negative y-offset
            location = carla.Transform(
                carla.Location(x=start_x + i * (barrier_length + 1), y=start_y - zigzag_offset, z=z),
                carla.Rotation(pitch=0, yaw=-angle, roll=0)
            )
        
        # Spawn the barrier and add it to the list
        barrier = world.spawn_actor(streetbarrier_bp, location)
        barriers.append(barrier)

    return barriers

try:
    # Vehicle setup
    blueprint_library = world.get_blueprint_library()
    tesla_bp = blueprint_library.find('vehicle.kawasaki.ninja')
    start_point = carla.Transform(carla.Location(x=-50.573, y=-29, z=1.917))
    # start_point = carla.Transform(carla.Location(x=-46.24822235107422, y=-12.396227836608887, z=1.917))
    vehicle = world.spawn_actor(tesla_bp, start_point)
    vehicle.set_autopilot(False)
        # Find the 'streetbarrier' blueprint
        #Parameters for the zigzag barriers
    start_x = -38         # Starting x-coordinate
    start_y = -30         # Starting y-coordinate
    z = 0                 # Z-coordinate for placement
    zigzag_offset = 0.5     # Distance between barriers in the zigzag
    barrier_length = 4    # Approximate length of a barrier
    angle = 90            # Angle for zigzag orientation
    num_barriers = 2      # Number of barriers to create

    # Call the function to create the zigzag barriers
    barriers = create_zigzag_barriers(blueprint_library, world, start_x, start_y, z, zigzag_offset, barrier_length, angle, num_barriers)

#
    # Initialize parking lot
    use_predefined = True  # Set this to False if you want to manually input coordinates
    initialize_parking_lot(use_predefined)

    while True:
        time.sleep(2)
        print("Parking Slot Status:")
        for idx, status in enumerate(slot_status):
            status_str = "Occupied" if status else "Unoccupied"
            print(f"Slot {idx + 1}: Status - {status_str}, Location: {parking_slots[idx]}")

        # Highlight parking slots in the CARLA environment
        visualize_parking_slots()
        vehicle_location = vehicle.get_location()
        closest_slot = find_closest_unoccupied_slot(vehicle_location)

        if closest_slot:
            print(f"Driving to closest unoccupied slot at location: {closest_slot}")
            drive_to_slot(vehicle, closest_slot)
        else:
            print("No unoccupied slots available.")

        render_parking_lot()
        clock.tick(30)

finally:
    if vehicle:
        vehicle.destroy()
    for sensor in slot_sensors:
        sensor.destroy()
    pygame.quit()
    print("Simulation ended. Cleaned up.")


# # Spawn n/2 vehicles for the parking slots generated
# def spawn_vehicles(num_slots):
#     #num_vehicles = num_slots // 4  # Half the number of slots
#     num_vehicles=1
#     spawned_vehicles = []

#     print(f"Spawning {num_vehicles} vehicles for {num_slots} parking slots.")
#     blueprint_library = world.get_blueprint_library()
#     vehicle_bp = blueprint_library.filter('vehicle.*')  # Get all available vehicle blueprints

#     for _ in range(num_vehicles):
#         while True:
#             # Randomly choose an initial location near the parking lot for the vehicle
#             spawn_x = random.uniform(parking_lot_boundary[0].x - 10, parking_lot_boundary[1].x + 10)
#             spawn_y = random.uniform(parking_lot_boundary[0].y - 10, parking_lot_boundary[3].y + 10)
#             spawn_point = carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=1.917))

#             try:
#                 # Randomly select a vehicle blueprint and spawn the vehicle
#                 vehicle_bp_choice = random.choice(vehicle_bp)
#                 vehicle = world.try_spawn_actor(vehicle_bp_choice, spawn_point)
#                 if vehicle:
#                     vehicle.set_autopilot(False)
#                     spawned_vehicles.append(vehicle)
#                     print(f"Spawned vehicle {vehicle.id} at {spawn_point.location}.")
#                     break
#             except Exception as e:
#                 print(f"Failed to spawn vehicle: {e}")
#                 continue  # Try again if spawning failed

#     return spawned_vehicles


# def visualize_parking_slots():
#     """
#     Visualize parking slots in the CARLA environment with occupancy status.
#     """
#     for idx, slot in enumerate(parking_slots):
#         # Define the midpoint of the slot and the box extent
#         midpoint = carla.Location(slot.x, slot.y, 0.5)  # Adjust z for better visualization
#         box_extent = carla.Vector3D(slot_dimensions[0] / 2, slot_dimensions[1] / 2, 0.1)
        
#         # Create the bounding box
#         bounding_box = carla.BoundingBox(midpoint, box_extent)
        
#         # Set color based on occupancy
#         color = carla.Color(0, 255, 0) if not slot_status[idx] else carla.Color(255, 0, 0)  # Green if unoccupied, Red if occupied
        
#         # Visualize the box in the CARLA environment
#         world.debug.draw_box(
#             bounding_box,
#             carla.Rotation(),
#             thickness=0.1,
#             color=color,
#             life_time=0.1  # Redraw quickly to keep them persistent
#         )

# def visualize_path(path):
#     """
#     Visualize the path in the CARLA environment.
#     """
#     if path:
#         for i in range(len(path) - 1):
#             world.debug.draw_line(
#                 path[i],
#                 path[i + 1],
#                 thickness=0.1,
#                 color=carla.Color(0, 0, 255),
#                 life_time=0.1  # Redraw to keep the path visible
#             )

# def manage_vehicles(spawned_vehicles):
#     """
#     Assign each vehicle the task of finding and navigating to the nearest unoccupied parking slot.
#     """
#     for vehicle in spawned_vehicles:
#         vehicle_location = vehicle.get_location()
#         closest_slot = find_closest_unoccupied_slot(vehicle_location)

#         if closest_slot:
#             print(f"Vehicle {vehicle.id} driving to slot at location: {closest_slot}")
#             path = a_star(vehicle_location, closest_slot)

#             if path:
#                 visualize_path(path)
#                 drive_to_slot(vehicle, closest_slot)
#             else:
#                 print(f"No path found for Vehicle {vehicle.id}.")
#         else:
#             print(f"Vehicle {vehicle.id} could not find an unoccupied slot.")

# try:
#     # Vehicle setup
#     blueprint_library = world.get_blueprint_library()

#     # Initialize parking lot
#     use_predefined = True  # Set this to False if you want to manually input coordinates
#     initialize_parking_lot(use_predefined)

#     # Spawn vehicles
#     spawned_vehicles = spawn_vehicles(len(parking_slots))

#     # Main simulation loop
#     while True:
#         time.sleep(0.1)  # Reduce sleep to ensure smoother rendering
#         print("Parking Slot Status:")
#         for idx, status in enumerate(slot_status):
#             status_str = "Occupied" if status else "Unoccupied"
#             print(f"Slot {idx + 1}: Status - {status_str}, Location: {parking_slots[idx]}")

#         # Highlight parking slots and paths in the CARLA environment
#         visualize_parking_slots()
#         manage_vehicles(spawned_vehicles)

#         # Render parking lot HUD (for debugging)
#         render_parking_lot()
#         clock.tick(30)

# finally:
#     # Cleanup
#     if vehicle:
#         vehicle.destroy()
#     for vehicle in spawned_vehicles:
#         vehicle.destroy()
#     for sensor in slot_sensors:
#         sensor.destroy()
