# import carla
# import pygame
# import time
# import keyboard
# import math

# # Initialize pygame for HUD display
# pygame.init()
# screen = pygame.display.set_mode((800, 600))
# pygame.display.set_caption("CARLA Parking Lot Simulation")
# font = pygame.font.Font(None, 24)
# clock = pygame.time.Clock()

# # Connect to CARLA server
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)
# world = client.get_world()

# # Parking lot and slot variables
# parking_lot_boundary = []  # Parking lot outer boundary (4 points)
# slot_boundaries = []  # Parking slot boundary (4 points for one slot)
# slot_dimensions = []  # Store slot dimensions (length and width)
# slot_status = []  # Slot status: True if occupied, False if empty
# slot_sensors = []  # Store obstacle sensors for slots
# parking_slots = []  # Store all parking slot locations

# # Vehicle initialization
# vehicle = None
# goal_slot = None
# scale_factor = 0.005

# def capture_coordinates():
#     """Captures coordinates from the spectator's current location."""
#     spectator = world.get_spectator()
#     location = spectator.get_transform().location
#     return carla.Location(location.x, location.y, location.z)

# def calculate_slots():
#     global parking_lot_boundary, slot_boundaries, parking_slots
#     min_x = min([coord.x for coord in parking_lot_boundary])
#     max_x = max([coord.x for coord in parking_lot_boundary])
#     min_y = min([coord.y for coord in parking_lot_boundary])
#     max_y = max([coord.y for coord in parking_lot_boundary])
#     slot_length = slot_dimensions[0]
#     slot_width = slot_dimensions[1]
#     slots_per_row = int((max_x - min_x) // slot_length)
#     rows = int((max_y - min_y) // slot_width)
#     parking_slots = []
#     for row in range(rows):
#         for col in range(slots_per_row):
#             slot_x = min_x + col * slot_length
#             slot_y = min_y + row * slot_width
#             parking_slots.append(carla.Location(x=slot_x, y=slot_y))
#     print(f"Generated {len(parking_slots)} parking slots.")
#     return parking_slots

# def process_obstacle_data(slot_index, event):
#     if event:
#         detected_object = event.actor
#         print(f"Detected object: {detected_object.type_id}, Location: {detected_object.get_location()}")
#         slot_status[slot_index] = True

# def use_predefined_coordinates():
#     # Predefined coordinates for outer boundary and slot boundary
#     predefined_outer_boundary = [
#         carla.Location( x=-31.222957611083984, y=-45.30682373046875, z=0),
#         carla.Location( x=-26.337318420410156, y=-45.30682373046875, z=0),
#         carla.Location(x=-26.337318420410156, y=-17.275760650634766, z=0),
#         carla.Location(x=-31.222957611083984, y=-17.275760650634766, z=0)
#     ]
#     predefined_slot_boundary = [
#         carla.Location(x=-31.222957611083984,  y=-25.736154556274414, z=0),
#         carla.Location(x=-26.781436920166016,  y=-25.736154556274414, z=0),
#         carla.Location(x=-26.781436920166016,  y=-28.633522033691406, z=0),
#         carla.Location(x=-31.222957611083984,  y=-28.633522033691406, z=0)
#     ]
#     return predefined_outer_boundary, predefined_slot_boundary

# def initialize_parking_lot(use_predefined=False):
#     global parking_lot_boundary, slot_boundaries, slot_dimensions, slot_status, slot_sensors

#     if use_predefined:
#         parking_lot_boundary, slot_boundary = use_predefined_coordinates()
#         print("Using predefined coordinates for parking lot and slot boundaries.")
#     else:
#         print("Press SPACE 4 times to capture the parking lot boundary coordinates.")
#         while len(parking_lot_boundary) < 4:
#             if keyboard.is_pressed('space'):
#                 parking_lot_boundary.append(capture_coordinates())
#                 print(f"Parking Lot Boundary coordinate {len(parking_lot_boundary)} captured.")
#                 time.sleep(0.5)

#         print("Press SPACE 4 times to capture the parking slot boundary coordinates.")
#         slot_boundary = []
#         while len(slot_boundary) < 4:
#             if keyboard.is_pressed('space'):
#                 slot_boundary.append(capture_coordinates())
#                 print(f"Parking Slot Boundary coordinate {len(slot_boundary)} captured.")
#                 time.sleep(0.5)

#     slot_boundaries.append(slot_boundary)
#     min_x = min([coord.x for coord in slot_boundary])
#     max_x = max([coord.x for coord in slot_boundary])
#     min_y = min([coord.y for coord in slot_boundary])
#     max_y = max([coord.y for coord in slot_boundary])
#     slot_length = max_x - min_x
#     slot_width = max_y - min_y
#     slot_dimensions.extend([slot_length, slot_width])
#     print(f"Slot length: {slot_length}, Slot width: {slot_width}")

#     parking_slots = calculate_slots()

#     for idx, slot in enumerate(parking_slots):
#         center_x = slot.x
#         center_y = slot.y
#         center_location = carla.Location(center_x, center_y, 2)
#         obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
#         print(obstacle_bp)
#         obstacle_bp.set_attribute('sensor_tick', '0.1')
#         sensor = world.spawn_actor(obstacle_bp, carla.Transform(center_location))
#         sensor.listen(lambda data, idx=idx: process_obstacle_data(idx, data))
#         slot_sensors.append(sensor)
#         slot_status.append(False)

# # In the main execution block, you can now choose whether to use predefined coordinates or not



# # def initialize_parking_lot():
# #     global parking_lot_boundary, slot_boundaries, slot_dimensions, slot_status, slot_sensors
# #     print("Press SPACE 4 times to capture the parking lot boundary coordinates.")
# #     while len(parking_lot_boundary) < 4:
# #         if keyboard.is_pressed('space'):
# #             parking_lot_boundary.append(capture_coordinates())
# #             print(f"Parking Lot Boundary coordinate {len(parking_lot_boundary)} captured.")
# #             time.sleep(0.5)

# #     print("Press SPACE 4 times to capture the parking slot boundary coordinates.")
# #     slot_boundary = []
# #     while len(slot_boundary) < 4:
# #         if keyboard.is_pressed('space'):
# #             slot_boundary.append(capture_coordinates())
# #             print(f"Parking Slot Boundary coordinate {len(slot_boundary)} captured.")
# #             time.sleep(0.5)

# #     slot_boundaries.append(slot_boundary)
# #     min_x = min([coord.x for coord in slot_boundary])
# #     max_x = max([coord.x for coord in slot_boundary])
# #     min_y = min([coord.y for coord in slot_boundary])
# #     max_y = max([coord.y for coord in slot_boundary])
# #     slot_length = max_x - min_x
# #     slot_width = max_y - min_y
# #     slot_dimensions.extend([slot_length, slot_width])
# #     print(f"Slot length: {slot_length}, Slot width: {slot_width}")

# #     parking_slots = calculate_slots()

# #     for idx, slot in enumerate(parking_slots):
# #         center_x = slot.x
# #         center_y = slot.y
# #         center_location = carla.Location(center_x, center_y, 2)
# #         obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
# #         print(obstacle_bp)
# #         obstacle_bp.set_attribute('sensor_tick', '0.1')
# #         sensor = world.spawn_actor(obstacle_bp, carla.Transform(center_location))
# #         sensor.listen(lambda data, idx=idx: process_obstacle_data(idx, data))
# #         slot_sensors.append(sensor)
# #         slot_status.append(False)

# def visualize_parking_slots():
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
#             life_time=2.0  # Box will be visible for 2 seconds
#         )


# def render_parking_lot():
#     screen.fill((255, 255, 255))
#     if len(parking_lot_boundary) == 4:
#         min_x = min([coord.x for coord in parking_lot_boundary])
#         max_x = max([coord.x for coord in parking_lot_boundary])
#         min_y = min([coord.y for coord in parking_lot_boundary])
#         max_y = max([coord.y for coord in parking_lot_boundary])
#         x1, y1 = min_x * scale_factor, min_y * scale_factor
#         x2, y2 = max_x * scale_factor, min_y * scale_factor
#         x3, y3 = max_x * scale_factor, max_y * scale_factor
#         x4, y4 = min_x * scale_factor, max_y * scale_factor
#         pygame.draw.polygon(screen, (0, 0, 0), [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], 3)

#     if slot_boundaries:
#         for idx, slot in enumerate(slot_boundaries):
#             min_x = min([coord.x for coord in slot])
#             max_x = max([coord.x for coord in slot])
#             min_y = min([coord.y for coord in slot])
#             max_y = max([coord.y for coord in slot])
#             x1, y1 = min_x * scale_factor, min_y * scale_factor
#             x2, y2 = max_x * scale_factor, min_y * scale_factor
#             x3, y3 = max_x * scale_factor, max_y * scale_factor
#             x4, y4 = min_x * scale_factor, max_y * scale_factor
#             color = (0, 255, 0) if not slot_status[idx] else (255, 0, 0)
#             pygame.draw.polygon(screen, color, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], 0)
#     pygame.display.flip()

# def calculate_distance(loc1, loc2):
#     return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

# def find_closest_unoccupied_slot(vehicle_location):
#     closest_slot = None
#     min_distance = float('inf')
    
#     for idx, slot in enumerate(parking_slots):
#         if not slot_status[idx]:
#             distance = calculate_distance(vehicle_location, slot)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_slot = slot
    
#     return closest_slot

# def drive_to_slot(vehicle, target_location):
#     # Simple driving logic (you may need to implement more sophisticated control)
#     while calculate_distance(vehicle.get_location(), target_location) > 1.0:
#         direction = target_location - vehicle.get_location()
#         direction = direction.make_unit_vector()
#         control = carla.VehicleControl()
#         control.throttle = 0.5
#         control.steer = direction.y * 0.5  # Simple steering
#         vehicle.apply_control(control)
#         time.sleep(0.1)
    
#     # Stop the vehicle
#     vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

# # try:
# #     blueprint_library = world.get_blueprint_library()
# #     tesla_bp = blueprint_library.find('vehicle.tesla.model3')
# #     start_point = carla.Transform(carla.Location(x=-50.573, y=-32, z=1.917))
# #     vehicle = world.spawn_actor(tesla_bp, start_point)
# #     vehicle.set_autopilot(False)

# #     print("Define parking lot by moving the spectator and pressing SPACE.")
# #     initialize_parking_lot()
# try:
#     # Vehicle setup
#     blueprint_library = world.get_blueprint_library()
#     tesla_bp = blueprint_library.find('vehicle.tesla.model3')
#     start_point = carla.Transform(carla.Location(x=-50.573, y=-32, z=1.917))
#     vehicle = world.spawn_actor(tesla_bp, start_point)
#     vehicle.set_autopilot(False)

#     # Initialize parking lot
#     use_predefined = True  # Set this to False if you want to manually input coordinates
#     initialize_parking_lot(use_predefined)

#     while True:
#         time.sleep(2)
#         print("Parking Slot Status:")
#         for idx, status in enumerate(slot_status):
#             status_str = "Occupied" if status else "Unoccupied"
#             print(f"Slot {idx + 1}: Status - {status_str}, Location: {parking_slots[idx]}")

#         # Highlight parking slots in the CARLA environment
#         visualize_parking_slots()
#         vehicle_location = vehicle.get_location()
#         closest_slot = find_closest_unoccupied_slot(vehicle_location)

#         if closest_slot:
#             print(f"Driving to closest unoccupied slot at location: {closest_slot}")
#             drive_to_slot(vehicle, closest_slot)
#         else:
#             print("No unoccupied slots available.")

#         render_parking_lot()
#         clock.tick(30)

# finally:
#     if vehicle:
#         vehicle.destroy()
#     for sensor in slot_sensors:
#         sensor.destroy()
#     pygame.quit()
#     print("Simulation ended. Cleaned up.")



import carla
import pygame
import time
import keyboard
import heapq  # For priority queue
import math

# Initialize pygame for HUD display
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Parking Lot Simulation")
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
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
        print(f"Detected object: {detected_object.type_id}, Location: {detected_object.get_location()}")
        slot_status[slot_index] = True

def use_predefined_coordinates():
    # Predefined coordinates for outer boundary and slot boundary
    predefined_outer_boundary = [
        carla.Location( x=-31.222957611083984, y=-45.30682373046875, z=0),
        carla.Location( x=-26.337318420410156, y=-45.30682373046875, z=0),
        carla.Location(x=-26.337318420410156, y=-17.275760650634766, z=0),
        carla.Location(x=-31.222957611083984, y=-17.275760650634766, z=0)
    ]
    predefined_slot_boundary = [
        carla.Location(x=-31.222957611083984,  y=-25.736154556274414, z=0),
        carla.Location(x=-26.781436920166016,  y=-25.736154556274414, z=0),
        carla.Location(x=-26.781436920166016,  y=-28.633522033691406, z=0),
        carla.Location(x=-31.222957611083984,  y=-28.633522033691406, z=0)
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
    slot_width = max_y - min_y
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
            life_time=2.0  # Box will be visible for 2 seconds
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

# A* algorithm to find the shortest path
def a_star(start, goal, grid_size=1.0):
    """
    Implements the A* algorithm to find the shortest path.
    :param start: carla.Location (start point)
    :param goal: carla.Location (goal point)
    :param grid_size: Resolution of the grid
    :return: List of carla.Location for the path
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: calculate_distance(start, goal)}

    def neighbors(location):
        offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
        return [
            carla.Location(location.x + dx, location.y + dy, location.z)
            for dx, dy in offsets
        ]

    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if calculate_distance(current, goal) < grid_size:
            # Path found; reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse path

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

    return []  # Return empty path if no path is found

def visualize_path(path):
    """
    Visualize the path in the CARLA environment.
    :param path: List of carla.Location representing the path
    """
    for i in range(len(path) - 1):
        world.debug.draw_line(
            path[i], path[i + 1], thickness=0.1, color=carla.Color(0, 0, 255), life_time=5.0
        )



def drive_to_slot(vehicle, target_location):
    """
    Navigate the vehicle to the specified parking slot using A* and stop upon reaching the target.
    :param vehicle: The vehicle actor
    :param target_location: carla.Location of the target parking slot
    """
    vehicle_location = vehicle.get_location()
    path = a_star(vehicle_location, target_location)

    if path:
        print(f"Path found: {len(path)} steps")
        visualize_path(path)

        for waypoint in path:
            while calculate_distance(vehicle.get_location(), waypoint) > 1.0:
                direction = carla.Vector3D(
                    waypoint.x - vehicle.get_location().x,
                    waypoint.y - vehicle.get_location().y,
                    0
                )
                direction = direction.make_unit_vector()
                
                control = carla.VehicleControl()
                control.throttle = 0.5  # Move forward
                control.steer = direction.y * 0.5  # Simple steering
                vehicle.apply_control(control)
                
                time.sleep(0.1)

        # Stop the vehicle at the final location
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        print("Parked Properly!")
    else:
        print("No path found. Check obstacles or grid resolution.")


try:
    # Vehicle setup
    blueprint_library = world.get_blueprint_library()
    tesla_bp = blueprint_library.find('vehicle.tesla.model3')
    start_point = carla.Transform(carla.Location(x=-50.573, y=-32, z=1.917))
    # start_point = carla.Transform(carla.Location(x=-46.24822235107422, y=-12.396227836608887, z=1.917))
    vehicle = world.spawn_actor(tesla_bp, start_point)
    vehicle.set_autopilot(False)

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


### tryign dynamic a star
# import carla
# import pygame
# import time
# import keyboard
# import heapq  # For priority queue
# import math

# # Initialize pygame for HUD display
# pygame.init()
# screen = pygame.display.set_mode((800, 600))
# pygame.display.set_caption("CARLA Parking Lot Simulation")
# font = pygame.font.Font(None, 24)
# clock = pygame.time.Clock()

# # Connect to CARLA server
# client = carla.Client('localhost', 2000)
# client.set_timeout(10.0)
# world = client.get_world()
# world_map = world.get_map()

# # Parking lot and slot variables
# parking_lot_boundary = []  # Parking lot outer boundary (4 points)
# slot_boundaries = []  # Parking slot boundary (4 points for one slot)
# slot_dimensions = []  # Store slot dimensions (length and width)
# slot_status = []  # Slot status: True if occupied, False if empty
# slot_sensors = []  # Store obstacle sensors for slots
# parking_slots = []  # Store all parking slot locations

# # Vehicle initialization
# vehicle = None
# goal_slot = None
# scale_factor = 0.005

# def capture_coordinates():
#     """Captures coordinates from the spectator's current location."""
#     spectator = world.get_spectator()
#     location = spectator.get_transform().location
#     return carla.Location(location.x, location.y, location.z)

# def calculate_slots():
#     global parking_lot_boundary, slot_boundaries, parking_slots
#     min_x = min([coord.x for coord in parking_lot_boundary])
#     max_x = max([coord.x for coord in parking_lot_boundary])
#     min_y = min([coord.y for coord in parking_lot_boundary])
#     max_y = max([coord.y for coord in parking_lot_boundary])
#     slot_length = slot_dimensions[0]
#     slot_width = slot_dimensions[1]
#     slots_per_row = int((max_x - min_x) // slot_length)
#     rows = int((max_y - min_y) // slot_width)
#     parking_slots = []
#     for row in range(rows):
#         for col in range(slots_per_row):
#             slot_x = min_x + col * slot_length
#             slot_y = min_y + row * slot_width
#             parking_slots.append(carla.Location(x=slot_x, y=slot_y))
#     print(f"Generated {len(parking_slots)} parking slots.")
#     return parking_slots

# def process_obstacle_data(slot_index, event):
#     if event:
#         slot_status[slot_index] = True

# def use_predefined_coordinates():
#     predefined_outer_boundary = [
#         carla.Location(x=-31.22, y=-45.30, z=0),
#         carla.Location(x=-26.33, y=-45.30, z=0),
#         carla.Location(x=-26.33, y=-17.27, z=0),
#         carla.Location(x=-31.22, y=-17.27, z=0),
#     ]
#     predefined_slot_boundary = [
#         carla.Location(x=-31.22, y=-25.73, z=0),
#         carla.Location(x=-26.78, y=-25.73, z=0),
#         carla.Location(x=-26.78, y=-28.63, z=0),
#         carla.Location(x=-31.22, y=-28.63, z=0),
#     ]
#     return predefined_outer_boundary, predefined_slot_boundary

# def initialize_parking_lot(use_predefined=False):
#     global parking_lot_boundary, slot_boundaries, slot_dimensions, slot_status, slot_sensors

#     if use_predefined:
#         parking_lot_boundary, slot_boundary = use_predefined_coordinates()
#         print("Using predefined coordinates for parking lot and slot boundaries.")
#     else:
#         print("Capture parking lot boundary and slot boundary coordinates manually.")
#         while len(parking_lot_boundary) < 4:
#             if keyboard.is_pressed('space'):
#                 parking_lot_boundary.append(capture_coordinates())
#                 time.sleep(0.5)

#         slot_boundary = []
#         while len(slot_boundary) < 4:
#             if keyboard.is_pressed('space'):
#                 slot_boundary.append(capture_coordinates())
#                 time.sleep(0.5)

#     slot_boundaries.append(slot_boundary)
#     min_x = min([coord.x for coord in slot_boundary])
#     max_x = max([coord.x for coord in slot_boundary])
#     min_y = min([coord.y for coord in slot_boundary])
#     max_y = max([coord.y for coord in slot_boundary])
#     slot_length = max_x - min_x
#     slot_width = max_y - min_y
#     slot_dimensions.extend([slot_length, slot_width])
#     print(f"Slot length: {slot_length}, Slot width: {slot_width}")

#     parking_slots = calculate_slots()

#     for idx, slot in enumerate(parking_slots):
#         center_location = carla.Location(slot.x, slot.y, 2)
#         obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
#         obstacle_bp.set_attribute('sensor_tick', '0.1')
#         sensor = world.spawn_actor(obstacle_bp, carla.Transform(center_location))
#         sensor.listen(lambda data, idx=idx: process_obstacle_data(idx, data))
#         slot_sensors.append(sensor)
#         slot_status.append(False)



# def find_closest_unoccupied_slot(vehicle_location):
#     closest_slot = None
#     min_distance = float('inf')
#     for idx, slot in enumerate(parking_slots):
#         if not slot_status[idx]:
#             distance = calculate_distance(vehicle_location, slot)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_slot = slot
#     return closest_slot

# def calculate_distance(loc1, loc2):
#     return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)

# def dynamic_a_star(start, goal, world_map, obstacles, grid_size=1.0):
#     """
#     Implements the Dynamic A* algorithm to find the shortest path while considering dynamic obstacles.
#     :param start: carla.Location (start point)
#     :param goal: carla.Location (goal point)
#     :param world_map: The map of the world (used to find waypoints)
#     :param obstacles: List of dynamic obstacles to avoid during pathfinding
#     :param grid_size: Resolution of the grid
#     :return: List of carla.Location for the path
#     """
#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     came_from = {}
#     g_score = {start: 0}
#     f_score = {start: calculate_distance(start, goal)}

#     def neighbors(location):
#         offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
#         return [
#             carla.Location(location.x + dx, location.y + dy, location.z)
#             for dx, dy in offsets
#         ]
    
#     closed_set = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if calculate_distance(current, goal) < grid_size:
#             # Path found; reconstruct path
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]  # Reverse path

#         closed_set.add(current)

#         # Get neighbors and check for dynamic obstacles
#         for neighbor in neighbors(current):
#             # Skip if the neighbor is already in closed set or is a dynamic obstacle
#             if neighbor in closed_set or neighbor in obstacles:
#                 continue

#             tentative_g_score = g_score[current] + calculate_distance(current, neighbor)

#             if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = tentative_g_score + calculate_distance(neighbor, goal)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))

#         # Check for new obstacles dynamically and add to closed set if necessary
#         for obstacle in obstacles:
#             if calculate_distance(current, obstacle) < grid_size:
#                 closed_set.add(current)

#     return []  # Return empty path if no path is found

# def update_obstacles(vehicle, world, sensor_data, obstacle_range=10.0):
#     """
#     Updates the list of dynamic obstacles based on sensor data.
#     :param vehicle: The vehicle actor
#     :param world: The CARLA world
#     :param sensor_data: The sensor data from the obstacle sensor
#     :param obstacle_range: Range within which obstacles are considered
#     :return: List of dynamic obstacles (Locations)
#     """
#     dynamic_obstacles = []

#     for data in sensor_data:
#         if data:
#             obstacle_location = data.transform.location
#             # Check if obstacle is within range
#             if calculate_distance(vehicle.get_location(), obstacle_location) < obstacle_range:
#                 dynamic_obstacles.append(obstacle_location)

#     return dynamic_obstacles

# def visualize_path(path, world):
#     """
#     Visualizes the path on the CARLA world.
#     :param path: The path to visualize
#     :param world: The CARLA world
#     """
#     for i in range(len(path) - 1):
#         world.debug.draw_line(path[i], path[i + 1], thickness=0.1, color=carla.Color(0, 0, 255), life_time=5.0)


# def visualize_parking_slots():
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
#             life_time=2.0  # Box will be visible for 2 seconds
#         )

# def drive_to_slot(vehicle, target_location, world, sensor_data, grid_size=1.0):
#     vehicle_location = vehicle.get_location()
#     obstacles = update_obstacles(vehicle, world, sensor_data)  # Get the dynamic obstacles from sensors
#     path = dynamic_a_star(vehicle_location, target_location, world, obstacles, grid_size)

#     if path:
#         print(f"Path found with {len(path)} waypoints.")
#         visualize_path(path, world)

#         for waypoint in path:
#             while calculate_distance(vehicle.get_location(), waypoint) > 1.0:
#                 # Re-evaluate obstacles in real-time
#                 obstacles = update_obstacles(vehicle, world, sensor_data)

#                 # Get the direction to the waypoint
#                 vehicle_location = vehicle.get_location()
#                 direction = carla.Vector3D(
#                     waypoint.x - vehicle_location.x,
#                     waypoint.y - vehicle_location.y,
#                     0
#                 )
#                 direction = direction.make_unit_vector()  # Normalize to get the unit vector

#                 # Calculate steering (difference in angle)
#                 angle_to_target = math.atan2(direction.y, direction.x)
#                 current_angle = vehicle.get_transform().rotation.yaw
#                 angle_diff = angle_to_target - math.radians(current_angle)
#                 angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize angle between -pi and pi

#                 # Proportional steering
#                 steer = max(-1.0, min(1.0, angle_diff / math.pi))  # Steering between -1 and 1

#                 # Throttle control based on distance
#                 distance = calculate_distance(vehicle.get_location(), waypoint)
#                 throttle = min(1.0, distance / 5.0)  # Max throttle if close to waypoint

#                 # Apply control
#                 control = carla.VehicleControl(throttle=throttle, steer=steer)
#                 vehicle.apply_control(control)

#                 # Simulate vehicle movement with small delay
#                 time.sleep(0.1)

#                 # Recompute the path if necessary
#                 if obstacles:
#                     path = dynamic_a_star(vehicle.get_location(), target_location, world, obstacles, grid_size)
#                     if not path:
#                         print("No valid path found. Reattempting...")
#                         break

#         # Once the car reaches the destination, apply brake
#         vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
#         print("Parked successfully!")

#     else:
#         print("No valid path found. Reattempting...")


# try:
#     blueprint_library = world.get_blueprint_library()
#     vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
#     start_point = carla.Transform(carla.Location(x=-50, y=-30, z=1.5))
#     vehicle = world.spawn_actor(vehicle_bp, start_point)

#     initialize_parking_lot(use_predefined=True)

#     while True:
#         visualize_parking_slots()
#         vehicle_location = vehicle.get_location()
#         closest_slot = find_closest_unoccupied_slot(vehicle_location)
#         if closest_slot:
#             print(f"Driving to closest unoccupied slot: {closest_slot}")
#             drive_to_slot(vehicle, closest_slot, world, sensor_data)
#         else:
#             print("No unoccupied slots available.")
#         clock.tick(30)

# finally:
#     if vehicle:
#         vehicle.destroy()
#     for sensor in slot_sensors:
#         sensor.destroy()
#     pygame.quit()
#     print("Simulation ended. Cleaned up.")
