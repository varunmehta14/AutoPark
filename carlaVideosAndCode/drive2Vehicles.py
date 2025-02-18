import carla
import pygame
import time
import keyboard
import heapq
import math
import random

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Parking Lot Simulation")
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

parking_lot_boundary = []
slot_boundaries = []
slot_dimensions = []
slot_status = []
slot_sensors = []
parking_slots = []

vehicle1 = None
vehicle2 = None
goal_slot = None
scale_factor = 0.005

class PIDController:
    def __init__(self, kp, ki, kd, max_output, min_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.prev_error = 0
        self.integral = 0

    def control(self, error):
        p = self.kp * error
        self.integral += error
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error)
        control_signal = p + i + d
        self.prev_error = error
        return max(self.min_output, min(self.max_output, control_signal))

def capture_coordinates():
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
        slot_status[slot_index] = True

def use_predefined_coordinates():
    predefined_outer_boundary = [
        carla.Location(x=-29.2, y=-44, z=0),
        carla.Location(x=-24.2, y=-44, z=0),
        carla.Location(x=-24.2, y=-14, z=0),
        carla.Location(x=-29.2, y=-14, z=0)
    ]
    predefined_slot_boundary = [
        carla.Location(x=-31.612957611083984, y=-25.633522033691406, z=0),
        carla.Location(x=-26.612957611083984, y=-25.633522033691406, z=0),
        carla.Location(x=-26.612957611083984, y=-28.633522033691406, z=0),
        carla.Location(x=-31.612957611083984, y=-28.633522033691406, z=0)
    ]
    return predefined_outer_boundary, predefined_slot_boundary

def initialize_parking_lot(use_predefined=False):
    global parking_lot_boundary, slot_boundaries, slot_dimensions, slot_status, slot_sensors
    if use_predefined:
        parking_lot_boundary, slot_boundary = use_predefined_coordinates()
        print("Using predefined coordinates for parking lot and slot boundaries.")
    else:
        # Manual coordinate capture code (omitted for brevity)
        pass

    slot_boundaries.append(slot_boundary)
    min_x = min([coord.x for coord in slot_boundary])
    max_x = max([coord.x for coord in slot_boundary])
    min_y = min([coord.y for coord in slot_boundary])
    max_y = max([coord.y for coord in slot_boundary])
    slot_length = max_x - min_x
    slot_width = max_y - min_y - 0.18
    slot_dimensions.extend([slot_length, slot_width])
    print(f"Slot length: {slot_length}, Slot width: {slot_width}")

    parking_slots = calculate_slots()
    for idx, slot in enumerate(parking_slots):
        center_x = slot.x
        center_y = slot.y
        center_location = carla.Location(center_x, center_y, 2)
        obstacle_bp = world.get_blueprint_library().find('sensor.other.obstacle')
        obstacle_bp.set_attribute('sensor_tick', '0.1')
        sensor = world.spawn_actor(obstacle_bp, carla.Transform(center_location))
        sensor.listen(lambda data, idx=idx: process_obstacle_data(idx, data))
        slot_sensors.append(sensor)
        slot_status.append(False)

def visualize_parking_slots():
    for idx, slot in enumerate(parking_slots):
        midpoint = carla.Location(slot.x, slot.y, 0.5)
        box_extent = carla.Vector3D(slot_dimensions[0] / 2, slot_dimensions[1] / 2, 0.1)
        bounding_box = carla.BoundingBox(midpoint, box_extent)
        color = carla.Color(0, 255, 0) if not slot_status[idx] else carla.Color(255, 0, 0)
        world.debug.draw_box(bounding_box, carla.Rotation(), thickness=0.1, color=color)

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

def find_two_closest_unoccupied_slots(vehicle1_location, vehicle2_location):
    available_slots = [slot for idx, slot in enumerate(parking_slots) if not slot_status[idx]]
    if len(available_slots) < 2:
        return None, None
    
    slots = sorted(available_slots, key=lambda slot: min(calculate_distance(vehicle1_location, slot), calculate_distance(vehicle2_location, slot)))
    return slots[0], slots[1]

def a_star_with_vehicle_avoidance(start, goal, other_vehicle_location, grid_size=10, car_length=4.69, car_width=1.85, turning_radius=6):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: calculate_distance(start, goal)}

    actors = world.get_actors()
    obstacles = (
        list(actors.filter('vehicle.*')) +
        list(actors.filter('walker.*')) +
        list(actors.filter('static.prop.streetbarrier'))
    )
    obstacle_locations = {obj.get_location(): obj for obj in obstacles}

    def is_location_free(location):
        radius_squared = 1.5 ** 2
        for obj_loc in obstacle_locations.keys():
            if (location.x - obj_loc.x) ** 2 + (location.y - obj_loc.y) ** 2 < radius_squared:
                return False
        if (location.x - other_vehicle_location.x) ** 2 + (location.y - other_vehicle_location.y) ** 2 < radius_squared:
            return False
        return True

    def neighbors(location):
        offsets = [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]
        valid_neighbors = []
        for dx, dy in offsets:
            neighbor = carla.Location(location.x + dx, location.y + dy, location.z)
            waypoint = world.get_map().get_waypoint(neighbor, project_to_road=True, lane_type=carla.LaneType.Any)
            if waypoint and is_location_free(neighbor) and is_turn_possible(location, neighbor):
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def calculate_angle(current_transform, next_location):
        delta_x = next_location.x - current_transform.location.x
        delta_y = next_location.y - current_transform.location.y
        target_angle = math.atan2(delta_y, delta_x)
        current_angle = current_transform.rotation.yaw * math.pi / 180
        angle_diff = target_angle - current_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return angle_diff

    def is_turn_possible(current_location, next_location):
        current_transform = vehicle1.get_transform()
        current_transform.location = current_location
        angle_to_turn = calculate_angle(current_transform, next_location)
        return abs(angle_to_turn) <= turning_radius

    closed_set = set()
    distance_cache = {}

    def get_cached_distance(loc1, loc2):
        key = tuple(sorted(((loc1.x, loc1.y, loc1.z), (loc2.x, loc2.y, loc2.z))))
        if key not in distance_cache:
            distance_cache[key] = calculate_distance(loc1, loc2)
        return distance_cache[key]

    while open_set:
        _, current = heapq.heappop(open_set)
        if get_cached_distance(current, goal) < grid_size * 1.5:
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

def smooth_path(path, weight_data=0.5, weight_smooth=0.2, tolerance=0.00001):
    new_path = [carla.Location(p.x, p.y, p.z) for p in path]
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(path) - 1):
            for coord in ['x', 'y']:
                old_value = getattr(new_path[i], coord)
                setattr(new_path[i], coord, old_value + weight_data * (getattr(path[i], coord) - old_value) + weight_smooth * (getattr(new_path[i - 1], coord) + getattr(new_path[i + 1], coord) - 2 * old_value))
                change += abs(old_value - getattr(new_path[i], coord))
    return new_path

def visualize_path(path):
    if len(path) < 2:
        print("Path must contain at least two points to draw a line.")
        return
    for i in range(len(path) - 1):
        start_point = carla.Location(path[i].x, path[i].y, path[i].z + 0.5)
        end_point = carla.Location(path[i + 1].x, path[i + 1].y, path[i + 1].z + 0.5)
        world.debug.draw_line(begin=start_point, end=end_point, thickness=0.1, color=carla.Color(0, 0, 255))

def drive_step(vehicle, waypoint):
    steering_pid = PIDController(kp=0.4, ki=0.05, kd=0.1, max_output=0.5, min_output=-0.5)
    speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, max_output=0.3, min_output=0.0)
    
    current_location = vehicle.get_location()
    current_heading = vehicle.get_transform().rotation.yaw
    target_heading = math.degrees(math.atan2(waypoint.y - current_location.y, waypoint.x - current_location.x))
    
    heading_error = target_heading - current_heading
    if heading_error > 180:
        heading_error -= 360
    elif heading_error < -180:
        heading_error += 360
    
    steering_control = steering_pid.control(heading_error)
    distance_to_target = calculate_distance(current_location, waypoint)
    speed_control = speed_pid.control(distance_to_target)
    
    steering_control = max(-1.0, min(1.0, steering_control))
    
    control = carla.VehicleControl()
    control.throttle = speed_control
    control.steer = steering_control
    vehicle.apply_control(control)
    
    time.sleep(0.1)


def drive_two_vehicles_to_slots(vehicle1, vehicle2, target_location1, target_location2):
    vehicle1_location = vehicle1.get_location()
    vehicle2_location = vehicle2.get_location()
    
    path1 = a_star_with_vehicle_avoidance(vehicle1_location, target_location1, vehicle2_location)
    path2 = a_star_with_vehicle_avoidance(vehicle2_location, target_location2, vehicle1_location)
    
    if path1 and path2:
        print(f"Paths found: Vehicle 1 - {len(path1)} steps, Vehicle 2 - {len(path2)} steps")
        visualize_path(path1)
        visualize_path(path2)
        
        # Drive both vehicles simultaneously
        for wp1, wp2 in zip(path1, path2):
            drive_step(vehicle1, wp1)
            drive_step(vehicle2, wp2)
        
        # Ensure both vehicles reach their final destinations
        while (calculate_distance(vehicle1.get_location(), target_location1) > 1.0 or
               calculate_distance(vehicle2.get_location(), target_location2) > 1.0):
            drive_step(vehicle1, target_location1)
            drive_step(vehicle2, target_location2)
        
        # Stop both vehicles
        vehicle1.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        vehicle2.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        print("Both vehicles parked properly!")
    else:
        print("No path found for one or both vehicles. Check obstacles or grid resolution.")

try:
    # Vehicle setup
    blueprint_library = world.get_blueprint_library()
    bike_bp = blueprint_library.find('vehicle.kawasaki.ninja')
    
    start_point1 = carla.Transform(carla.Location(x=-50.573, y=-29, z=1.917))
    start_point2 = carla.Transform(carla.Location(x=-50.573, y=-31, z=1.917))
    spawn_point = carla.Transform(carla.Location(x=-29.200001, y=-27.080000, z=1))
    tesla_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_2 = world.spawn_actor(tesla_bp, spawn_point)
    spawn_point = carla.Transform(carla.Location(x=-29.200001, y=-35.5, z=1))
    tesla_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_3 = world.spawn_actor(tesla_bp, spawn_point)
    vehicle1 = world.spawn_actor(bike_bp, start_point1)
    vehicle2 = world.spawn_actor(bike_bp, start_point2)
    vehicle1.set_autopilot(False)
    vehicle2.set_autopilot(False)
    
    # ... (rest of the setup code remains the same)
    use_predefined = True  # Set this to False if you want to manually input coordinates
    initialize_parking_lot(use_predefined)
    
    while True:
        time.sleep(2)
        print("Parking Slot Status:")
        for idx, status in enumerate(slot_status):
            status_str = "Occupied" if status else "Unoccupied"
            print(f"Slot {idx + 1}: Status - {status_str}, Location: {parking_slots[idx]}")
        
        visualize_parking_slots()
        
        vehicle1_location = vehicle1.get_location()
        vehicle2_location = vehicle2.get_location()
        
        closest_slot1, closest_slot2 = find_two_closest_unoccupied_slots(vehicle1_location, vehicle2_location)
        
        if closest_slot1 and closest_slot2:
            print(f"Driving vehicles to closest unoccupied slots at locations: {closest_slot1} and {closest_slot2}")
            drive_two_vehicles_to_slots(vehicle1, vehicle2, closest_slot1, closest_slot2)
        else:
            print("Not enough unoccupied slots available for both vehicles.")
        
        render_parking_lot()
        clock.tick(30)

finally:
    if vehicle1:
        vehicle1.destroy()
    if vehicle2:
        vehicle2.destroy()
    for sensor in slot_sensors:
        sensor.destroy()
    pygame.quit()
    print("Simulation ended. Cleaned up.")