import numpy as np

def calculate_optimized_grid(field_length, field_width, sensor_radius):
    # Ensure at least one node for very small fields
    num_x = max(1, int(np.ceil(field_length / (2 * sensor_radius))))
    num_y = max(1, int(np.ceil(field_width / (2 * sensor_radius))))
    return num_x, num_y

def calculate_grid_points(field_length, field_width, sensor_radius):
    num_x, num_y = calculate_optimized_grid(field_length, field_width, sensor_radius)
    node_diameter = 2 * sensor_radius
    
    grid_points = []
    for i in range(num_x):
        for j in range(num_y):
            x = min(i * node_diameter + sensor_radius, field_length / 2 if num_x == 1 else field_length)
            y = min(j * node_diameter + sensor_radius, field_width / 2 if num_y == 1 else field_width)
            grid_points.append((x, y))
    
    # Remove duplicates in case the min function causes overlaps
    grid_points = list(set(grid_points))
    return sorted(grid_points)

def get_field_input_and_calculate_nodes():
    print("\nnode Optimization Calculation\n")
    try:
        field_length = float(input("Enter field length (in meters): "))
        field_width = float(input("Enter field width (in meters): "))
        sensor_radius = 10.0  # Fixed radius for DHT node
        # print(f"\nUsing node radius: {sensor_radius} meters")
        
        grid_points = calculate_grid_points(field_length, field_width, sensor_radius)
        
        print(f"\nOptimal node Placement:")
        print(f"- Total number of nodes needed: {len(grid_points)}")
        print("\nGrid Points (X, Y):")
        for idx, (x, y) in enumerate(grid_points):
            print(f"  node {idx + 1}: ({x:.2f}, {y:.2f})")
        
    except ValueError:
        print("\nInvalid input. Please enter numerical values.")

# Run the system
get_field_input_and_calculate_nodes()
