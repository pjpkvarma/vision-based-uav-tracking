import airsim
import numpy as np
import math
import time
import matplotlib.pyplot as plt

def generate_path_vector(path_type, duration=0.2, total_time=300):
    """
    Generates a velocity vector list based on the given motion type.

    Args:
        path_type (str): One of ['sinex', 'siney', 'sinez'] defining sine wave axis.
        duration (float): Duration between each step (seconds).
        total_time (int): Total motion duration (seconds).

    Returns:
        list: List of (vx, vy, vz) tuples representing velocity commands.
    """
    path_vector = []
    steps = int(total_time / duration)
    amplitude = np.random.normal(5, 1)
    frequency = np.random.normal(0.08, 0.05)
    forward_speed = 0.25

    for step in range(steps):
        elapsed_time = step * duration
        vx, vy, vz = forward_speed, 0, 0

        if path_type == "sinex":
            vy = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)
        elif path_type == "sinez":
            vz = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)
        elif path_type == "siney":
            vx = amplitude * math.sin(2 * math.pi * frequency * elapsed_time)

        path_vector.append((vx, vy, vz))

    return path_vector

def plot_curve(target_move, calculated_velocities, commanded_velocities, duration):
    """
    Plots the trajectory and velocity comparison between commanded and simulated.

    Args:
        target_move (list): List of target positions over time.
        calculated_velocities (list): Velocities derived from positions.
        commanded_velocities (list): Original commanded velocity vectors.
        duration (float): Time interval between each point.
    """
    target_move = np.array(target_move)
    commanded_velocities = np.array(commanded_velocities)
    calculated_velocities = np.array(calculated_velocities)
    time = np.arange(len(calculated_velocities)) * duration

    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(target_move[:, 0], target_move[:, 1], label='Target Trajectory', color='blue')
    plt.scatter(target_move[0, 0], target_move[0, 1], color='green', label='Start')
    plt.scatter(target_move[-1, 0], target_move[-1, 1], color='red', label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Trajectory of Target Drone')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(4, 1, 2)
    plt.plot(time, commanded_velocities[:len(calculated_velocities), 0], label='Commanded Vx', color='red')
    plt.plot(time, calculated_velocities[:, 0], label='Simulated Vx', color='pink')

    plt.subplot(4, 1, 3)
    plt.plot(time, commanded_velocities[:len(calculated_velocities), 1], label='Commanded Vy', color='green')
    plt.plot(time, calculated_velocities[:, 1], label='Simulated Vy', color='yellow')

    plt.subplot(4, 1, 4)
    plt.plot(time, commanded_velocities[:len(calculated_velocities), 2], label='Commanded Vz', color='blue')
    plt.plot(time, calculated_velocities[:, 2], label='Simulated Vz', color='magenta')

    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Commanded vs Simulated Velocities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sine_curve.png")
    plt.show()

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.enableApiControl(True, "Drone1")
    client.enableApiControl(True, "Drone2")
    client.armDisarm(True, "Drone1")
    client.armDisarm(True, "Drone2")
    client.moveToPositionAsync(0, 0, -30, 5, vehicle_name="Drone1").join()
    client.moveToPositionAsync(5, 0, -30, 5, vehicle_name="Drone2").join()

    target_move = []
    chaser_move = []
    commanded_velocities = []
    calculated_velocities = []

    duration = 0.2
    for vx, vy, vz in generate_path_vector("sinex", duration):
        if not client.isApiControlEnabled(vehicle_name="Drone1"):
            client.enableApiControl(True, "Drone1")
        if not client.isApiControlEnabled(vehicle_name="Drone2"):
            client.enableApiControl(True, "Drone2")

        commanded_velocities.append([vx, vy, vz])
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration, vehicle_name="Drone2").join()
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration, vehicle_name="Drone1").join()

        pos2 = client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
        pos1 = client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
        target_move.append([pos2.x_val, pos2.y_val, pos2.z_val])
        chaser_move.append([pos1.x_val, pos1.y_val, pos1.z_val])

    target_move = np.array(target_move)
    chaser_move = np.array(chaser_move)
    commanded_velocities = np.array(commanded_velocities)

    for i in range(1, len(target_move)):
        delta = (target_move[i] - target_move[i - 1]) / duration
        calculated_velocities.append(delta)

    calculated_velocities = np.array(calculated_velocities)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(target_move[:, 0], target_move[:, 1], label='Target', color='blue')
    plt.plot(chaser_move[:, 0], chaser_move[:, 1], label='Chaser', color='orange')
    plt.scatter(target_move[0, 0], target_move[0, 1], color='green', label='Target Start')
    plt.scatter(target_move[-1, 0], target_move[-1, 1], color='red', label='Target End')
    plt.scatter(chaser_move[0, 0], chaser_move[0, 1], color='purple', label='Chaser Start')
    plt.scatter(chaser_move[-1, 0], chaser_move[-1, 1], color='brown', label='Chaser End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Trajectory: Target vs Chaser')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2, 1, 2)
    time = np.arange(len(calculated_velocities)) * duration
    plt.plot(time, commanded_velocities[1:, 0] - calculated_velocities[:, 0], label='X Velocity Error', color='red')
    plt.plot(time, commanded_velocities[1:, 1] - calculated_velocities[:, 1], label='Y Velocity Error', color='green')
    plt.plot(time, commanded_velocities[1:, 2] - calculated_velocities[:, 2], label='Z Velocity Error', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.title('Commanded vs Simulated Velocity Errors')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("trajectory_and_velocity_difference.png")
    plt.show()
