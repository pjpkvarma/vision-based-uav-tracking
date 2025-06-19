import os
import time
import random
import math

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from collections import deque

import airsim
from detect_drone import load_yolo_model, detect_yolo
from path_generator import generate_path_vector

class DroneEnv(gym.Env):
    """
    Gym-compatible environment for active vision-based UAV tracking in AirSim
    using a hybrid YOLO + KCF detection pipeline.
    """
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.image_size = (640, 480)
        self.detection_model, self.device = load_yolo_model()
        self.detection_threshold = 0.7
        self.image_center = np.array([self.image_size[0] / 2, self.image_size[1] / 2])
        self.max_distance = np.linalg.norm(self.image_center - self.image_size)

        self.move_drone1 = None
        self.move_drone2 = None

        self.action_space = Discrete(21)
        self.observation_space = Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(45,), dtype=np.float32)
        self.state_history = deque(maxlen=5)

        self.current_frame = None
        self.using_yolo = 1

        self.no_detection_steps = 0
        self.max_no_detection_steps = 25
        self.continuous_detection_steps = 0
        self.max_steps = 500
        self.current_step = 0
        self.cummulative_reward = 0
        self.desired_distance = 5
        self.duration = 0.3

        self.metrics = {
            'total_frames': 0,
            'detection_success_frames': 0,
            'distances': [],
            'time_within_desired_distance': 0,
            'avg_alignment': 0
        }

        self.desired_distance_threshold = 7
        self.alpha1 = 1
        self.alpha2 = 0.5

        self.step_timings = {
            'image_capture': [],
            'yolo_inference': [],
            'action_execution': [],
            'reward_computation': [],
            'state_update': [],
            'total_step': []
        }

        self.reset()

    def reset(self, seed=81):
        self.client.simPause(False)
        self.client.reset()

        self.no_detection_steps = 0
        self.continuous_detection_steps = 0
        self.current_step = 0
        self.cummulative_reward = 0
        self.state_history.clear()
        self.start_time = time.time()
        self.done = False
        self.path_idx = 0

        self.path_types = ["siney"]
        path_type = random.choice(self.path_types)
        path = generate_path_vector(path_type)

        self.vx = [p[0] for p in path]
        self.vy = [p[1] for p in path]
        self.vz = [p[2] for p in path]

        self.client.enableApiControl(True, "Drone1")
        self.client.enableApiControl(True, "Drone2")
        self.client.armDisarm(True, "Drone1")
        self.client.armDisarm(True, "Drone2")

        move1 = self.client.moveToPositionAsync(0, 0, -50, 5, vehicle_name="Drone1")
        move2 = self.client.moveToPositionAsync(5, 0, -55, 5, vehicle_name="Drone2")
        move1.join()
        move2.join()

        self.metrics = {
            'total_frames': 0,
            'detection_success_frames': 0,
            'distances': [],
            'time_within_desired_distance': 0,
            'avg_alignment': 0
        }

        initial_state = self.get_initial_state()
        return initial_state, {}

    def get_initial_state(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
            ])
            if not responses or responses[0].image_data_uint8 is None:
                bboxes = [0, 0, 0, 0]
                no_detection = True
            else:
                rgb_image = responses[0]
                img1d = np.frombuffer(rgb_image.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(rgb_image.height, rgb_image.width, 3)
                bboxes, no_detection = detect_yolo(self.detection_model, img_rgb, confidence_threshold=0.5, mode='kcf')[:2]
        except:
            bboxes = [0, 0, 0, 0]
            no_detection = True

        drone1_state = self.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated
        vx = drone1_state.linear_velocity.x_val
        vy = drone1_state.linear_velocity.y_val
        vz = drone1_state.linear_velocity.z_val
        _, _, yaw = airsim.to_eularian_angles(drone1_state.orientation)
        yaw = math.degrees(yaw)

        pos1 = drone1_state.position
        pos2 = self.client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
        actual_distance = np.linalg.norm([pos2.x_val - pos1.x_val, pos2.y_val - pos1.y_val, pos2.z_val - pos1.z_val])

        current_state = np.array([*bboxes, actual_distance, vx, vy, vz, yaw], dtype=np.float32)
        self.state_history.append(current_state)

        if len(self.state_history) < 5:
            return np.array(list(self.state_history) + [current_state] * (5 - len(self.state_history))).flatten()
        else:
            return np.array(self.state_history).flatten()

    def get_state(self, bboxes, no_detection):
        drone1_state = self.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated
        vx = drone1_state.linear_velocity.x_val
        vy = drone1_state.linear_velocity.y_val
        vz = drone1_state.linear_velocity.z_val
        _, _, yaw = airsim.to_eularian_angles(drone1_state.orientation)
        yaw = math.degrees(yaw)

        pos1 = drone1_state.position
        pos2 = self.client.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
        actual_distance = np.linalg.norm([pos2.x_val - pos1.x_val, pos2.y_val - pos1.y_val, pos2.z_val - pos1.z_val])

        current_state = np.array([*bboxes, actual_distance, vx, vy, vz, yaw], dtype=np.float32)
        self.state_history.append(current_state)

        if len(self.state_history) < 5:
            return np.array(list(self.state_history) + [current_state] * (5 - len(self.state_history))).flatten()
        else:
            return np.array(self.state_history).flatten()

    def _execute_action(self, action):
        macro_actions = [
            (1, 0, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 30), (0, 0, 0, -30),
            (1, 0, 0, 30), (1, 0, 0, -30), (1, 1, -1, 0), (1, -1, -1, 0),
            (1, 1, 1, 0), (1, -1, 1, 0), (-1, 0, 0, 30), (-1, 0, 0, -30),
            (-1, 1, -1, 0), (-1, -1, -1, 0), (-1, 1, 1, 0), (-1, -1, 1, 0),
            (0, 1, -1, 0), (0, -1, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 0, 0, 0)
        ]
        vx, vy, vz, yaw_rate = macro_actions[action]
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.move_drone1 = self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, yaw_mode=yaw_mode, duration=self.duration, vehicle_name="Drone1")
        self.move_drone1.join()

    def reward(self, bbox, distance, no_detection):
        if no_detection:
            return 0, False, 0, 0, 0

        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        distance1 = np.linalg.norm(self.image_center - bbox_center)
        R_align = np.exp(-self.alpha1 * distance1 / self.max_distance)

        if distance > self.desired_distance:
            R_track = np.exp(-self.alpha2 * abs(distance - self.desired_distance))
        else:
            R_track = -1

        x = self.continuous_detection_steps / self.max_steps
        R_continuous = 1 / (np.exp(1 - 2 * x))
        total_reward = R_align + 2 * R_track + R_continuous
        return total_reward, self.current_step >= self.max_steps, R_align, R_track, R_continuous

    def update_metrics(self, bbox, distance, no_detection):
        self.metrics['total_frames'] += 1
        if not no_detection and distance < 50:
            self.metrics['detection_success_frames'] += 1
            self.metrics['distances'].append(distance)
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            distance_to_center = np.linalg.norm(self.image_center - bbox_center)
            alignment = np.exp(-self.alpha1 * distance_to_center / self.max_distance)
            self.metrics['alignment'] = self.metrics.get('alignment', 0) + alignment
        if abs(distance - self.desired_distance) <= self.desired_distance_threshold:
            self.metrics['time_within_desired_distance'] += 1

    def calculate_episode_metrics(self):
        if self.metrics['total_frames'] > 0:
            avg_distance = np.mean(self.metrics['distances'])
            distance_variance = np.var(self.metrics['distances'])
            time_within_dist_pct = (self.metrics['time_within_desired_distance'] / self.metrics['total_frames']) * 100
            detection_success = (self.metrics['detection_success_frames'] / self.metrics['total_frames']) * 100
            avg_alignment = self.metrics.get('alignment', 0) / self.metrics['detection_success_frames']
        else:
            avg_distance = 0
            distance_variance = 0
            time_within_dist_pct = 0
            detection_success = 0
            avg_alignment = 0

        return {
            'avg_distance': avg_distance,
            'distance_variance': distance_variance,
            'time_within_distance_percentage': time_within_dist_pct,
            'detection_success_rate': detection_success,
            'avg_alignment': avg_alignment
        }
