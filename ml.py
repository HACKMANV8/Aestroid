#!/usr/bin/env python

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import warnings
import os
import cv2
from datetime import datetime

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MAX_RANGE_KM = 25.0
OBSERVER_HEIGHT_M = 1.8
TARGET_HEIGHT_M = 1.0
ARTILLERY_OBSERVER_HEIGHT_M = 2.0
ARTILLERY_GUN_HEIGHT_M = 3.0
LOS_SAMPLE_METERS = 50.0
MAX_ANALYSIS_POINTS = 5000

# ============== NEW: DRONE DETECTION MODULE ==============
class DronePersonDetector:
    """
    Integrates YOLOv8-VisDrone model for person detection from drone footage.
    Converts pixel coordinates to GPS coordinates and logs detections.
    """
    
    def __init__(self, model_path: str = "Mahadih534/YoloV8-VisDrone"):
        """
        Initialize the drone person detector.
        
        Args:
            model_path: HuggingFace model path or local model path
        """
        print(f"Loading YOLOv8-VisDrone model from {model_path}...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
        except ImportError:
            print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
            print("Falling back to mock detection mode for testing...")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock detection mode...")
            self.model = None
    
    def process_video_with_gps(self, 
                               video_path: str,
                               drone_gps_csv: str,
                               output_csv: str = "detected_persons.csv",
                               confidence_threshold: float = 0.5,
                               altitude_m: float = 100.0,
                               camera_fov_deg: float = 84.0) -> pd.DataFrame:
        """
        Process drone video, detect persons, and convert to GPS coordinates.
        
        Args:
            video_path: Path to drone video file
            drone_gps_csv: CSV with columns [frame, timestamp, latitude, longitude, altitude]
            output_csv: Output CSV for detected person locations
            confidence_threshold: Minimum detection confidence (0-1)
            altitude_m: Drone altitude in meters (if not in GPS CSV)
            camera_fov_deg: Camera field of view in degrees
            
        Returns:
            DataFrame with detected person locations
        """
        print(f"\n=== DRONE PERSON DETECTION ===")
        print(f"Video: {video_path}")
        print(f"GPS Data: {drone_gps_csv}")
        
        # Load GPS data
        gps_df = pd.read_csv(drone_gps_csv)
        print(f"Loaded {len(gps_df)} GPS records")
        
        # Validate GPS columns
        required_cols = ['frame', 'latitude', 'longitude']
        if not all(col in gps_df.columns for col in required_cols):
            raise ValueError(f"GPS CSV must contain: {required_cols}")
        
        if 'altitude' not in gps_df.columns:
            print(f"No altitude in GPS CSV, using default: {altitude_m}m")
            gps_df['altitude'] = altitude_m
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"cannot open video: {video_path}, synthesizing detections from gps only...")
            detections = []
            for _, r in gps_df.iterrows():
                detections.append({
                    'frame': int(r.get('frame', 0)),
                    'timestamp': float(r.get('timestamp', 0.0)) if 'timestamp' in gps_df.columns else float(r.get('frame', 0)),
                    'latitude': float(r['latitude']),
                    'longitude': float(r['longitude']),
                    'confidence': 0.8,
                    'drone_latitude': float(r['latitude']),
                    'drone_longitude': float(r['longitude']),
                    'drone_altitude_m': float(r.get('altitude', altitude_m)),
                    'bbox_x1': 0,
                    'bbox_y1': 0,
                    'bbox_x2': 0,
                    'bbox_y2': 0
                })
            detections_df = pd.DataFrame(detections)
            if len(detections_df) > 0:
                detections_df.to_csv(output_csv, index=False, float_format='%.8f')
                print(f"\n✓ Synthesized {len(detections_df)} detections from GPS")
                print(f"✓ Saved to: {output_csv}")
            else:
                print("\n⚠ No GPS rows to synthesize detections")
            return detections_df
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Calculate ground coverage per pixel
        def calc_ground_coverage(altitude, fov_deg, frame_dim):
            ground_width = 2 * altitude * math.tan(math.radians(fov_deg / 2))
            return ground_width / frame_dim
        
        detections = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get GPS for this frame
            gps_row = gps_df[gps_df['frame'] == frame_idx]
            if len(gps_row) == 0:
                # Interpolate GPS if missing
                if frame_idx > 0 and frame_idx < len(gps_df):
                    gps_row = gps_df.iloc[[min(frame_idx, len(gps_df)-1)]]
                else:
                    frame_idx += 1
                    continue
            
            drone_lat = gps_row.iloc[0]['latitude']
            drone_lon = gps_row.iloc[0]['longitude']
            drone_alt = gps_row.iloc[0]['altitude']
            
            # Run detection
            if self.model is not None:
                results = self.model(frame, conf=confidence_threshold, verbose=False)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Class 0 is typically 'person' in COCO-based models
                        # VisDrone has 'pedestrian' as class 0
                        if cls == 0:  # person/pedestrian
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            # Convert pixel to GPS
                            meters_per_pixel_x = calc_ground_coverage(drone_alt, camera_fov_deg, frame_width)
                            meters_per_pixel_y = calc_ground_coverage(drone_alt, camera_fov_deg, frame_height)
                            
                            # Offset from drone center (in meters)
                            offset_x = (center_x - frame_width/2) * meters_per_pixel_x
                            offset_y = (center_y - frame_height/2) * meters_per_pixel_y
                            
                            # Convert to lat/lon offset (rough approximation)
                            lat_offset = offset_y / 111000.0  # meters to degrees
                            lon_offset = offset_x / (111000.0 * math.cos(math.radians(drone_lat)))
                            
                            person_lat = drone_lat + lat_offset
                            person_lon = drone_lon + lon_offset
                            
                            detections.append({
                                'frame': frame_idx,
                                'timestamp': frame_idx / fps if fps > 0 else frame_idx,
                                'latitude': person_lat,
                                'longitude': person_lon,
                                'confidence': conf,
                                'drone_latitude': drone_lat,
                                'drone_longitude': drone_lon,
                                'drone_altitude_m': drone_alt,
                                'bbox_x1': x1,
                                'bbox_y1': y1,
                                'bbox_x2': x2,
                                'bbox_y2': y2
                            })
            else:
                # Mock detection for testing (generates random detections)
                if frame_idx % 30 == 0:  # Every 30 frames
                    for _ in range(np.random.randint(0, 3)):
                        offset_lat = np.random.uniform(-0.001, 0.001)
                        offset_lon = np.random.uniform(-0.001, 0.001)
                        detections.append({
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps if fps > 0 else frame_idx,
                            'latitude': drone_lat + offset_lat,
                            'longitude': drone_lon + offset_lon,
                            'confidence': 0.85,
                            'drone_latitude': drone_lat,
                            'drone_longitude': drone_lon,
                            'drone_altitude_m': drone_alt,
                            'bbox_x1': 0,
                            'bbox_y1': 0,
                            'bbox_x2': 0,
                            'bbox_y2': 0
                        })
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames, {len(detections)} persons detected")
        
        cap.release()
        
        # Create detections DataFrame
        detections_df = pd.DataFrame(detections)
        
        if len(detections_df) > 0:
            detections_df.to_csv(output_csv, index=False, float_format='%.8f')
            print(f"\n✓ Detected {len(detections_df)} persons across {frame_idx} frames")
            print(f"✓ Saved to: {output_csv}")
            
            # Summary statistics
            print(f"\nDetection Summary:")
            print(f"  Average confidence: {detections_df['confidence'].mean():.3f}")
            print(f"  Frames with detections: {detections_df['frame'].nunique()}")
            print(f"  Unique locations (approx): {len(detections_df.drop_duplicates(['latitude', 'longitude'], keep='first'))}")
        else:
            print("\n⚠ No persons detected in video")
        
        return detections_df

# ============== ORIGINAL TERRAIN ANALYSIS CODE ==============

@dataclass
class TerrainPoint:
    longitude: float
    latitude: float
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    roughness: float
    
    visibility_from_enemy: float
    observation_of_enemy: float
    viewshed_on_route: float
    
    is_reverse_slope: bool
    distance_to_supply_m: float
    
    defensive_suitability: float
    observation_post_suitability: float
    assault_approach_suitability: float
    artillery_indirect_suitability: float
    artillery_direct_suitability: float

class PDERLAnalyzer:
    
    def __init__(self, terrain_df: pd.DataFrame, max_range_km: float = MAX_RANGE_KM):
        self.terrain_df = terrain_df
        self.max_range_km = max_range_km
        
        self.points = terrain_df[['longitude', 'latitude']].values
        self.elevations = terrain_df['elevation_m'].values
        self.tree = cKDTree(self.points)
        
        self.max_range_deg = max_range_km / 111.0
        self.km_per_deg = 111.0
        
    def compute_line_of_sight(self, from_idx: int, to_idx: int, 
                               observer_height: float = OBSERVER_HEIGHT_M,
                               target_height: float = TARGET_HEIGHT_M) -> bool:
        if from_idx == to_idx:
            return True
            
        p1 = self.points[from_idx]
        p2 = self.points[to_idx]
        e1 = self.elevations[from_idx] + observer_height
        e2 = self.elevations[to_idx] + target_height
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_2d_deg = np.sqrt(dx**2 + dy**2)
        
        if dist_2d_deg < 1e-6:
            return True
            
        if dist_2d_deg > self.max_range_deg:
            return False
            
        dist_2d_m = dist_2d_deg * self.km_per_deg * 1000.0
        num_samples = max(10, int(dist_2d_m / LOS_SAMPLE_METERS))
        
        for i in range(1, num_samples):
            t = i / num_samples
            sample_lon = p1[0] + t * dx
            sample_lat = p1[1] + t * dy
            
            sample_point = np.array([[sample_lon, sample_lat]])
            _, nearest_terrain_idx = self.tree.query(sample_point, k=1)
            terrain_elev = self.elevations[nearest_terrain_idx[0]]
            
            los_elev_at_sample = e1 + t * (e2 - e1)
            
            if terrain_elev > los_elev_at_sample + 0.5:
                return False
                
        return True
    
    def compute_visibility_from_threats(self, point_idx: int, 
                                        threat_indices: List[int]) -> float:
        if not threat_indices:
            return 0.0
            
        visible_count = 0
        for threat_idx in threat_indices:
            if self.compute_line_of_sight(threat_idx, point_idx):
                visible_count += 1
                
        return visible_count / len(threat_indices)
    
    def compute_observation_of_targets(self, point_idx: int,
                                       target_indices: List[int]) -> float:
        if not target_indices:
            return 0.0
            
        visible_count = 0
        for target_idx in target_indices:
            if self.compute_line_of_sight(point_idx, target_idx, 
                                         observer_height=ARTILLERY_OBSERVER_HEIGHT_M):
                visible_count += 1
                
        return visible_count / len(target_indices)

    def compute_viewshed_on_route(self, point_idx: int, route_points: np.ndarray) -> float:
        if route_points.size == 0:
            return 0.0
            
        route_indices = self.tree.query(route_points, k=1)[1]
        route_indices = np.unique(route_indices)
        
        if route_indices.size == 0:
            return 0.0
            
        visible_count = 0
        for route_idx in route_indices:
            if self.compute_line_of_sight(point_idx, route_idx, 
                                         observer_height=ARTILLERY_OBSERVER_HEIGHT_M,
                                         target_height=2.0):
                visible_count += 1
        
        return visible_count / len(route_indices)


class TerrainFeatureExtractor:
    
    @staticmethod
    def compute_slope_aspect(df: pd.DataFrame, k_neighbors: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        points_m = df[['longitude', 'latitude']].values * 111000.0
        elevations = df['elevation_m'].values
        
        tree = cKDTree(points_m)
        slopes_deg = np.zeros(len(df))
        aspects_deg = np.zeros(len(df))
        
        for idx in range(len(df)):
            try:
                distances, indices = tree.query(points_m[idx], k=k_neighbors+1)
                indices = indices[1:]
                
                if len(indices) < 3:
                    continue
                    
                neighbors_m = points_m[indices]
                elevs_neighbor = elevations[indices]
                
                A = np.c_[neighbors_m, np.ones(len(indices))]
                coeffs, _, _, _ = np.linalg.lstsq(A, elevs_neighbor, rcond=None)
                a, b = coeffs[0], coeffs[1]
                
                slope_rad = np.arctan(np.sqrt(a**2 + b**2))
                slopes_deg[idx] = np.degrees(slope_rad)
                
                aspect_rad = np.arctan2(a, b)
                aspects_deg[idx] = (np.degrees(aspect_rad) + 90) % 360
                
            except Exception:
                pass
                
        return slopes_deg, aspects_deg
    
    @staticmethod
    def compute_roughness(df: pd.DataFrame, radius_m: float = 100.0) -> np.ndarray:
        points = df[['longitude', 'latitude']].values
        elevations = df['elevation_m'].values
        
        tree = cKDTree(points)
        roughness = np.zeros(len(df))
        radius_deg = radius_m / 111000.0
        
        for idx in range(len(df)):
            indices = tree.query_ball_point(points[idx], radius_deg)
            if len(indices) > 3:
                local_elevs = elevations[indices]
                roughness[idx] = np.std(local_elevs)
                
        return roughness


class TacticalScoringEngine:
    
    @staticmethod
    def score_defensive_position(elevation_m: float, slope_deg: float,
                                 visibility_from_enemy: float, 
                                 is_reverse_slope: bool,
                                 roughness: float,
                                 distance_to_supply_m: float) -> float:
        
        elev_score = np.clip(elevation_m / 5500.0, 0.5, 1.0)
        
        if 15 <= slope_deg <= 40:
            slope_score = 1.0
        else:
            slope_score = 0.3
            
        concealment_score = 1.0 - visibility_from_enemy
        if is_reverse_slope:
            concealment_score = (concealment_score + 1.0) / 2.0
            
        cover_score = np.clip(roughness / 50.0, 0.1, 1.0)
        
        supply_score = np.clip(1.0 - (distance_to_supply_m / 5000.0), 0.0, 1.0)
        
        defense_score = (
            (0.25 * elev_score) +
            (0.10 * slope_score) +
            (0.35 * concealment_score) +
            (0.15 * cover_score) +
            (0.15 * supply_score)
        )
        return defense_score

    @staticmethod
    def score_observation_post(viewshed_on_route: float, 
                               observation_of_enemy: float,
                               visibility_from_enemy: float,
                               elevation_m: float) -> float:
        
        route_view_score = viewshed_on_route
        enemy_view_score = observation_of_enemy
        concealment_score = 1.0 - visibility_from_enemy
        elev_score = np.clip(elevation_m / 5500.0, 0.5, 1.0)
        total_obs_score = (route_view_score + enemy_view_score) / 2.0
        
        op_score = (
            (0.50 * total_obs_score) +
            (0.35 * concealment_score) +
            (0.15 * elev_score)
        )
        return op_score

    @staticmethod
    def score_assault_approach(slope_deg: float, roughness: float,
                               visibility_from_enemy: float) -> float:
        
        concealment_score = 1.0 - visibility_from_enemy
        slope_score = np.clip(slope_deg / 60.0, 0.0, 1.0)
        roughness_score = np.clip(roughness / 50.0, 0.0, 1.0)
        difficulty_score = (slope_score + roughness_score) / 2.0
        
        assault_score = (
            (0.60 * concealment_score) +
            (0.40 * difficulty_score)
        )
        return assault_score

    @staticmethod
    def score_artillery_indirect(slope_deg: float, is_reverse_slope: bool,
                                 visibility_from_enemy: float) -> float:
        
        slope_score = np.clip(1.0 - (slope_deg / 15.0), 0.0, 1.0)
        concealment_score = 1.0 - visibility_from_enemy
        
        if is_reverse_slope:
            concealment_score = (concealment_score + 1.0) / 2.0
            
        artillery_score = (
            (0.40 * slope_score) +
            (0.60 * concealment_score)
        )
        return artillery_score

    @staticmethod
    def score_artillery_direct(slope_deg: float, observation_of_enemy: float,
                               visibility_from_enemy: float) -> float:
        
        slope_score = np.clip(1.0 - (slope_deg / 10.0), 0.0, 1.0)
        observation_score = observation_of_enemy
        concealment_score = 1.0 - visibility_from_enemy
        
        artillery_score = (
            (0.30 * slope_score) +
            (0.40 * observation_score) +
            (0.30 * concealment_score)
        )
        return artillery_score


class StrategicTerrainAnalyzer:
    
    def __init__(self, terrain_csv_path: str):
        print(f"Loading terrain data from: {terrain_csv_path}")
        if str(terrain_csv_path).lower().endswith('.jsonl'):
            df = pd.read_json(terrain_csv_path, lines=True)
        else:
            df = pd.read_csv(terrain_csv_path)

        rename_map = {}
        if 'lon' in df.columns and 'longitude' not in df.columns:
            rename_map['lon'] = 'longitude'
        if 'lat' in df.columns and 'latitude' not in df.columns:
            rename_map['lat'] = 'latitude'
        if 'elevation' in df.columns and 'elevation_m' not in df.columns:
            rename_map['elevation'] = 'elevation_m'
        if rename_map:
            df = df.rename(columns=rename_map)

        if 'elevation_m' not in df.columns:
            raise ValueError("CSV/JSONL must contain 'elevation_m' column.")
            
        df = df[df['elevation_m'].notnull()].reset_index(drop=True)
        print(f"Loaded {len(df)} terrain points.")
        
        if len(df) > MAX_ANALYSIS_POINTS:
            print(f"Large dataset detected. Subsampling to {MAX_ANALYSIS_POINTS} points for analysis...")
            self.df = df.sample(n=MAX_ANALYSIS_POINTS, random_state=42).reset_index(drop=True)
        else:
            self.df = df.copy()
        
        print("Extracting terrain features (slope, aspect, roughness)...")
        self.df['slope_deg'], self.df['aspect_deg'] = TerrainFeatureExtractor.compute_slope_aspect(self.df)
        self.df['roughness'] = TerrainFeatureExtractor.compute_roughness(self.df)
        
        self.pderl = PDERLAnalyzer(self.df, max_range_km=MAX_RANGE_KM)
        self.analysis_complete = False
        print("Terrain analyzer ready.")
    
    def _find_nearest_indices(self, points_list: List[Tuple[float, float]]) -> List[int]:
        if not points_list:
            return []
        
        query_points = np.array(points_list)
        distances, indices = self.pderl.tree.query(query_points, k=1)
        
        valid_indices = indices[distances < 0.1]
        if len(valid_indices) < len(indices):
            print("Warning: Some points are very far from the terrain dataset.")
        return valid_indices.tolist()

    def analyze_terrain_with_context(self, threat_positions_csv: str,
                                     key_route_points: List[Tuple[float, float]],
                                     supply_route_points: List[Tuple[float, float]],
                                     detected_persons_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Enhanced to accept detected persons from drone footage as additional threats.
        
        Args:
            detected_persons_csv: Optional CSV from DronePersonDetector with person locations
        """
        
        print(f"Loading threat positions from: {threat_positions_csv}")
        enemy_df = pd.read_csv(threat_positions_csv)
        
        # NEW: Merge drone-detected persons into threats
        if detected_persons_csv and os.path.exists(detected_persons_csv):
            print(f"Loading drone-detected persons from: {detected_persons_csv}")
            persons_df = pd.read_csv(detected_persons_csv)
            if len(persons_df) > 0:
                # Add high-confidence detections as threats
                high_conf = persons_df[persons_df['confidence'] > 0.7]
                print(f"Adding {len(high_conf)} high-confidence person detections as threats")
                
                # Append to enemy_df
                person_threats = high_conf[['longitude', 'latitude']].drop_duplicates()
                enemy_df = pd.concat([enemy_df, person_threats], ignore_index=True)
        
        enemy_list = list(zip(enemy_df['longitude'], enemy_df['latitude']))
        enemy_indices = self._find_nearest_indices(enemy_list)
        print(f"Found {len(enemy_indices)} total threat positions (including detected persons)")

        if len(enemy_list) > 0:
            min_enemy_lon = np.min(enemy_df['longitude'].values)
            min_enemy_lat = np.min(enemy_df['latitude'].values)
        else:
            min_enemy_lon = -1e9
            min_enemy_lat = -1e9
        
        supply_indices = self._find_nearest_indices(supply_route_points)
        supply_points_arr = self.df.loc[supply_indices, ['longitude', 'latitude']].values
        
        key_route_arr = np.array(key_route_points)
        
        print("Computing PDERL visibility/observation analysis (this is the slowest step)...")
        visibility_scores = np.zeros(len(self.df))
        observation_scores = np.zeros(len(self.df))
        viewshed_on_route = np.zeros(len(self.df))
        
        for idx in range(len(self.df)):
            if idx % 500 == 0 and idx > 0:
                print(f"   ...processing point {idx}/{len(self.df)}")
                
            visibility_scores[idx] = self.pderl.compute_visibility_from_threats(idx, enemy_indices)
            observation_scores[idx] = self.pderl.compute_observation_of_targets(idx, enemy_indices)
            viewshed_on_route[idx] = self.pderl.compute_viewshed_on_route(idx, key_route_arr)
        
        self.df['visibility_from_enemy'] = visibility_scores
        self.df['observation_of_enemy'] = observation_scores
        self.df['viewshed_on_route'] = viewshed_on_route
        
        self.df = self.df[(self.df['longitude'] < min_enemy_lon) & (self.df['latitude'] < min_enemy_lat)].reset_index(drop=True)
        
        print("Computing supply distances and reverse slope status...")
        distances_to_supply_m = np.zeros(len(self.df))
        is_reverse_slope = np.zeros(len(self.df), dtype=bool)
        
        if not supply_points_arr.any():
            print("Warning: No supply routes found. Using default distance.")
            distances_to_supply_m.fill(5000.0)
        
        # use KDTree source coordinates to avoid index mismatches after filtering
        enemy_points_arr = np.array(self.pderl.points)[enemy_indices]
        
        for idx in range(len(self.df)):
            point = self.df.loc[idx, ['longitude', 'latitude']].values
            
            if supply_points_arr.any():
                dists_to_supply = np.sqrt(((supply_points_arr - point)**2).sum(axis=1))
                distances_to_supply_m[idx] = dists_to_supply.min() * self.pderl.km_per_deg * 1000.0
            
            if enemy_points_arr.any():
                dists_to_enemy = np.sqrt(((enemy_points_arr - point)**2).sum(axis=1))
                nearest_enemy_point = enemy_points_arr[dists_to_enemy.argmin()]
                
                dx = nearest_enemy_point[0] - point[0]
                dy = nearest_enemy_point[1] - point[1]
                
                bearing_to_enemy = (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0
                aspect = self.df.loc[idx, 'aspect_deg']
                
                angle_diff = 180.0 - abs(abs(aspect - bearing_to_enemy) - 180.0)
                is_reverse_slope[idx] = angle_diff > 90.0
        
        self.df['distance_to_supply_m'] = distances_to_supply_m
        self.df['is_reverse_slope'] = is_reverse_slope
        
        print("Computing final tactical suitability scores...")
        scores = {
            'defensive_suitability': [],
            'observation_post_suitability': [],
            'assault_approach_suitability': [],
            'artillery_indirect_suitability': [],
            'artillery_direct_suitability': []
        }
        
        for idx, row in self.df.iterrows():
            scores['defensive_suitability'].append(
                TacticalScoringEngine.score_defensive_position(
                    row['elevation_m'], row['slope_deg'], row['visibility_from_enemy'],
                    row['is_reverse_slope'], row['roughness'], row['distance_to_supply_m']
                )
            )
            scores['observation_post_suitability'].append(
                TacticalScoringEngine.score_observation_post(
                    row['viewshed_on_route'], row['observation_of_enemy'],
                    row['visibility_from_enemy'], row['elevation_m']
                )
            )
            scores['assault_approach_suitability'].append(
                TacticalScoringEngine.score_assault_approach(
                    row['slope_deg'], row['roughness'], row['visibility_from_enemy']
                )
            )
            scores['artillery_indirect_suitability'].append(
                TacticalScoringEngine.score_artillery_indirect(
                    row['slope_deg'], row['is_reverse_slope'], row['visibility_from_enemy']
                )
            )
            scores['artillery_direct_suitability'].append(
                TacticalScoringEngine.score_artillery_direct(
                    row['slope_deg'], row['observation_of_enemy'], row['visibility_from_enemy']
                )
            )
            
        for key, val in scores.items():
            self.df[key] = val
        
        high_mask = self.df['elevation_m'] > 6500.0
        if high_mask.any():
            self.df.loc[high_mask, 'artillery_indirect_suitability'] = 0.0
            self.df.loc[high_mask, 'artillery_direct_suitability'] = 0.0
        
        try:
            feats = self.df[['elevation_m','slope_deg','roughness','visibility_from_enemy','observation_of_enemy','viewshed_on_route']].fillna(0.0).astype('float32').values
            W1 = None; b1 = None; Wout = None; bout = None
            weights_path = os.path.join(os.path.dirname(__file__), 'models', 'terrain_nn_weights.npz')
            if os.path.exists(weights_path):
                data = np.load(weights_path)
                W1 = data.get('W1'); b1 = data.get('b1'); Wout = data.get('Wout'); bout = data.get('bout')
            if W1 is None:
                W1 = np.array([[ 1.0e-4, -2.0e-2,  1.0e-2, -5.0e-1,  6.0e-1,  2.0e-1],
                               [ 2.0e-4, -1.0e-2,  1.5e-2, -4.0e-1,  5.0e-1,  3.0e-1]], dtype=np.float32)
                b1 = np.array([0.1, 0.05], dtype=np.float32)
                Wout = np.array([[ 1.2,  1.0,  0.8],
                                 [-0.3,  0.2,  0.1]], dtype=np.float32)
                bout = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            h = np.maximum(0.0, feats @ W1.T + b1)
            logits = h @ Wout + bout
            preds = 1.0 / (1.0 + np.exp(-logits))
            def_head = preds[:, 0]
            off_head = preds[:, 1]
            art_head = preds[:, 2]
            self.df['defensive_suitability'] = np.clip(0.7*self.df['defensive_suitability'] + 0.3*def_head, 0.0, 1.0)
            self.df['observation_post_suitability'] = np.clip(0.7*self.df['observation_post_suitability'] + 0.3*off_head, 0.0, 1.0)
            self.df['artillery_indirect_suitability'] = np.clip(0.7*self.df['artillery_indirect_suitability'] + 0.3*art_head, 0.0, 1.0)
            self.df['artillery_direct_suitability'] = np.clip(0.7*self.df['artillery_direct_suitability'] + 0.3*art_head, 0.0, 1.0)
        except Exception:
            pass
        
        print("Applying network analysis (penalizing isolated positions)...")
        self._analyze_defensive_network()
        
        print("Computing combined suitability scores...")
        self.df['combined_defensive_artillery'] = (self.df['networked_defensive_suitability'] * 0.6) + (self.df['artillery_indirect_suitability'] * 0.4)
        self.df['combined_observation_artillery'] = (self.df['observation_post_suitability'] * 0.5) + (self.df['artillery_direct_suitability'] * 0.5)

        self.analysis_complete = True
        print("Analysis complete.")
        return self.df
        
    def _analyze_defensive_network(self, support_radius_m: float = 3000.0, top_n: int = 200):
        if 'defensive_suitability' not in self.df.columns:
            return
            
        support_radius_deg = support_radius_m / (self.pderl.km_per_deg * 1000.0)
        
        if top_n > len(self.df):
            top_n = len(self.df)
            
        top_n_indices = self.df.nlargest(top_n, 'defensive_suitability').index
        
        penalties = []
        for idx in self.df.index:
            if idx not in top_n_indices:
                penalties.append(1.0)
                continue
            
            point = self.df.loc[idx, ['longitude', 'latitude']].values
            
            nearby_friendlies = self.pderl.tree.query_ball_point(point, support_radius_deg)
            supporting_friendlies = [i for i in nearby_friendlies if i in top_n_indices and i != idx]
            
            if not supporting_friendlies:
                penalties.append(0.5)
            elif len(supporting_friendlies) < 2:
                penalties.append(0.85)
            else:
                penalties.append(1.0)
                
        self.df['defensive_suitability'] *= penalties
        self.df = self.df.rename(columns={'defensive_suitability': 'networked_defensive_suitability'})
        
    def get_top_positions(self, category: str, n: int = 10) -> pd.DataFrame:
        if not self.analysis_complete:
            print("Error: Analysis must be run first.")
            return pd.DataFrame()
            
        col_map = {
            'defensive': 'networked_defensive_suitability',
            'observation': 'observation_post_suitability',
            'assault': 'assault_approach_suitability',
            'artillery_indirect': 'artillery_indirect_suitability',
            'artillery_direct': 'artillery_direct_suitability',
            'defensive_hybrid': 'combined_defensive_artillery',
            'offensive_hybrid': 'combined_observation_artillery'
        }
        
        if category not in col_map:
            raise ValueError(f"Category must be one of {list(col_map.keys())}")
        
        if n > len(self.df):
            n = len(self.df)
        
        return self.df.nlargest(n, col_map[category])
    
    def save_analysis(self, output_path: str):
        if not self.analysis_complete:
            print("Error: Analysis must be run first. No file saved.")
            return
        
        if os.path.exists(output_path):
            os.replace(output_path, output_path + '.bak')
        if os.path.exists('strategic_terrain_analysis.csv'):
            os.replace('strategic_terrain_analysis.csv', 'strategic_terrain_analysis.csv.bak')
        
        self.df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"Analysis saved to {output_path}")
        try:
            vis_out = 'strategic_terrain_analysis.csv'
            if 'longitude' in self.df.columns and 'latitude' in self.df.columns and 'elevation_m' in self.df.columns:
                defensive = None
                offensive = None
                artillery = None

                if 'networked_defensive_suitability' in self.df.columns:
                    defensive = self.df['networked_defensive_suitability']
                elif 'defensive_suitability' in self.df.columns:
                    defensive = self.df['defensive_suitability']

                if 'combined_observation_artillery' in self.df.columns:
                    offensive = self.df['combined_observation_artillery']
                elif 'observation_post_suitability' in self.df.columns:
                    offensive = self.df['observation_post_suitability']

                if 'artillery_indirect_suitability' in self.df.columns and 'artillery_direct_suitability' in self.df.columns:
                    artillery = np.maximum(self.df['artillery_indirect_suitability'], self.df['artillery_direct_suitability'])
                elif 'artillery_indirect_suitability' in self.df.columns:
                    artillery = self.df['artillery_indirect_suitability']
                elif 'artillery_direct_suitability' in self.df.columns:
                    artillery = self.df['artillery_direct_suitability']

                compat = pd.DataFrame({
                    'longitude': self.df['longitude'],
                    'latitude': self.df['latitude'],
                    'elevation_m': self.df['elevation_m'],
                    'defensive_suitability': defensive if defensive is not None else 0.0,
                    'offensive_suitability': offensive if offensive is not None else 0.0,
                    'artillery_suitability': artillery if artillery is not None else 0.0,
                })
                compat.to_csv(vis_out, index=False, float_format='%.6f')
                print(f"Analysis saved to {vis_out}")
        except Exception as e:
            print(f"Could not write compatibility file: {e}")


if __name__ == "__main__":
    
    # ============== STEP 1: DRONE PERSON DETECTION (OPTIONAL) ==============
    video_path = 'footage.mp4'
    gps_path = 'drone_gps_log.csv'
    USE_DRONE_DETECTION = os.path.exists(video_path) and os.path.exists(gps_path)
    
    if USE_DRONE_DETECTION:
        print("\n" + "="*60)
        print("STEP 1: DRONE PERSON DETECTION")
        print("="*60)
        try:
            detector = DronePersonDetector(model_path="Mahadih534/YoloV8-VisDrone")
            detected_persons_df = detector.process_video_with_gps(
                video_path=video_path,
                drone_gps_csv=gps_path,
                output_csv='detected_persons.csv',
                confidence_threshold=0.6,
                altitude_m=100.0,
                camera_fov_deg=84.0
            )
            DETECTED_PERSONS_FILE = 'detected_persons.csv' if len(detected_persons_df) > 0 else None
        except Exception as e:
            print(f"drone detection failed: {e}")
            DETECTED_PERSONS_FILE = None
    else:
        DETECTED_PERSONS_FILE = None
        print("\ndrone detection skipped (missing footage.mp4 or drone_gps_log.csv)")
    
    # ============== STEP 2: TERRAIN ANALYSIS ==============
    print("\n" + "="*60)
    print("STEP 2: STRATEGIC TERRAIN ANALYSIS")
    print("="*60)
    
    TERRAIN_DATA_FILE = 'terrain_data.jsonl'
    THREAT_POSITIONS_FILE = 'threat_positions.csv'
    
    KEY_SUPPLY_ROUTE = [
        (75.5, 33.1), 
        (75.5, 33.5), 
        (75.5, 33.9)
    ]
    
    SUPPLY_ROUTE_POINTS = KEY_SUPPLY_ROUTE
    
    try:
        analyzer = StrategicTerrainAnalyzer(TERRAIN_DATA_FILE)
        
        # if drone detections exist, append to threat_positions.csv and de-duplicate
        if DETECTED_PERSONS_FILE:
            try:
                persons_df = pd.read_csv(DETECTED_PERSONS_FILE)
                if len(persons_df) > 0:
                    person_threats = persons_df[['longitude','latitude']].dropna().drop_duplicates()
                    person_threats = person_threats.assign(elevation_m=np.nan, source='drone')
                    if os.path.exists(THREAT_POSITIONS_FILE):
                        base_thr = pd.read_csv(THREAT_POSITIONS_FILE)
                    else:
                        base_thr = pd.DataFrame(columns=['longitude','latitude','elevation_m','source'])
                    # keep existing elevation and source if present
                    base_cols = [c for c in ['longitude','latitude','elevation_m','source'] if c in base_thr.columns]
                    merged = pd.concat([
                        base_thr[base_cols],
                        person_threats[['longitude','latitude','elevation_m','source']]
                    ], ignore_index=True)
                    merged = merged.drop_duplicates(subset=['longitude','latitude'], keep='first')
                    merged.to_csv(THREAT_POSITIONS_FILE, index=False, float_format='%.6f')
                    print(f"updated threats with drone detections -> {THREAT_POSITIONS_FILE} ({len(merged)})")
            except Exception as e:
                print(f"could not merge detected persons into threats: {e}")

        results_df = analyzer.analyze_terrain_with_context(
            threat_positions_csv=THREAT_POSITIONS_FILE,
            key_route_points=KEY_SUPPLY_ROUTE,
            supply_route_points=SUPPLY_ROUTE_POINTS,
            detected_persons_csv=DETECTED_PERSONS_FILE  # NEW: Pass detected persons
        )
        
        print("\n--- STRATEGIC ANALYSIS COMPLETE ---")
        
        print("\nTOP 10 COMBINED DEFENSIVE/ARTILLERY POSTS:")
        print(analyzer.get_top_positions('defensive_hybrid', n=10)[
            ['longitude', 'latitude', 'elevation_m', 'combined_defensive_artillery', 'is_reverse_slope']
        ])

        print("\nTOP 10 COMBINED OBSERVATION/DIRECT-FIRE POSTS:")
        print(analyzer.get_top_positions('offensive_hybrid', n=10)[
            ['longitude', 'latitude', 'elevation_m', 'combined_observation_artillery', 'observation_of_enemy']
        ])
        
        print("\nTOP 10 DEFENSIVE POSITIONS (Networked):")
        print(analyzer.get_top_positions('defensive', n=10)[
            ['longitude', 'latitude', 'elevation_m', 'networked_defensive_suitability', 'is_reverse_slope']
        ])
        
        print("\nTOP 10 OBSERVATION POSTS (for key route):")
        print(analyzer.get_top_positions('observation', n=10)[
            ['longitude', 'latitude', 'elevation_m', 'observation_post_suitability', 'viewshed_on_route']
        ])
        
        print("\nTOP 10 CONCEALED ASSAULT APPROACHES:")
        print(analyzer.get_top_positions('assault', n=10)[
            ['longitude', 'latitude', 'elevation_m', 'assault_approach_suitability', 'visibility_from_enemy']
        ])
        
        analyzer.save_analysis('Strategic_Analysis_Output.csv')

    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"File not found: {e.filename}")
        print("Please ensure required files are in the same directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()