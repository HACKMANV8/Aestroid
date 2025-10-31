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

"""Drone model code removed; detection is provided via detected_persons.csv input."""

# ============== TERRAIN ANALYSIS MODULE ==============

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
        if from_idx == to_idx: return True
        p1, p2 = self.points[from_idx], self.points[to_idx]
        e1, e2 = self.elevations[from_idx] + observer_height, self.elevations[to_idx] + target_height
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist_2d_deg = np.sqrt(dx**2 + dy**2)
        if dist_2d_deg < 1e-6: return True
        if dist_2d_deg > self.max_range_deg: return False
        
        dist_2d_m = dist_2d_deg * self.km_per_deg * 1000.0
        num_samples = max(10, int(dist_2d_m / LOS_SAMPLE_METERS))
        
        for i in range(1, num_samples):
            t = i / num_samples
            sample_lon, sample_lat = p1[0] + t * dx, p1[1] + t * dy
            _, nearest_terrain_idx = self.tree.query([[sample_lon, sample_lat]], k=1)
            terrain_elev = self.elevations[nearest_terrain_idx[0]]
            los_elev_at_sample = e1 + t * (e2 - e1)
            if terrain_elev > los_elev_at_sample + 0.5:
                return False
        return True
    
    def compute_visibility_from_threats(self, point_idx: int, threat_indices: List[int]) -> float:
        if not threat_indices: return 0.0
        visible_count = sum(1 for idx in threat_indices if self.compute_line_of_sight(idx, point_idx))
        return visible_count / len(threat_indices)
    
    def compute_observation_of_targets(self, point_idx: int, target_indices: List[int]) -> float:
        if not target_indices: return 0.0
        visible_count = sum(1 for idx in target_indices if self.compute_line_of_sight(point_idx, idx, ARTILLERY_OBSERVER_HEIGHT_M))
        return visible_count / len(target_indices)

    def compute_viewshed_on_route(self, point_idx: int, route_points: np.ndarray) -> float:
        if route_points.size == 0: return 0.0
        route_indices = np.unique(self.tree.query(route_points, k=1)[1])
        if route_indices.size == 0: return 0.0
        visible_count = sum(1 for idx in route_indices if self.compute_line_of_sight(point_idx, idx, ARTILLERY_OBSERVER_HEIGHT_M, 2.0))
        return visible_count / len(route_indices)


class TerrainFeatureExtractor:
    
    @staticmethod
    def compute_slope_aspect(df: pd.DataFrame, k_neighbors: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        points_m = df[['longitude', 'latitude']].values * 111000.0
        elevations = df['elevation_m'].values
        tree = cKDTree(points_m)
        slopes_deg, aspects_deg = np.zeros(len(df)), np.zeros(len(df))
        
        for idx in range(len(df)):
            try:
                distances, indices = tree.query(points_m[idx], k=k_neighbors+1)
                indices = indices[1:]
                if len(indices) < 3: continue
                neighbors_m, elevs_neighbor = points_m[indices], elevations[indices]
                A = np.c_[neighbors_m, np.ones(len(indices))]
                coeffs, _, _, _ = np.linalg.lstsq(A, elevs_neighbor, rcond=None)
                a, b = coeffs[0], coeffs[1]
                slopes_deg[idx] = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))
                aspects_deg[idx] = (np.degrees(np.arctan2(a, b)) + 90) % 360
            except Exception: pass
        return slopes_deg, aspects_deg
    
    @staticmethod
    def compute_roughness(df: pd.DataFrame, radius_m: float = 100.0) -> np.ndarray:
        points, elevations = df[['longitude', 'latitude']].values, df['elevation_m'].values
        tree = cKDTree(points)
        roughness = np.zeros(len(df))
        radius_deg = radius_m / 111000.0
        for idx in range(len(df)):
            indices = tree.query_ball_point(points[idx], radius_deg)
            if len(indices) > 3:
                roughness[idx] = np.std(elevations[indices])
        return roughness


class TacticalScoringEngine:
    
    @staticmethod
    def score_defensive_position(e, s, v, rs, r, d_supply) -> float:
        elev_score = np.clip(e / 5500.0, 0.5, 1.0)
        slope_score = 1.0 if 15 <= s <= 40 else 0.3
        concealment_score = (1.0 - v + (1.0 if rs else 0.0)) / 2.0
        cover_score = np.clip(r / 50.0, 0.1, 1.0)
        supply_score = np.clip(1.0 - (d_supply / 5000.0), 0.0, 1.0)
        return ((0.25 * elev_score) + (0.10 * slope_score) + (0.35 * concealment_score) +
                (0.15 * cover_score) + (0.15 * supply_score))

    @staticmethod
    def score_observation_post(v_route, v_enemy, v_from_enemy, e) -> float:
        total_obs_score = (v_route + v_enemy) / 2.0
        concealment_score = 1.0 - v_from_enemy
        elev_score = np.clip(e / 5500.0, 0.5, 1.0)
        return (0.50 * total_obs_score) + (0.35 * concealment_score) + (0.15 * elev_score)

    @staticmethod
    def score_assault_approach(s, r, v_from_enemy) -> float:
        concealment_score = 1.0 - v_from_enemy
        difficulty_score = (np.clip(s / 60.0, 0.0, 1.0) + np.clip(r / 50.0, 0.0, 1.0)) / 2.0
        return (0.60 * concealment_score) + (0.40 * difficulty_score)

    @staticmethod
    def score_artillery_indirect(s, rs, v_from_enemy) -> float:
        slope_score = np.clip(1.0 - (s / 15.0), 0.0, 1.0)
        concealment_score = (1.0 - v_from_enemy + (1.0 if rs else 0.0)) / 2.0
        return (0.40 * slope_score) + (0.60 * concealment_score)

    @staticmethod
    def score_artillery_direct(s, v_enemy, v_from_enemy) -> float:
        slope_score = np.clip(1.0 - (s / 10.0), 0.0, 1.0)
        return (0.30 * slope_score) + (0.40 * v_enemy) + (0.30 * (1.0 - v_from_enemy))


class StrategicTerrainAnalyzer:
    
    def __init__(self, terrain_csv_path: str):
        print(f"Loading terrain data from: {terrain_csv_path}")
        if str(terrain_csv_path).lower().endswith('.jsonl'):
            df = pd.read_json(terrain_csv_path, lines=True)
        else:
            df = pd.read_csv(terrain_csv_path)

        rename_map = {}
        if 'lon' in df.columns: rename_map['lon'] = 'longitude'
        if 'lat' in df.columns: rename_map['lat'] = 'latitude'
        if 'elevation' in df.columns: rename_map['elevation'] = 'elevation_m'
        if rename_map: df = df.rename(columns=rename_map)

        if 'elevation_m' not in df.columns:
            raise ValueError("CSV/JSONL must contain 'elevation_m' column.")
            
        df = df[df['elevation_m'].notnull()].reset_index(drop=True)
        print(f"Loaded {len(df)} terrain points.")
        
        if len(df) > MAX_ANALYSIS_POINTS:
            print(f"Large dataset detected. Subsampling to {MAX_ANALYSIS_POINTS} points...")
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
        if not points_list: return []
        query_points = np.array(points_list)
        distances, indices = self.pderl.tree.query(query_points, k=1)
        valid_indices = indices[distances < 0.1] # ~10km threshold
        return valid_indices.tolist()

    def analyze_terrain_with_context(self, threat_positions_csv: str,
                                     key_route_points: List[Tuple[float, float]],
                                     supply_route_points: List[Tuple[float, float]],
                                     detected_persons_csv: Optional[str] = None,
                                     min_person_confidence: float = 0.5) -> pd.DataFrame:
        
        print(f"Loading base threat positions from: {threat_positions_csv}")
        try:
            enemy_df = pd.read_csv(threat_positions_csv)
        except FileNotFoundError:
            print(f"Warning: {threat_positions_csv} not found. Proceeding with no base threats.")
            enemy_df = pd.DataFrame(columns=['longitude', 'latitude'])

        # Merge drone-detected persons into threats
        if detected_persons_csv and os.path.exists(detected_persons_csv):
            print(f"Loading drone-detected persons from: {detected_persons_csv}")
            persons_df = pd.read_csv(detected_persons_csv)
            if len(persons_df) > 0:
                high_conf = persons_df[persons_df['confidence'] >= min_person_confidence]
                print(f"Adding {len(high_conf)} person detections (conf >= {min_person_confidence}) as threats")
                person_threats = high_conf[['longitude', 'latitude']].drop_duplicates()
                enemy_df = pd.concat([enemy_df, person_threats], ignore_index=True).drop_duplicates(subset=['longitude', 'latitude'])
        
        enemy_list = list(zip(enemy_df['longitude'], enemy_df['latitude']))
        enemy_indices = self._find_nearest_indices(enemy_list)
        print(f"Analyzing with {len(enemy_indices)} total unique threat positions.")

        supply_indices = self._find_nearest_indices(supply_route_points)
        supply_points_arr = self.df.loc[supply_indices, ['longitude', 'latitude']].values
        key_route_arr = np.array(key_route_points)
        
        print("Computing PDERL visibility/observation analysis (this is the slowest step)...")
        scores = {
            'visibility_from_enemy': np.zeros(len(self.df)),
            'observation_of_enemy': np.zeros(len(self.df)),
            'viewshed_on_route': np.zeros(len(self.df)),
            'distance_to_supply_m': np.full(len(self.df), 5000.0), # Default 5km
            'is_reverse_slope': np.zeros(len(self.df), dtype=bool)
        }
        
        enemy_points_arr = self.df.loc[enemy_indices, ['longitude', 'latitude']].values
        
        for idx in range(len(self.df)):
            if idx % 500 == 0 and idx > 0:
                print(f"   ...processing point {idx}/{len(self.df)}")
            
            scores['visibility_from_enemy'][idx] = self.pderl.compute_visibility_from_threats(idx, enemy_indices)
            scores['observation_of_enemy'][idx] = self.pderl.compute_observation_of_targets(idx, enemy_indices)
            scores['viewshed_on_route'][idx] = self.pderl.compute_viewshed_on_route(idx, key_route_arr)
            
            point = self.df.loc[idx, ['longitude', 'latitude']].values
            
            if supply_points_arr.any():
                dists_to_supply = np.sqrt(((supply_points_arr - point)**2).sum(axis=1))
                scores['distance_to_supply_m'][idx] = dists_to_supply.min() * self.pderl.km_per_deg * 1000.0
            
            if enemy_points_arr.any():
                dists_to_enemy = np.sqrt(((enemy_points_arr - point)**2).sum(axis=1))
                nearest_enemy_point = enemy_points_arr[dists_to_enemy.argmin()]
                dx, dy = nearest_enemy_point[0] - point[0], nearest_enemy_point[1] - point[1]
                bearing_to_enemy = (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0
                aspect = self.df.loc[idx, 'aspect_deg']
                angle_diff = 180.0 - abs(abs(aspect - bearing_to_enemy) - 180.0)
                scores['is_reverse_slope'][idx] = angle_diff > 90.0

        for key, val in scores.items():
            self.df[key] = val
        
        print("Computing final tactical suitability scores...")
        self.df['defensive_suitability'] = self.df.apply(
            lambda row: TacticalScoringEngine.score_defensive_position(
                row['elevation_m'], row['slope_deg'], row['visibility_from_enemy'],
                row['is_reverse_slope'], row['roughness'], row['distance_to_supply_m']
            ), axis=1
        )
        self.df['observation_post_suitability'] = self.df.apply(
            lambda row: TacticalScoringEngine.score_observation_post(
                row['viewshed_on_route'], row['observation_of_enemy'],
                row['visibility_from_enemy'], row['elevation_m']
            ), axis=1
        )
        self.df['assault_approach_suitability'] = self.df.apply(
            lambda row: TacticalScoringEngine.score_assault_approach(
                row['slope_deg'], row['roughness'], row['visibility_from_enemy']
            ), axis=1
        )
        self.df['artillery_indirect_suitability'] = self.df.apply(
            lambda row: TacticalScoringEngine.score_artillery_indirect(
                row['slope_deg'], row['is_reverse_slope'], row['visibility_from_enemy']
            ), axis=1
        )
        self.df['artillery_direct_suitability'] = self.df.apply(
            lambda row: TacticalScoringEngine.score_artillery_direct(
                row['slope_deg'], row['observation_of_enemy'], row['visibility_from_enemy']
            ), axis=1
        )
            
        print("Applying network analysis (penalizing isolated positions)...")
        self._analyze_defensive_network()
        
        print("Computing combined suitability scores...")
        self.df['combined_defensive_artillery'] = (self.df['networked_defensive_suitability'] * 0.6) + (self.df['artillery_indirect_suitability'] * 0.4)
        self.df['combined_observation_artillery'] = (self.df['observation_post_suitability'] * 0.5) + (self.df['artillery_direct_suitability'] * 0.5)

        self.analysis_complete = True
        print("Analysis complete.")
        return self.df
        
    def _analyze_defensive_network(self, support_radius_m: float = 3000.0, top_n_percent: float = 0.1):
        if 'defensive_suitability' not in self.df.columns:
            return
            
        support_radius_deg = support_radius_m / (self.pderl.km_per_deg * 1000.0)
        
        top_n = max(100, int(len(self.df) * top_n_percent))
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
            
            if not supporting_friendlies: penalties.append(0.5)
            elif len(supporting_friendlies) < 2: penalties.append(0.85)
            else: penalties.append(1.0)
                
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
        
        if n > len(self.df): n = len(self.df)
        return self.df.nlargest(n, col_map[category])
    
    def save_analysis(self, output_path: str):
        if not self.analysis_complete:
            print("Error: Analysis must be run first. No file saved.")
            return
        
        self.df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"Full analysis saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Strategic terrain analysis using threats and detected persons.')
    parser.add_argument('--terrain', default='terrain_data.jsonl', help='terrain data CSV/JSONL')
    parser.add_argument('--threat-positions', default='threat_positions.csv', help='base threat positions CSV')
    parser.add_argument('--detected-persons', default='detected_persons.csv', help='detected persons CSV (from detection pipeline)')
    parser.add_argument('--min-person-confidence', type=float, default=0.5, help='min confidence to consider a person as threat')
    parser.add_argument('--output', default='Strategic_Analysis_Output.csv', help='output CSV path')
    args = parser.parse_args()

    KEY_SUPPLY_ROUTE = [
        (75.5, 33.1), 
        (75.5, 33.5), 
        (75.5, 33.9)
    ]

    try:
        analyzer = StrategicTerrainAnalyzer(args.terrain)
        analyzer.analyze_terrain_with_context(
            threat_positions_csv=args.threat_positions,
            key_route_points=KEY_SUPPLY_ROUTE,
            supply_route_points=KEY_SUPPLY_ROUTE,
            detected_persons_csv=(args.detected_persons if os.path.exists(args.detected_persons) else None),
            min_person_confidence=args.min_person_confidence,
        )

        print("\n--- STRATEGIC ANALYSIS COMPLETE ---")
        analyzer.save_analysis(args.output)

    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()