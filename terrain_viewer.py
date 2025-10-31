import pygame
import pandas as pd
import numpy as np
import threading
import time
import requests
import json

from ui_components import (
    GyroControl, MarkerSelector, draw_markers, export_markers_to_json,
    MARKER_TYPES, GYRO_CENTER, GYRO_RADIUS
)

API_URL = "http://localhost:8080/api/markers"
UPDATE_INTERVAL = 1.0  # seconds


# --- Live update client thread ---
class LiveClient(threading.Thread):
    def __init__(self, get_state_func):
        super().__init__(daemon=True)
        self.get_state_func = get_state_func
        self.running = True

    def run(self):
        while self.running:
            try:
                data = self.get_state_func()
                if data:
                    requests.post(API_URL, json=data, timeout=2)
            except Exception as e:
                # Fail silently to avoid crashing the viewer
                print(f"⚠️ Live update failed: {e}")
            time.sleep(UPDATE_INTERVAL)


# --- Load terrain data ---
df = pd.read_json("terrain_data.jsonl", lines=True)
df = df[df["elevation"].notnull()]

lats = sorted(df["lat"].unique())
lons = sorted(df["lon"].unique())

DOWNSAMPLE = 2
lats = lats[::DOWNSAMPLE]
lons = lons[::DOWNSAMPLE]

elev_grid = np.zeros((len(lats), len(lons)))
ndvi_grid = np.zeros((len(lats), len(lons)))
ndwi_grid = np.zeros((len(lats), len(lons)))

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        cell = df.iloc[((df["lat"] - lat)**2 + (df["lon"] - lon)**2).argmin()]
        elev_grid[i, j] = cell["elevation"]
        ndvi_grid[i, j] = cell.get("ndvi", np.nan)
        ndwi_grid[i, j] = cell.get("ndwi", np.nan)

zmin, zmax = elev_grid.min(), elev_grid.max()
elev_grid_norm = (elev_grid - zmin) / (zmax - zmin)

color_grid = np.zeros((len(lats), len(lons), 3), dtype=np.uint8)
for i in range(len(lats)):
    for j in range(len(lons)):
        ndvi, ndwi = ndvi_grid[i, j], ndwi_grid[i, j]
        z = elev_grid_norm[i, j]
        if not np.isnan(ndwi) and ndwi > 0.1:
            color = np.array([40, 100, 255])
        elif not np.isnan(ndvi) and ndvi > 0.3:
            color = np.array([60, int(150 + 80 * ndvi), 60])
        else:
            color = np.array([150 + 50 * z, 100 + 30 * z, 60 + 40 * z])
        if i < len(elev_grid_norm) - 1 and j < len(elev_grid_norm[0]) - 1:
            dzdx = elev_grid_norm[i, j + 1] - elev_grid_norm[i, j]
            dzdy = elev_grid_norm[i + 1, j] - elev_grid_norm[i, j]
            shade = 1 - 0.8 * (dzdx + dzdy)
            color = np.clip(color * shade, 0, 255)
        color_grid[i, j] = color.astype(np.uint8)

# --- Pygame setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Terrain Viewer (Yaw + Roll + Live Sync)")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 18)

gyro_control = GyroControl(GYRO_CENTER, GYRO_RADIUS, font, font_small)
marker_selector = MarkerSelector(280, 310, WIDTH - 280 - 20, 20, font, font_small)
markers = []

SCALE_X, SCALE_Y, HEIGHT_SCALE = 10, 10, 250
angle_yaw = np.radians(45)
angle_roll = np.radians(20)
mid_x, mid_y = len(lons) / 2, len(lats) / 2
offset_x, offset_y = WIDTH // 2, HEIGHT // 2
zoom = 1.0

dragging = False
gyro_dragging = False
last_mouse_pos = (0, 0)
terrain_surface = None
points = None
needs_redraw = True


# --- Projection ---
def project_vectorized():
    i_grid, j_grid = np.meshgrid(range(len(lats)), range(len(lons)), indexing='ij')
    z = elev_grid_norm * HEIGHT_SCALE
    x = (j_grid - mid_x) * SCALE_X
    y = (i_grid - mid_y) * SCALE_Y

    x2 = x * np.cos(angle_yaw) - y * np.sin(angle_yaw)
    y2 = x * np.sin(angle_yaw) + y * np.cos(angle_yaw)

    y3 = y2 * np.cos(angle_roll) - z * np.sin(angle_roll)
    z2 = y2 * np.sin(angle_roll) + z * np.cos(angle_roll)

    px = offset_x + x2 * zoom
    py = offset_y + y3 * zoom - z2 * 0.2 * zoom
    return px.astype(int), py.astype(int)


def recompute_surface():
    surf = pygame.Surface((WIDTH, HEIGHT))
    surf.fill((15, 15, 25))
    px, py = project_vectorized()
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            pts = [(px[i, j], py[i, j]),
                   (px[i, j + 1], py[i, j + 1]),
                   (px[i + 1, j + 1], py[i + 1, j + 1]),
                   (px[i + 1, j], py[i + 1, j])]
            pygame.draw.polygon(surf, tuple(color_grid[i, j]), pts)
    return surf, (px, py)


def get_live_state():
    """Prepare JSON payload for live updates"""
    return {
        "timestamp": time.time(),
        "yaw_deg": float(np.degrees(angle_yaw) % 360),
        "roll_deg": float(np.degrees(angle_roll)),
        "marker_count": len(markers),
        "markers": [
            {
                "id": m["id"],
                "team": m["team"],
                "type": m["type"],
                "lat": float(lats[m["pos"][0]]),
                "lon": float(lons[m["pos"][1]]),
                "elev": float(elev_grid[m["pos"][0], m["pos"][1]]),
            } for m in markers
        ]
    }


# Start live client thread
live_client = LiveClient(get_live_state)
live_client.start()

# --- Initial terrain render ---
terrain_surface, points = recompute_surface()
needs_redraw = False

running = True
while running:
    keys_pressed = pygame.key.get_pressed()
    ctrl_pressed = keys_pressed[pygame.K_LCTRL] or keys_pressed[pygame.K_RCTRL]
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if marker_selector.is_over_button(event.pos, marker_selector.friendly_rect):
                    if marker_selector.current_team != 'friendly':
                        marker_selector.current_team = 'friendly'
                        marker_selector.current_marker_type = '1'
                        print("Switched to FRIENDLY markers")
                elif marker_selector.is_over_button(event.pos, marker_selector.enemy_rect):
                    if marker_selector.current_team != 'enemy':
                        marker_selector.current_team = 'enemy'
                        marker_selector.current_marker_type = '1'
                        print("Switched to ENEMY markers")
                elif gyro_control.is_inside(event.pos):
                    gyro_dragging = True
                    angle_yaw, angle_roll = gyro_control.update_angles(event.pos, angle_yaw, angle_roll)
                    needs_redraw = True
                else:
                    dragging = True
                    last_mouse_pos = event.pos
            elif event.button == 3:
                mx, my = event.pos
                px, py = points
                dist = (px - mx) ** 2 + (py - my) ** 2
                best_idx = np.unravel_index(np.argmin(dist), dist.shape)
                i, j = best_idx
                current_type = marker_selector.get_current_type()
                current_team = marker_selector.get_current_team()
                marker_id = marker_selector.generate_marker_id()
                markers.append({
                    'id': marker_id,
                    'pos': best_idx,
                    'type': current_type,
                    'team': current_team
                })
                info = MARKER_TYPES[current_team][current_type]['name']
                print(f"📍 Placed {current_team.upper()} {info} (ID: {marker_id})")
                needs_redraw = True
            elif event.button == 4:
                zoom *= 1.1
                needs_redraw = True
            elif event.button == 5:
                zoom /= 1.1
                needs_redraw = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
                gyro_dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if gyro_dragging:
                angle_yaw, angle_roll = gyro_control.update_angles(event.pos, angle_yaw, angle_roll)
                needs_redraw = True
            elif dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                offset_x += dx
                offset_y += dy
                last_mouse_pos = event.pos
                needs_redraw = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                angle_yaw -= np.radians(5)
                needs_redraw = True
            elif event.key == pygame.K_RIGHT:
                angle_yaw += np.radians(5)
                needs_redraw = True
            elif event.key == pygame.K_UP:
                angle_roll = max(0, angle_roll - np.radians(5))
                needs_redraw = True
            elif event.key == pygame.K_DOWN:
                angle_roll = min(np.pi, angle_roll + np.radians(5))
                needs_redraw = True
            elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if markers:
                    removed = markers.pop()
                    print(f"🗑️ Removed marker {removed['id']}")
                    needs_redraw = True
            elif event.key == pygame.K_f:
                if marker_selector.current_team != 'friendly':
                    marker_selector.current_team = 'friendly'
                    marker_selector.current_marker_type = '1'
                    print("Switched to FRIENDLY markers")
            elif event.key == pygame.K_e:
                if marker_selector.current_team != 'enemy':
                    marker_selector.current_team = 'enemy'
                    marker_selector.current_marker_type = '1'
                    print("Switched to ENEMY markers")
            elif event.key == pygame.K_s and ctrl_pressed:
                if markers:
                    export_markers_to_json(markers, lats, lons, elev_grid)
                else:
                    print("⚠️ No markers to export")
            elif event.unicode in MARKER_TYPES[marker_selector.get_current_team()]:
                marker_selector.set_marker_type(event.unicode)
                team = marker_selector.get_current_team()
                print(f"Selected marker: {MARKER_TYPES[team][event.unicode]['name']}")

    if needs_redraw:
        terrain_surface, points = recompute_surface()
        needs_redraw = False

    screen.fill((0, 0, 0))
    screen.blit(terrain_surface, (0, 0))
    draw_markers(screen, markers, points, font)
    gyro_control.draw(screen, angle_yaw, angle_roll)
    marker_selector.draw(screen)

    pygame.display.flip()
    clock.tick(60)

live_client.running = False
pygame.quit()

