import pygame
import pandas as pd
import numpy as np
import threading
import time
import requests
import json

from ui_components import (
    MarkerSelector, export_markers_to_json,
    MARKER_TYPES, GYRO_CENTER, GYRO_RADIUS
)

API_URL = "http://localhost:8080/api/markers"
REMOTE_URL = "http://localhost:6969"  # your ncat server
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
                print(f"⚠️ Live update failed: {e}")
            time.sleep(UPDATE_INTERVAL)


# --- Remote marker fetcher ---
class RemoteMarkerFetcher(threading.Thread):
    def __init__(self, url, update_callback, interval=2.0):
        super().__init__(daemon=True)
        self.url = url
        self.update_callback = update_callback
        self.interval = interval
        self.running = True
        self.last_success = None
        self.error_count = 0

    def run(self):
        print(f"🚀 Remote marker fetcher started, polling: {self.url}")
        while self.running:
            try:
                resp = requests.get(self.url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    if "lat" in data and "lon" in data:
                        self.update_callback(data["lat"], data["lon"])
                        self.last_success = time.time()
                        self.error_count = 0
                    else:
                        print(f"⚠️ Remote data missing lat/lon: {data}")
                else:
                    print(f"⚠️ Remote server returned status {resp.status_code}")
                    self.error_count += 1
            except requests.exceptions.ConnectionError:
                self.error_count += 1
                if self.error_count == 1:  # Print only first error
                    print(f"⚠️ Cannot connect to {self.url} - is the server running?")
            except requests.exceptions.Timeout:
                self.error_count += 1
                if self.error_count == 1:
                    print(f"⚠️ Request timeout to {self.url}")
            except Exception as e:
                if self.error_count == 0:
                    print(f"❌ Remote fetch error: {e}")
                self.error_count += 1
            
            time.sleep(self.interval)


# --- Load terrain data ---
print("Loading terrain data...")
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

print("Computing terrain colors...")
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
pygame.display.set_caption("3D Terrain Viewer (Live API + Remote Marker)")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 18)

# --- Gyro control class ---
class GyroControl:
    """Gyro control widget for adjusting yaw and roll angles"""
    
    def __init__(self, center, radius, font, font_small):
        self.center = center
        self.radius = radius
        self.font = font
        self.font_small = font_small
    
    def is_inside(self, pos):
        """Check if position is inside the gyro control"""
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius
    
    def update_angles(self, pos, angle_yaw, angle_roll):
        """Update angles based on mouse position in gyro"""
        import math
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        # Constrain to radius
        if dist > self.radius * 0.8:
            s = self.radius * 0.8 / dist
            dx *= s
            dy *= s
        
        yaw_norm = (dx / (self.radius * 0.8)) / 2 + 0.5
        roll_norm = -(dy / (self.radius * 0.8)) / 2 + 0.5
        new_yaw = yaw_norm * 2 * np.pi
        new_roll = np.clip(roll_norm * np.pi, 0, np.pi)
        
        return new_yaw, new_roll
    
    def draw(self, surface, angle_yaw, angle_roll):
        """Draw the gyro control widget"""
        import math
        cx, cy = self.center
        
        # Draw background circles
        pygame.draw.circle(surface, (40, 40, 50), self.center, self.radius + 5)
        pygame.draw.circle(surface, (25, 25, 35), self.center, self.radius)
        
        # Draw crosshairs
        pygame.draw.line(surface, (60, 60, 70), (cx - self.radius, cy), (cx + self.radius, cy))
        pygame.draw.line(surface, (60, 60, 70), (cx, cy - self.radius), (cx, cy + self.radius))
        
        # Draw indicator
        yaw_norm = (angle_yaw / (2 * np.pi)) % 1
        roll_norm = (angle_roll / np.pi)
        dx = (yaw_norm - 0.5) * 2 * self.radius * 0.8
        dy = -(roll_norm - 0.5) * 2 * self.radius * 0.8
        ind_x = int(cx + dx)
        ind_y = int(cy + dy)
        pygame.draw.circle(surface, (100, 200, 255), (ind_x, ind_y), 8)
        pygame.draw.circle(surface, (150, 220, 255), (ind_x, ind_y), 6)
        
        # Draw label
        label = self.font.render("GYRO", True, (200, 200, 200))
        surface.blit(label, (cx - 25, cy - self.radius - 25))
        
        # Draw angle info
        yaw_deg = math.degrees(angle_yaw) % 360
        roll_deg = math.degrees(angle_roll)
        info = self.font_small.render(f"Yaw: {yaw_deg:.0f}°", True, (180, 180, 180))
        surface.blit(info, (cx - 35, cy + self.radius + 10))
        info2 = self.font_small.render(f"Roll: {roll_deg:.0f}°", True, (180, 180, 180))
        surface.blit(info2, (cx - 35, cy + self.radius + 28))


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


# --- Enhanced marker drawing function ---
def draw_markers(surface, markers, points, font):
    """Draw all placed markers on the terrain with special handling for remote markers"""
    px, py = points
    
    for marker in markers:
        i, j = marker['pos']
        mx, my = px[i, j], py[i, j]
        team = marker['team']
        marker_type = marker['type']
        
        # Check if this is a remote marker
        is_remote = marker['id'].startswith('REMOTE-')
        
        info = MARKER_TYPES[team][marker_type]
        color = info['color']
        
        if is_remote:
            # Make remote marker more prominent with pulsing effect
            pulse = int(15 * abs(np.sin(pygame.time.get_ticks() / 300)))
            
            # Larger pulsing circle for remote marker
            pygame.draw.circle(surface, (255, 255, 0), (mx, my), 18 + pulse // 2, 3)
            pygame.draw.circle(surface, color, (mx, my), 14, 2)
            
            # Draw special crosshair
            pygame.draw.line(surface, (255, 255, 0), (mx - 10, my - 10), (mx + 10, my + 10), 4)
            pygame.draw.line(surface, (255, 255, 0), (mx - 10, my + 10), (mx + 10, my - 10), 4)
            
            # Draw "LIVE" label
            label = font.render("LIVE", True, (255, 255, 0))
            label_bg = pygame.Surface((label.get_width() + 8, label.get_height() + 4))
            label_bg.fill((0, 0, 0))
            label_bg.set_alpha(180)
            surface.blit(label_bg, (mx - label.get_width() // 2 - 4, my - 40))
            surface.blit(label, (mx - label.get_width() // 2, my - 38))
            
            # Draw simple star for location
            star_size = 8
            pygame.draw.circle(surface, (255, 255, 100), (mx, my), star_size, 0)
        else:
            # Regular marker drawing
            pygame.draw.line(surface, color, (mx - 6, my - 6), (mx + 6, my + 6), 3)
            pygame.draw.line(surface, color, (mx - 6, my + 6), (mx + 6, my - 6), 3)
            pygame.draw.circle(surface, color, (mx, my), 10, 2)
            symbol = font.render(info['symbol'], True, color)
            surface.blit(symbol, (mx - 8, my - 25))


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


def update_remote_marker(lat, lon):
    """Add or move a special remote marker from external API"""
    global markers, needs_redraw
    
    print(f"🔍 Received remote coords: lat={lat:.5f}, lon={lon:.5f}")
    print(f"   Lat range: {min(lats):.5f} to {max(lats):.5f}")
    print(f"   Lon range: {min(lons):.5f} to {max(lons):.5f}")
    
    # Check if coordinates are in range
    if lat < min(lats) or lat > max(lats):
        print(f"⚠️ WARNING: Latitude {lat:.5f} is OUT OF RANGE!")
        return
    if lon < min(lons) or lon > max(lons):
        print(f"⚠️ WARNING: Longitude {lon:.5f} is OUT OF RANGE!")
        return
    
    i = np.abs(np.array(lats) - lat).argmin()
    j = np.abs(np.array(lons) - lon).argmin()
    marker_id = "REMOTE-0001"
    
    print(f"   Mapped to grid: i={i}, j={j} (out of {len(lats)}x{len(lons)})")
    print(f"   Actual coords: lat={lats[i]:.5f}, lon={lons[j]:.5f}")

    # Remove old remote marker
    markers = [m for m in markers if m["id"] != marker_id]

    # Add new remote marker
    markers.append({
        "id": marker_id,
        "pos": (i, j),
        "type": "5",  # Objective icon
        "team": "friendly"
    })
    
    print(f"✅ Remote marker placed! Total markers: {len(markers)}")
    needs_redraw = True


# --- Start threads ---
live_client = LiveClient(get_live_state)
live_client.start()

remote_fetcher = RemoteMarkerFetcher(REMOTE_URL, update_remote_marker)
remote_fetcher.start()

print(f"📍 Terrain bounds: lat=[{min(lats):.5f}, {max(lats):.5f}], lon=[{min(lons):.5f}, {max(lons):.5f}]")
print("✅ Terrain viewer ready!")

# --- Initial terrain render ---
print("Rendering initial terrain...")
terrain_surface, points = recompute_surface()
needs_redraw = False

running = True
frame_count = 0

while running:
    frame_count += 1
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

    # Check if we need to redraw for remote marker animation
    has_remote_marker = any(m['id'].startswith('REMOTE-') for m in markers)
    
    if needs_redraw:
        terrain_surface, points = recompute_surface()
        needs_redraw = False
    
    # Always render the screen
    screen.fill((0, 0, 0))
    screen.blit(terrain_surface, (0, 0))
    draw_markers(screen, markers, points, font)
    gyro_control.draw(screen, angle_yaw, angle_roll)
    marker_selector.draw(screen)
    
    pygame.display.flip()
    clock.tick(60 if has_remote_marker else 30)  # Higher FPS when animating

live_client.running = False
remote_fetcher.running = False
pygame.quit()
print("👋 Terrain viewer closed")

