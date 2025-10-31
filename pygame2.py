import pygame
import pandas as pd
import numpy as np
import math

# --- Load terrain data ---
df = pd.read_json("terrain_data.jsonl", lines=True)
df = df[df["elevation"].notnull()]

lats = sorted(df["lat"].unique())
lons = sorted(df["lon"].unique())

# Downsample for performance
DOWNSAMPLE = 2
lats = lats[::DOWNSAMPLE]
lons = lons[::DOWNSAMPLE]

elev_grid = np.zeros((len(lats), len(lons)))
ndvi_grid = np.zeros((len(lats), len(lons)))
ndwi_grid = np.zeros((len(lats), len(lons)))

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        cell = df[(df["lat"] == lat) & (df["lon"] == lon)].iloc[0]
        elev_grid[i, j] = cell["elevation"]
        ndvi_grid[i, j] = cell.get("ndvi", np.nan)
        ndwi_grid[i, j] = cell.get("ndwi", np.nan)

# Normalize elevation
zmin, zmax = elev_grid.min(), elev_grid.max()
elev_grid_norm = (elev_grid - zmin) / (zmax - zmin)

# Compute terrain colors
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
        # Simple shading
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
pygame.display.set_caption("3D Terrain Viewer (Yaw + Roll)")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 18)

# --- Marker types ---
MARKER_TYPES = {
    '1': {'name': 'Tank', 'color': (255, 0, 0), 'symbol': '▲'},
    '2': {'name': 'Soldier', 'color': (0, 150, 255), 'symbol': '●'},
    '3': {'name': 'Vehicle', 'color': (255, 255, 0), 'symbol': '■'},
    '4': {'name': 'Building', 'color': (150, 150, 150), 'symbol': '▪'},
    '5': {'name': 'Objective', 'color': (0, 255, 0), 'symbol': '★'},
    '6': {'name': 'Hazard', 'color': (255, 128, 0), 'symbol': '⚠'},
}
current_marker_type = '1'
markers = []

# --- View parameters ---
SCALE_X, SCALE_Y, HEIGHT_SCALE = 10, 10, 250
angle_yaw = np.radians(45)
angle_roll = np.radians(20)
mid_x, mid_y = len(lons) / 2, len(lats) / 2

offset_x, offset_y = WIDTH // 2, HEIGHT // 2
zoom = 1.0

# Mouse + gyro
dragging = False
gyro_dragging = False
last_mouse_pos = (0, 0)
GYRO_CENTER = (100, 100)
GYRO_RADIUS = 70

terrain_surface = None
points = None
needs_redraw = True

# --- Projection ---
def project_vectorized():
    """Vectorized projection using yaw + roll"""
    i_grid, j_grid = np.meshgrid(range(len(lats)), range(len(lons)), indexing='ij')
    z = elev_grid_norm * HEIGHT_SCALE
    x = (j_grid - mid_x) * SCALE_X
    y = (i_grid - mid_y) * SCALE_Y

    # Yaw rotation
    x2 = x * np.cos(angle_yaw) - y * np.sin(angle_yaw)
    y2 = x * np.sin(angle_yaw) + y * np.cos(angle_yaw)

    # Roll rotation (around forward axis)
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

# --- Gyro ---
def draw_gyro(surface):
    cx, cy = GYRO_CENTER
    pygame.draw.circle(surface, (40, 40, 50), GYRO_CENTER, GYRO_RADIUS + 5)
    pygame.draw.circle(surface, (25, 25, 35), GYRO_CENTER, GYRO_RADIUS)
    pygame.draw.line(surface, (60, 60, 70), (cx - GYRO_RADIUS, cy), (cx + GYRO_RADIUS, cy))
    pygame.draw.line(surface, (60, 60, 70), (cx, cy - GYRO_RADIUS), (cx, cy + GYRO_RADIUS))

    yaw_norm = (angle_yaw / (2 * np.pi)) % 1
    roll_norm = (angle_roll / np.pi)
    dx = (yaw_norm - 0.5) * 2 * GYRO_RADIUS * 0.8
    dy = -(roll_norm - 0.5) * 2 * GYRO_RADIUS * 0.8
    ind_x = int(cx + dx)
    ind_y = int(cy + dy)
    pygame.draw.circle(surface, (100, 200, 255), (ind_x, ind_y), 8)
    pygame.draw.circle(surface, (150, 220, 255), (ind_x, ind_y), 6)

    label = font.render("GYRO", True, (200, 200, 200))
    surface.blit(label, (cx - 25, cy - GYRO_RADIUS - 25))
    yaw_deg = math.degrees(angle_yaw) % 360
    roll_deg = math.degrees(angle_roll)
    info = font_small.render(f"Yaw: {yaw_deg:.0f}°", True, (180, 180, 180))
    surface.blit(info, (cx - 35, cy + GYRO_RADIUS + 10))
    info2 = font_small.render(f"Roll: {roll_deg:.0f}°", True, (180, 180, 180))
    surface.blit(info2, (cx - 35, cy + GYRO_RADIUS + 28))

def is_in_gyro(pos):
    dx = pos[0] - GYRO_CENTER[0]
    dy = pos[1] - GYRO_CENTER[1]
    return dx * dx + dy * dy <= GYRO_RADIUS * GYRO_RADIUS

def update_angles_from_gyro(pos):
    global angle_yaw, angle_roll
    dx = pos[0] - GYRO_CENTER[0]
    dy = pos[1] - GYRO_CENTER[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > GYRO_RADIUS * 0.8:
        s = GYRO_RADIUS * 0.8 / dist
        dx *= s
        dy *= s
    yaw_norm = (dx / (GYRO_RADIUS * 0.8)) / 2 + 0.5
    roll_norm = -(dy / (GYRO_RADIUS * 0.8)) / 2 + 0.5
    angle_yaw = yaw_norm * 2 * np.pi
    angle_roll = np.clip(roll_norm * np.pi, 0, np.pi)

# --- HUD ---
def draw_hud(surface):
    hud_width, hud_height = 280, 280
    hud_x, hud_y = WIDTH - hud_width - 20, 20
    hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
    pygame.draw.rect(hud_surf, (20, 20, 30, 200), (0, 0, hud_width, hud_height), border_radius=10)
    pygame.draw.rect(hud_surf, (60, 60, 80, 200), (0, 0, hud_width, hud_height), 2, border_radius=10)
    title = font.render("MARKER SELECTOR", True, (200, 200, 200))
    hud_surf.blit(title, (10, 10))
    y_offset = 45
    for key, marker_info in MARKER_TYPES.items():
        btn_rect = pygame.Rect(10, y_offset, hud_width - 20, 30)
        if key == current_marker_type:
            pygame.draw.rect(hud_surf, (80, 80, 100, 220), btn_rect, border_radius=5)
            pygame.draw.rect(hud_surf, marker_info['color'], btn_rect, 2, border_radius=5)
        else:
            pygame.draw.rect(hud_surf, (40, 40, 50, 180), btn_rect, border_radius=5)
        hud_surf.blit(font.render(f"[{key}]", True, (150, 150, 150)), (15, y_offset + 5))
        hud_surf.blit(font.render(marker_info['symbol'], True, marker_info['color']), (60, y_offset + 3))
        hud_surf.blit(font.render(marker_info['name'], True, (200, 200, 200)), (85, y_offset + 5))
        y_offset += 35
    y_offset += 10
    hud_surf.blit(font_small.render("Right-Click: Place marker", True, (150, 150, 150)), (10, y_offset))
    hud_surf.blit(font_small.render("Del: Remove last marker", True, (150, 150, 150)), (10, y_offset + 18))
    surface.blit(hud_surf, (hud_x, hud_y))

terrain_surface, points = recompute_surface()
needs_redraw = False

# --- Main loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if is_in_gyro(event.pos):
                    gyro_dragging = True
                    update_angles_from_gyro(event.pos)
                    needs_redraw = True
                else:
                    dragging = True
                    last_mouse_pos = event.pos
            elif event.button == 3:  # Right-click = place marker
                mx, my = event.pos
                px, py = points
                dist = (px - mx) ** 2 + (py - my) ** 2
                best_idx = np.unravel_index(np.argmin(dist), dist.shape)
                i, j = best_idx
                markers.append({'pos': best_idx, 'type': current_marker_type})
                info = MARKER_TYPES[current_marker_type]['name']
                print(f"📍 Placed {info} at lat={lats[i]:.6f}, lon={lons[j]:.6f}, elev={elev_grid[i,j]:.1f}")
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
                update_angles_from_gyro(event.pos)
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
            elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                if markers:
                    markers.pop()
                    print("🗑️ Removed last marker")
                    needs_redraw = True
            elif event.unicode in MARKER_TYPES:
                current_marker_type = event.unicode
                print(f"Selected marker: {MARKER_TYPES[current_marker_type]['name']}")

    if needs_redraw:
        terrain_surface, points = recompute_surface()
        needs_redraw = False

    # Draw terrain
    screen.fill((0, 0, 0))
    screen.blit(terrain_surface, (0, 0))

    # Draw markers
    px, py = points
    for marker in markers:
        i, j = marker['pos']
        mx, my = px[i, j], py[i, j]
        info = MARKER_TYPES[marker['type']]
        color = info['color']
        pygame.draw.line(screen, color, (mx - 6, my - 6), (mx + 6, my + 6), 3)
        pygame.draw.line(screen, color, (mx - 6, my + 6), (mx + 6, my - 6), 3)
        pygame.draw.circle(screen, color, (mx, my), 10, 2)
        symbol = font.render(info['symbol'], True, color)
        screen.blit(symbol, (mx - 8, my - 25))

    draw_gyro(screen)
    draw_hud(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()


