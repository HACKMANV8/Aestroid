import pygame
import pandas as pd
import numpy as np
import math
import os

# --- Load terrain data ---
df = pd.read_json("terrain_data.jsonl", lines=True)
df = df[df["elevation"].notnull()]

lats = sorted(df["lat"].unique())
lons = sorted(df["lon"].unique())

# Downsample grid for speed (adjust factor)
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

# Normalize
zmin, zmax = elev_grid.min(), elev_grid.max()
elev_grid_norm = (elev_grid - zmin) / (zmax - zmin)

# Pre-compute colors once
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
        
        # Shading
        if i < len(elev_grid_norm) - 1 and j < len(elev_grid_norm[0]) - 1:
            dzdx = elev_grid_norm[i, j + 1] - elev_grid_norm[i, j]
            dzdy = elev_grid_norm[i + 1, j] - elev_grid_norm[i, j]
            shade = 1 - 0.8 * (dzdx + dzdy)
            color = np.clip(color * shade, 0, 255)
        
        color_grid[i, j] = color.astype(np.uint8)

# --- Load strategic analysis points (CSV -> JSONL, then read JSONL) ---
STRAT_CSV = "strategic_terrain_analysis.csv"
STRAT_JSONL = "strategic_terrain_analysis.jsonl"

def _safe_get(row, keys, default=np.nan):
    for k in keys:
        if k in row and not pd.isna(row[k]):
            return row[k]
    return default

def _nearest_index(sorted_values, value):
    # sorted_values is a list-like sorted ascending
    arr = np.array(sorted_values)
    idx = np.searchsorted(arr, value)
    if idx <= 0:
        return 0
    if idx >= len(arr):
        return len(arr) - 1
    prev_idx = idx - 1
    if abs(arr[idx] - value) < abs(arr[prev_idx] - value):
        return idx
    return prev_idx

# Convert CSV to JSONL before any processing, then load JSONL
strategic_points = []  # Each: {'pos': (i, j), 'category': 'defensive'|'offensive'|'artillery', 'score': float}
try:
    strat_df = None
    if os.path.exists(STRAT_CSV):
        tmp_df = pd.read_csv(STRAT_CSV)
        tmp_df.to_json(STRAT_JSONL, orient='records', lines=True)
        strat_df = pd.read_json(STRAT_JSONL, lines=True)
    elif os.path.exists(STRAT_JSONL):
        strat_df = pd.read_json(STRAT_JSONL, lines=True)

    if strat_df is not None and not strat_df.empty:
        # Column variants
        lat_keys = ["latitude", "lat", "Latitude", "LAT"]
        lon_keys = ["longitude", "lon", "Longitude", "LON"]
        d_key = next((k for k in ["defensive_suitability", "defense", "defensive"] if k in strat_df.columns), None)
        o_key = next((k for k in ["offensive_suitability", "offense", "offensive"] if k in strat_df.columns), None)
        a_key = next((k for k in ["artillery_suitability", "artillery"] if k in strat_df.columns), None)

        for _, row in strat_df.iterrows():
            lat = _safe_get(row, lat_keys)
            lon = _safe_get(row, lon_keys)
            if pd.isna(lat) or pd.isna(lon):
                continue
            i = _nearest_index(lats, float(lat))
            j = _nearest_index(lons, float(lon))

            # Determine category based on highest available score
            candidates = []
            if d_key is not None and not pd.isna(row.get(d_key, np.nan)):
                candidates.append(("defensive", float(row.get(d_key, 0.0))))
            if o_key is not None and not pd.isna(row.get(o_key, np.nan)):
                candidates.append(("offensive", float(row.get(o_key, 0.0))))
            if a_key is not None and not pd.isna(row.get(a_key, np.nan)):
                candidates.append(("artillery", float(row.get(a_key, 0.0))))

            if candidates:
                category, score = max(candidates, key=lambda x: x[1])
            else:
                category, score = ("defensive", 0.0)

            strategic_points.append({"pos": (i, j), "category": category, "score": score})
        # Keep top 10 per category and display all
        if strategic_points:
            def _topk(cat):
                pts = [p for p in strategic_points if p.get("category") == cat]
                pts.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
                return pts[:10]
            defensive_top = _topk("defensive")
            offensive_top = _topk("offensive")
            artillery_top = _topk("artillery")
            strategic_points = defensive_top + offensive_top + artillery_top
except Exception as _e:
    # If anything goes wrong, we simply skip strategic overlay
    strategic_points = []

# --- Pygame setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Terrain Viewer with Gyro Controls")
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

# --- Parameters ---
SCALE_X, SCALE_Y, HEIGHT_SCALE = 10, 10, 250
angle_x, angle_y = np.radians(45), np.radians(35)
mid_x, mid_y = len(lons) / 2, len(lats) / 2

offset_x, offset_y = WIDTH // 2, HEIGHT // 2
zoom = 1.0
markers = []  # Each marker: {'pos': (i, j), 'type': '1'}

# Strategic overlay styling
STRAT_COLORS = {
    "defensive": (0, 220, 120),
    "offensive": (240, 80, 80),
    "artillery": (255, 215, 0),
}
STRAT_ABBR = {
    "defensive": "DEF",
    "offensive": "OFF",
    "artillery": "ART",
}

# Mouse drag state
dragging = False
gyro_dragging = False
last_mouse_pos = (0, 0)

# Gyro control
GYRO_CENTER = (100, 100)
GYRO_RADIUS = 70

# Cache
terrain_surface = None
points = None
needs_redraw = True

# --- Helpers ---
def project_vectorized():
    """Vectorized projection for all points at once"""
    i_grid, j_grid = np.meshgrid(range(len(lats)), range(len(lons)), indexing='ij')
    z = elev_grid_norm
    
    px = ((j_grid - mid_x) * SCALE_X * np.cos(angle_x) + 
          (i_grid - mid_y) * SCALE_Y * np.cos(angle_x)) * zoom
    py = ((j_grid + mid_x - i_grid - mid_y) * SCALE_Y * np.sin(angle_y) - 
          z * HEIGHT_SCALE) * zoom
    
    return (offset_x + px).astype(int), (offset_y + py).astype(int)

def recompute_surface():
    """Draw terrain mesh using vectorized operations"""
    surf = pygame.Surface((WIDTH, HEIGHT))
    surf.fill((15, 15, 25))
    
    px, py = project_vectorized()
    
    # Draw polygons
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            pts = [(px[i, j], py[i, j]), 
                   (px[i, j + 1], py[i, j + 1]),
                   (px[i + 1, j + 1], py[i + 1, j + 1]), 
                   (px[i + 1, j], py[i + 1, j])]
            pygame.draw.polygon(surf, tuple(color_grid[i, j]), pts)
    
    return surf, (px, py)

def draw_gyro(surface):
    """Draw gyro control interface"""
    cx, cy = GYRO_CENTER
    
    # Background circle
    pygame.draw.circle(surface, (40, 40, 50), GYRO_CENTER, GYRO_RADIUS + 5)
    pygame.draw.circle(surface, (25, 25, 35), GYRO_CENTER, GYRO_RADIUS)
    
    # Cross lines
    pygame.draw.line(surface, (60, 60, 70), (cx - GYRO_RADIUS, cy), (cx + GYRO_RADIUS, cy), 1)
    pygame.draw.line(surface, (60, 60, 70), (cx, cy - GYRO_RADIUS), (cx, cy + GYRO_RADIUS), 1)
    
    # Calculate position based on angles
    yaw_normalized = (angle_x / (2 * np.pi)) % 1
    pitch_normalized = angle_y / np.pi
    
    dx = (yaw_normalized - 0.5) * 2 * GYRO_RADIUS * 0.8
    dy = -(pitch_normalized - 0.5) * 2 * GYRO_RADIUS * 0.8
    
    ind_x = int(cx + dx)
    ind_y = int(cy + dy)
    
    pygame.draw.circle(surface, (100, 200, 255), (ind_x, ind_y), 8)
    pygame.draw.circle(surface, (150, 220, 255), (ind_x, ind_y), 6)
    
    label = font.render("GYRO", True, (200, 200, 200))
    surface.blit(label, (cx - 25, cy - GYRO_RADIUS - 25))
    
    yaw_deg = math.degrees(angle_x) % 360
    pitch_deg = math.degrees(angle_y)
    info = font_small.render(f"Yaw: {yaw_deg:.0f}°", True, (180, 180, 180))
    surface.blit(info, (cx - 35, cy + GYRO_RADIUS + 10))
    info2 = font_small.render(f"Pitch: {pitch_deg:.0f}°", True, (180, 180, 180))
    surface.blit(info2, (cx - 35, cy + GYRO_RADIUS + 28))

def draw_hud(surface):
    """Draw HUD with marker selector and info"""
    # Semi-transparent background for HUD
    hud_width = 280
    hud_height = 280
    hud_x = WIDTH - hud_width - 20
    hud_y = 20
    
    hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
    pygame.draw.rect(hud_surf, (20, 20, 30, 200), (0, 0, hud_width, hud_height), border_radius=10)
    pygame.draw.rect(hud_surf, (60, 60, 80, 200), (0, 0, hud_width, hud_height), 2, border_radius=10)
    
    # Title
    title = font.render("MARKER SELECTOR", True, (200, 200, 200))
    hud_surf.blit(title, (10, 10))
    
    # Marker type buttons
    y_offset = 45
    for key, marker_info in MARKER_TYPES.items():
        # Button background
        btn_rect = pygame.Rect(10, y_offset, hud_width - 20, 30)
        if key == current_marker_type:
            pygame.draw.rect(hud_surf, (80, 80, 100, 220), btn_rect, border_radius=5)
            pygame.draw.rect(hud_surf, marker_info['color'], btn_rect, 2, border_radius=5)
        else:
            pygame.draw.rect(hud_surf, (40, 40, 50, 180), btn_rect, border_radius=5)
        
        # Key number
        key_text = font.render(f"[{key}]", True, (150, 150, 150))
        hud_surf.blit(key_text, (15, y_offset + 5))
        
        # Symbol and name
        symbol = font.render(marker_info['symbol'], True, marker_info['color'])
        hud_surf.blit(symbol, (60, y_offset + 3))
        
        name = font.render(marker_info['name'], True, (200, 200, 200))
        hud_surf.blit(name, (85, y_offset + 5))
        
        y_offset += 35
    
    # Instructions
    y_offset += 10
    inst = font_small.render("Right-Click: Place marker", True, (150, 150, 150))
    hud_surf.blit(inst, (10, y_offset))
    inst2 = font_small.render("Del: Remove last marker", True, (150, 150, 150))
    hud_surf.blit(inst2, (10, y_offset + 18))
    
    surface.blit(hud_surf, (hud_x, hud_y))
    
    # Stats panel
    stats_y = hud_y + hud_height + 20
    stats_surf = pygame.Surface((hud_width, 140), pygame.SRCALPHA)
    pygame.draw.rect(stats_surf, (20, 20, 30, 200), (0, 0, hud_width, 140), border_radius=10)
    pygame.draw.rect(stats_surf, (60, 60, 80, 200), (0, 0, hud_width, 140), 2, border_radius=10)
    
    stats_title = font.render("STATS", True, (200, 200, 200))
    stats_surf.blit(stats_title, (10, 10))
    
    zoom_text = font_small.render(f"Zoom: {zoom:.2f}x", True, (180, 180, 180))
    stats_surf.blit(zoom_text, (10, 40))
    
    marker_count = len(markers)
    marker_text = font_small.render(f"Markers: {marker_count}", True, (180, 180, 180))
    stats_surf.blit(marker_text, (10, 60))
    
    # Strategic overlay legend
    try:
        def_count = sum(1 for p in strategic_points if p['category'] == 'defensive')
        off_count = sum(1 for p in strategic_points if p['category'] == 'offensive')
        art_count = sum(1 for p in strategic_points if p['category'] == 'artillery')

        y0 = 80
        # Defensive square swatch
        pygame.draw.rect(stats_surf, STRAT_COLORS["defensive"], (10, y0 + 2, 12, 12))
        stats_surf.blit(font_small.render(f"Defensive: {def_count}", True, (200, 200, 200)), (28, y0))
        # Offensive triangle swatch
        tri = [(16, y0 + 22), (10, y0 + 36), (22, y0 + 36)]
        pygame.draw.polygon(stats_surf, STRAT_COLORS["offensive"], tri)
        stats_surf.blit(font_small.render(f"Offensive: {off_count}", True, (200, 200, 200)), (28, y0 + 28))
        # Artillery circle + cross swatch
        pygame.draw.circle(stats_surf, STRAT_COLORS["artillery"], (16, y0 + 56), 6)
        pygame.draw.line(stats_surf, (0, 0, 0), (8, y0 + 56), (24, y0 + 56), 2)
        pygame.draw.line(stats_surf, (0, 0, 0), (16, y0 + 48), (16, y0 + 64), 2)
        stats_surf.blit(font_small.render(f"Artillery: {art_count}", True, (200, 200, 200)), (28, y0 + 52))
    except Exception:
        pass

    surface.blit(stats_surf, (hud_x, stats_y))

def is_in_gyro(pos):
    """Check if position is inside gyro control"""
    dx = pos[0] - GYRO_CENTER[0]
    dy = pos[1] - GYRO_CENTER[1]
    return dx*dx + dy*dy <= GYRO_RADIUS*GYRO_RADIUS

def update_angles_from_gyro(pos):
    """Update angles based on gyro position"""
    global angle_x, angle_y
    
    dx = pos[0] - GYRO_CENTER[0]
    dy = pos[1] - GYRO_CENTER[1]
    
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > GYRO_RADIUS * 0.8:
        scale = GYRO_RADIUS * 0.8 / dist
        dx *= scale
        dy *= scale
    
    yaw_normalized = (dx / (GYRO_RADIUS * 0.8)) / 2 + 0.5
    pitch_normalized = -(dy / (GYRO_RADIUS * 0.8)) / 2 + 0.5
    
    angle_x = yaw_normalized * 2 * np.pi
    angle_y = np.clip(pitch_normalized * np.pi, 0, np.pi)

terrain_surface, points = recompute_surface()
needs_redraw = False

# --- Main loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left click
                if is_in_gyro(event.pos):
                    gyro_dragging = True
                    update_angles_from_gyro(event.pos)
                    needs_redraw = True
                else:
                    dragging = True
                    last_mouse_pos = event.pos
            elif event.button == 4:  # scroll up
                zoom *= 1.1
                needs_redraw = True
            elif event.button == 5:  # scroll down
                zoom /= 1.1
                needs_redraw = True
            elif event.button == 3:  # right click -> marker
                mx, my = event.pos
                px, py = points
                dist = (px - mx)**2 + (py - my)**2
                best_idx = np.unravel_index(np.argmin(dist), dist.shape)
                i, j = best_idx
                markers.append({'pos': best_idx, 'type': current_marker_type})
                marker_info = MARKER_TYPES[current_marker_type]
                print(f"📍 {marker_info['name']} marker at lat={lats[i]:.6f}, lon={lons[j]:.6f}, elev={elev_grid[i,j]:.1f}")

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
            if event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                zoom *= 1.1
                needs_redraw = True
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                zoom /= 1.1
                needs_redraw = True
            elif event.key == pygame.K_LEFT:
                angle_x -= np.radians(5)
                needs_redraw = True
            elif event.key == pygame.K_RIGHT:
                angle_x += np.radians(5)
                needs_redraw = True
            elif event.key == pygame.K_UP:
                angle_y = max(0, angle_y - np.radians(5))
                needs_redraw = True
            elif event.key == pygame.K_DOWN:
                angle_y = min(np.pi, angle_y + np.radians(5))
                needs_redraw = True
            elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                if markers:
                    markers.pop()
                    print("Removed last marker")
            # Number keys to select marker type
            elif event.unicode in MARKER_TYPES:
                current_marker_type = event.unicode
                print(f"Selected marker type: {MARKER_TYPES[current_marker_type]['name']}")

    # Redraw only when needed
    if needs_redraw:
        terrain_surface, points = recompute_surface()
        needs_redraw = False

    # Draw
    screen.fill((0, 0, 0))
    screen.blit(terrain_surface, (0, 0))

    # Draw markers with different colors/symbols
    px, py = points
    for marker in markers:
        i, j = marker['pos']
        mx, my = px[i, j], py[i, j]
        marker_info = MARKER_TYPES[marker['type']]
        color = marker_info['color']
        
        # Draw X marker
        pygame.draw.line(screen, color, (mx - 6, my - 6), (mx + 6, my + 6), 3)
        pygame.draw.line(screen, color, (mx - 6, my + 6), (mx + 6, my - 6), 3)
        
        # Draw circle around it
        pygame.draw.circle(screen, color, (mx, my), 10, 2)
        
        # Draw symbol above
        symbol_text = font.render(marker_info['symbol'], True, color)
        screen.blit(symbol_text, (mx - 8, my - 25))

    # Draw strategic analysis points overlay (high-visibility, category-distinct)
    if strategic_points:
        for sp in strategic_points:
            i, j = sp['pos']
            mx, my = px[i, j], py[i, j]
            cat = sp['category']
            color = STRAT_COLORS.get(cat, (255, 255, 255))
            score = float(sp.get('score', 0.0))
            base = 6 + int(min(6, score * 6))
            x, y = int(mx), int(my)

            # Halo for contrast
            pygame.draw.circle(screen, (0, 0, 0), (x, y), base + 3)
            pygame.draw.circle(screen, (255, 255, 255), (x, y), base + 1)

            # Category-specific shape
            if cat == 'defensive':
                r = base
                pygame.draw.rect(screen, color, (x - r, y - r, 2 * r, 2 * r))
                pygame.draw.rect(screen, (0, 0, 0), (x - r, y - r, 2 * r, 2 * r), 2)
            elif cat == 'offensive':
                r = base + 2
                pts = [(x, y - r), (x - r, y + r), (x + r, y + r)]
                pygame.draw.polygon(screen, color, pts)
                pygame.draw.polygon(screen, (0, 0, 0), pts, 2)
            else:  # artillery
                r = base + 1
                pygame.draw.circle(screen, color, (x, y), r)
                pygame.draw.circle(screen, (0, 0, 0), (x, y), r, 2)
                pygame.draw.line(screen, (0, 0, 0), (x - r - 2, y), (x + r + 2, y), 2)
                pygame.draw.line(screen, (0, 0, 0), (x, y - r - 2), (x, y + r + 2), 2)

            # Elevation label with category
            elev_val = float(elev_grid[i, j]) if 0 <= i < elev_grid.shape[0] and 0 <= j < elev_grid.shape[1] else float('nan')
            label_text = f"{STRAT_ABBR.get(cat, '').upper()} {elev_val:.0f}m"
            label_surface = font_small.render(label_text, True, (10, 10, 10))
            lw, lh = label_surface.get_width(), label_surface.get_height()
            bg_rect = pygame.Rect(x + 10, y - lh - 8, lw + 8, lh + 6)
            pygame.draw.rect(screen, (245, 245, 245), bg_rect)
            pygame.draw.rect(screen, color, bg_rect, 2)
            screen.blit(label_surface, (bg_rect.x + 4, bg_rect.y + 3))

    # Draw gyro interface
    draw_gyro(screen)
    
    # Draw HUD
    draw_hud(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

