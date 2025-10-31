import pygame
import numpy as np
import math
import json
from datetime import datetime

# --- Marker types configuration ---
MARKER_TYPES = {
    'friendly': {
        '1': {'name': 'Tank', 'color': (0, 150, 255), 'symbol': '▲'},
        '2': {'name': 'Soldier', 'color': (100, 200, 255), 'symbol': '●'},
        '3': {'name': 'Vehicle', 'color': (150, 220, 255), 'symbol': '■'},
        '4': {'name': 'Building', 'color': (80, 180, 230), 'symbol': '▪'},
        '5': {'name': 'Objective', 'color': (0, 255, 150), 'symbol': '★'},
        '6': {'name': 'Medical', 'color': (0, 200, 100), 'symbol': '✚'},
    },
    'enemy': {
        '1': {'name': 'Tank', 'color': (255, 0, 0), 'symbol': '▲'},
        '2': {'name': 'Soldier', 'color': (255, 100, 100), 'symbol': '●'},
        '3': {'name': 'Vehicle', 'color': (255, 150, 0), 'symbol': '■'},
        '4': {'name': 'Building', 'color': (200, 80, 80), 'symbol': '▪'},
        '5': {'name': 'Objective', 'color': (255, 50, 50), 'symbol': '★'},
        '6': {'name': 'Hazard', 'color': (255, 128, 0), 'symbol': '⚠'},
    }
}

# --- Gyro control configuration ---
GYRO_CENTER = (100, 100)
GYRO_RADIUS = 70


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


class MarkerSelector:
    """HUD panel for selecting marker types"""
    
    def __init__(self, width, height, x, y, font, font_small):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.font = font
        self.font_small = font_small
        self.current_marker_type = '1'
        self.current_team = 'friendly'  # 'friendly' or 'enemy'
        self.marker_counter = 0
    
    def set_marker_type(self, marker_type):
        """Set the current marker type"""
        if marker_type in MARKER_TYPES[self.current_team]:
            self.current_marker_type = marker_type
            return True
        return False
    
    def toggle_team(self):
        """Toggle between friendly and enemy teams"""
        self.current_team = 'enemy' if self.current_team == 'friendly' else 'friendly'
        # Reset to marker type 1 if current type doesn't exist in new team
        if self.current_marker_type not in MARKER_TYPES[self.current_team]:
            self.current_marker_type = '1'
    
    def get_current_type(self):
        """Get the current marker type"""
        return self.current_marker_type
    
    def get_current_team(self):
        """Get the current team"""
        return self.current_team
    
    def generate_marker_id(self):
        """Generate a unique marker ID"""
        self.marker_counter += 1
        return f"{self.current_team[0].upper()}-{self.marker_counter:04d}"
    
    def is_over_button(self, pos, button_rect):
        """Check if position is over a button (accounting for HUD offset)"""
        adjusted_rect = button_rect.move(self.x, self.y)
        return adjusted_rect.collidepoint(pos)
    
    def draw(self, surface):
        """Draw the marker selector HUD"""
        hud_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw background
        pygame.draw.rect(hud_surf, (20, 20, 30, 200), (0, 0, self.width, self.height), border_radius=10)
        pygame.draw.rect(hud_surf, (60, 60, 80, 200), (0, 0, self.width, self.height), 2, border_radius=10)
        
        # Draw title
        title = self.font.render("MARKER SELECTOR", True, (200, 200, 200))
        hud_surf.blit(title, (10, 10))
        
        # Draw team toggle buttons
        friendly_rect = pygame.Rect(10, 35, 125, 25)
        enemy_rect = pygame.Rect(145, 35, 125, 25)
        
        # Friendly button
        if self.current_team == 'friendly':
            pygame.draw.rect(hud_surf, (0, 100, 200, 220), friendly_rect, border_radius=5)
            pygame.draw.rect(hud_surf, (0, 150, 255), friendly_rect, 2, border_radius=5)
        else:
            pygame.draw.rect(hud_surf, (40, 40, 50, 180), friendly_rect, border_radius=5)
        friendly_text = self.font_small.render("FRIENDLY [F]", True, (200, 200, 200))
        hud_surf.blit(friendly_text, (15, 40))
        
        # Enemy button
        if self.current_team == 'enemy':
            pygame.draw.rect(hud_surf, (150, 0, 0, 220), enemy_rect, border_radius=5)
            pygame.draw.rect(hud_surf, (255, 0, 0), enemy_rect, 2, border_radius=5)
        else:
            pygame.draw.rect(hud_surf, (40, 40, 50, 180), enemy_rect, border_radius=5)
        enemy_text = self.font_small.render("ENEMY [E]", True, (200, 200, 200))
        hud_surf.blit(enemy_text, (160, 40))
        
        # Store button rects for click detection
        self.friendly_rect = friendly_rect
        self.enemy_rect = enemy_rect
        
        # Draw marker type buttons
        y_offset = 70
        marker_types = MARKER_TYPES[self.current_team]
        for key, marker_info in marker_types.items():
            btn_rect = pygame.Rect(10, y_offset, self.width - 20, 30)
            
            if key == self.current_marker_type:
                pygame.draw.rect(hud_surf, (80, 80, 100, 220), btn_rect, border_radius=5)
                pygame.draw.rect(hud_surf, marker_info['color'], btn_rect, 2, border_radius=5)
            else:
                pygame.draw.rect(hud_surf, (40, 40, 50, 180), btn_rect, border_radius=5)
            
            hud_surf.blit(self.font.render(f"[{key}]", True, (150, 150, 150)), (15, y_offset + 5))
            hud_surf.blit(self.font.render(marker_info['symbol'], True, marker_info['color']), (60, y_offset + 3))
            hud_surf.blit(self.font.render(marker_info['name'], True, (200, 200, 200)), (85, y_offset + 5))
            y_offset += 35
        
        # Draw instructions
        y_offset += 10
        hud_surf.blit(self.font_small.render("Right-Click: Place marker", True, (150, 150, 150)), (10, y_offset))
        hud_surf.blit(self.font_small.render("Del: Remove last marker", True, (150, 150, 150)), (10, y_offset + 18))
        hud_surf.blit(self.font_small.render("Ctrl+S: Export markers", True, (150, 150, 150)), (10, y_offset + 36))
        
        surface.blit(hud_surf, (self.x, self.y))


def draw_markers(surface, markers, points, font):
    """Draw all placed markers on the terrain"""
    px, py = points
    
    for marker in markers:
        i, j = marker['pos']
        mx, my = px[i, j], py[i, j]
        team = marker['team']
        marker_type = marker['type']
        info = MARKER_TYPES[team][marker_type]
        color = info['color']
        
        # Draw crosshair
        pygame.draw.line(surface, color, (mx - 6, my - 6), (mx + 6, my + 6), 3)
        pygame.draw.line(surface, color, (mx - 6, my + 6), (mx + 6, my - 6), 3)
        
        # Draw circle
        pygame.draw.circle(surface, color, (mx, my), 10, 2)
        
        # Draw symbol
        symbol = font.render(info['symbol'], True, color)
        surface.blit(symbol, (mx - 8, my - 25))


def export_markers_to_json(markers, lats, lons, elev_grid, filename=None):
    """Export markers to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"markers_export_{timestamp}.json"
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "marker_count": len(markers),
        "markers": []
    }
    
    for marker in markers:
        i, j = marker['pos']
        team = marker['team']
        marker_type = marker['type']
        marker_info = MARKER_TYPES[team][marker_type]
        
        marker_data = {
            "id": marker['id'],
            "team": team,
            "type": marker_type,
            "type_name": marker_info['name'],
            "latitude": float(lats[i]),
            "longitude": float(lons[j]),
            "elevation": float(elev_grid[i, j]),
            "color_rgb": marker_info['color'],
            "symbol": marker_info['symbol']
        }
        export_data["markers"].append(marker_data)
    
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Exported {len(markers)} markers to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error exporting markers: {e}")
        return False

