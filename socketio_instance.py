"""SocketIO instance to avoid circular imports"""
from flask_socketio import SocketIO

socketio = None

def init_socketio(app):
    """Initialize SocketIO instance"""
    global socketio
    # Try to use eventlet, fallback to threading if not available
    try:
        import eventlet
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    except:
        # Fallback to threading mode if eventlet is not available
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    return socketio

def get_socketio():
    """Get SocketIO instance"""
    return socketio

