from flask import Flask
from flask_cors import CORS
from flask_socketio import emit
from config import Config
from database import db
from socketio_instance import init_socketio, get_socketio

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize SocketIO for real-time updates
socketio = init_socketio(app)

# Initialize database indexes before first request
with app.app_context():
    db.create_indexes()

# Register blueprints (import after socketio initialization to avoid circular import)
from routes import routes
app.register_blueprint(routes)

# Serve static HTML file for testing WebSocket updates
@app.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('static', 'index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to chatbot backend'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('subscribe_updates')
def handle_subscribe():
    """Handle client subscription to updates"""
    emit('subscribed', {'data': 'Subscribed to response updates'})

if __name__ == '__main__':
    # Run the Flask application with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

