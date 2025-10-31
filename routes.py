from flask import Blueprint, request, jsonify
from flask_socketio import emit
import uuid
from datetime import datetime
from database import db
from groq_service import GroqChatbotService
import json

routes = Blueprint('routes', __name__)
groq_service = GroqChatbotService()

@routes.route('/api/chatbot/query', methods=['POST'])
def chatbot_query():
    """Endpoint to get chatbot response"""
    try:
        data = request.get_json()
        user_input = data.get('query', '')
        
        if not user_input:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get response from chatbot
        response_text, conversation_data = groq_service.get_chatbot_response(user_input)
        
        # Generate response ID
        response_id = str(uuid.uuid4())
        
        # Store in database
        response_data = {
            'response_id': response_id,
            'query': user_input,
            'response': response_text,
            'timestamp': datetime.utcnow().isoformat(),
            'messages': conversation_data.get('messages', []) if conversation_data else [],
            'location': None,
            'mobile_timestamp': None,
            'updated_at': None
        }
        
        db.insert_response(response_data)
        
        # Emit real-time update via SocketIO
        from socketio_instance import get_socketio
        socketio = get_socketio()
        socketio.emit('response_updated', {
            'response_id': response_id,
            'response': response_text,
            'timestamp': response_data['timestamp']
        }, namespace='/')
        
        return jsonify({
            'response_id': response_id,
            'response': response_text,
            'timestamp': response_data['timestamp']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/api/mobile/update', methods=['POST'])
def mobile_update():
    """Endpoint for mobile app to update location and timestamp"""
    try:
        data = request.get_json()
        response_id = data.get('response_id')
        location = data.get('location')
        timestamp = data.get('timestamp')
        
        if not response_id:
            return jsonify({'error': 'response_id is required'}), 400
        
        # Update response with location and timestamp
        update_data = {
            'location': location,
            'mobile_timestamp': timestamp,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        result = db.update_with_location(response_id, location, timestamp)
        
        if result.matched_count == 0:
            return jsonify({'error': 'Response not found'}), 404
        
        # Get updated response
        updated_response = db.get_response(response_id)
        
        # Emit real-time update
        from socketio_instance import get_socketio
        socketio = get_socketio()
        socketio.emit('response_updated', {
            'response_id': response_id,
            'response': updated_response.get('response'),
            'location': location,
            'timestamp': updated_response.get('updated_at'),
            'mobile_timestamp': timestamp
        }, namespace='/')
        
        return jsonify({
            'success': True,
            'response_id': response_id,
            'updated_response': updated_response
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/api/pygame/update', methods=['POST'])
def pygame_update():
    """Endpoint for pygame application to send updates"""
    try:
        data = request.get_json()
        update_data = data.get('update_data')
        response_id = data.get('response_id')  # Optional: to continue conversation
        
        if not update_data:
            return jsonify({'error': 'update_data is required'}), 400
        
        # Process update with chatbot
        new_response_text, conversation_data = groq_service.process_update(update_data, response_id)
        
        # Generate new response ID if not provided
        if not response_id:
            new_response_id = str(uuid.uuid4())
        else:
            new_response_id = response_id
        
        # Store or update in database
        response_data = {
            'response_id': new_response_id,
            'query': json.dumps(update_data),
            'response': new_response_text,
            'timestamp': datetime.utcnow().isoformat(),
            'messages': conversation_data.get('messages', []) if conversation_data else [],
            'update_source': 'pygame',
            'location': None,
            'mobile_timestamp': None,
            'updated_at': None
        }
        
        if response_id:
            # Update existing response
            db.update_response(response_id, response_data)
        else:
            # Insert new response
            db.insert_response(response_data)
        
        # Emit real-time update
        from socketio_instance import get_socketio
        socketio = get_socketio()
        socketio.emit('response_updated', {
            'response_id': new_response_id,
            'response': new_response_text,
            'timestamp': response_data['timestamp'],
            'update_source': 'pygame'
        }, namespace='/')
        
        return jsonify({
            'response_id': new_response_id,
            'response': new_response_text,
            'timestamp': response_data['timestamp']
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/api/responses', methods=['GET'])
def get_all_responses():
    """Get all responses from database"""
    try:
        responses = db.get_all_responses()
        # Convert ObjectId to string for JSON serialization
        for resp in responses:
            resp['_id'] = str(resp['_id'])
            # Remove messages if not needed for list view
            if 'messages' in resp:
                del resp['messages']
        return jsonify({'responses': responses}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/api/responses/<response_id>', methods=['GET'])
def get_response(response_id):
    """Get a specific response by ID"""
    try:
        response = db.get_response(response_id)
        if not response:
            return jsonify({'error': 'Response not found'}), 404
        
        response['_id'] = str(response['_id'])
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'chatbot-backend'}), 200

