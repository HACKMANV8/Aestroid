# How to Start the Backend Server

## Issue
The backend server needs to be running for the frontend to work. If you're getting connection errors, follow these steps:

## Quick Fix

**Open a NEW terminal window** (keep it open alongside your frontend terminal) and run:

```bash
npm run server:dev
```

You should see output like:
```
✅ Groq API key configured
Server running on port 5000
WebSocket server ready for connections
```

## Alternative: Start Both Together

Stop any running processes (Ctrl+C) and then in ONE terminal run:

```bash
npm run dev
```

This starts both backend AND frontend together.

## Verify Backend is Running

Open your browser and go to: **http://localhost:5000/health**

You should see:
```json
{"status":"Server is running","timestamp":"..."}
```

## Troubleshooting

### Port 5000 Already in Use
If you get "port 5000 already in use":
```bash
# Find what's using the port (Windows)
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### MongoDB Connection Errors
The server will still run even if MongoDB isn't connected. You'll see warnings but the server works.

### Still Having Issues?
Check the terminal output for error messages - they will tell you what's wrong.

