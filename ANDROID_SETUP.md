# Android App Setup for Military Location Tracking

This Android app has been configured to integrate with the Military Location Tracking System backend.

## Configuration

### Backend URL

The app is configured to connect to the backend server. To change the server URL, edit `ApiClient.kt`:

```kotlin
internal const val BASE_URL = "http://10.0.2.2:5000" // For Android emulator
```

**Important Notes:**
- **Android Emulator**: Use `http://10.0.2.2:5000` (maps to `localhost:5000` on your development machine)
- **Physical Device**: Use your computer's IP address, e.g., `http://192.168.1.100:5000`
  - Find your IP: On Linux/Mac use `ip addr` or `ifconfig`, on Windows use `ipconfig`
  - Ensure your phone and computer are on the same network
  - Make sure your firewall allows connections on port 5000

### Unit Configuration

The app allows you to configure:
- **Unit ID**: Unique identifier for this unit (e.g., "TANK-001", "SOLDIER-123")
- **Unit Type**: Type of military unit (e.g., "tank", "helicopter", "soldier", "drone", "vehicle", "mobile")

**How to configure:**
1. Open the app
2. Tap the info icon (ℹ️) in the top-right corner
3. Enter your Unit ID and Unit Type
4. Tap "Save Configuration"

Default values:
- Unit ID: `UNIT-001`
- Unit Type: `mobile`

## Features

### Location Tracking
- Automatically tracks device location every 10 seconds (when active)
- Sends location data to backend at `/api/location`
- Runs as a foreground service with notification
- Supports both foreground and background location updates

### Data Format

The app sends location data in this format to match the backend API:

```json
{
  "unitId": "TANK-001",
  "unitType": "tank",
  "latitude": 12.9716,
  "longitude": 77.5946
}
```

### Permissions Required

The app requires the following permissions:
- `ACCESS_FINE_LOCATION` - For precise GPS location
- `ACCESS_COARSE_LOCATION` - For network-based location
- `ACCESS_BACKGROUND_LOCATION` - For location updates when app is in background
- `FOREGROUND_SERVICE` - For foreground location service
- `FOREGROUND_SERVICE_LOCATION` - For location foreground service
- `INTERNET` - For API communication

All permissions are already declared in `AndroidManifest.xml`.

## Usage

1. **Configure Unit Information** (optional):
   - Tap the info icon in the app bar
   - Set your Unit ID and Unit Type
   - Save the configuration

2. **Start Location Tracking**:
   - Tap "Start Location Tracking" button
   - Grant location permissions when prompted
   - Location updates will begin automatically

3. **Monitor Status**:
   - The app displays the latest location update status
   - Shows unit ID, coordinates, and success/error messages
   - Location data is sent to the backend every 10 seconds

## Testing

### With Android Emulator

1. Start your backend server on `localhost:5000`
2. The emulator is pre-configured to connect to `http://10.0.2.2:5000`
3. Run the app and start location tracking

### With Physical Device

1. Find your computer's IP address (must be on same network as device)
2. Update `BASE_URL` in `ApiClient.kt` to use your IP
3. Start your backend server
4. Ensure firewall allows connections on port 5000
5. Run the app on your device

## Troubleshooting

### Location Not Sending

- **Check backend is running**: Verify the backend server is accessible
- **Check network connection**: Ensure device/emulator has internet access
- **Check BASE_URL**: Verify the URL in `ApiClient.kt` is correct
- **Check logs**: Use `adb logcat` to see error messages
  ```bash
  adb logcat | grep -E "(ApiClient|LocationService|DataRepository)"
  ```

### Permission Issues

- Android 10+ requires background location permission separately
- Go to Settings > Apps > Aestroid > Permissions
- Ensure "Location" is set to "Allow all the time"

### Connection Issues on Physical Device

- Verify phone and computer are on the same WiFi network
- Check firewall settings on your computer
- Try pinging the device from your computer
- Use `adb forward tcp:5000 tcp:5000` as an alternative

## Project Structure

```
app/src/main/java/com/aestroid/mobileapp/
├── ApiClient.kt              # HTTP client configuration and API calls
├── DataRepository.kt         # Repository for location data
├── LocationService.kt        # Foreground service for location tracking
├── UnitConfig.kt             # SharedPreferences helper for unit config
├── MainActivity.kt           # Main activity entry point
├── MainScreen.kt             # Compose UI screen
├── ViewModel/
│   └── MainViewModel.kt      # ViewModel for UI state management
└── dataclasses/
    └── DataClasses.kt        # Data classes for API communication
```

## Dependencies

Key dependencies used:
- **Ktor Client**: HTTP client for API calls
- **Google Play Services Location**: For location tracking
- **Kotlin Coroutines**: For asynchronous operations
- **Jetpack Compose**: For modern UI
- **Jetpack ViewModel**: For state management

All dependencies are already configured in `build.gradle.kts`.

## API Endpoints Used

- `POST /api/location` - Send location data
- `POST /api/status` - Send status updates (optional)
- `GET /api/strategy` - Get strategy responses (optional)

## Next Steps

1. Deploy backend to a server with a public IP/domain
2. Update `BASE_URL` to use production server URL
3. Consider adding:
   - Offline location caching
   - Retry logic for failed requests
   - Location accuracy filtering
   - Multiple unit support

