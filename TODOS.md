# TODO List - Military Location Tracking Android App

This document lists remaining high-priority TODO comments in the codebase.

## Remaining High Priority TODOs

### API Client (`ApiClient.kt`)

1. **Update BASE_URL for production deployment**
   - Currently set to emulator default: `http://10.0.2.2:5000`
   - Must be updated to production server URL before deployment
   - For physical device testing: use your computer's IP address, e.g., `http://192.168.1.100:5000`

### Location Service (`LocationService.kt`)

2. **Replace with a proper notification icon (white/transparent icon)**
   - Currently uses app launcher icon (`R.mipmap.ic_launcher`)
   - Should use a white/transparent icon for better visibility on Android
   - Create a drawable resource with a white icon for better notification visibility

## Completed TODOs ✅

### ✅ Implemented Retry Logic with Exponential Backoff (`ApiClient.kt`)
- **Status**: ✅ Completed
- **Implementation**: Retry logic with exponential backoff (1s, 2s, 4s, 8s, up to 30s max)
- **Features**:
  - Maximum 5 retry attempts
  - Exponential backoff starting at 1 second
  - Distinguishes between transient network errors and permanent failures
  - Retries 5xx server errors
  - Proper error logging with retry attempt information

### ✅ Implemented Android 10+ Background Location Permission Handling (`MainScreen.kt`)
- **Status**: ✅ Completed
- **Implementation**: Proper handling of background location permission for Android 10+
- **Features**:
  - Checks Android version (API 29+)
  - Requests foreground location permission first
  - Then requests background location permission separately
  - Shows rationale dialog explaining why background location is needed
  - Gracefully handles permission denial
  - Still works with foreground-only permission if background is denied
  - User-friendly error messages via SnackBar

## Implementation Notes

### Retry Logic
The retry mechanism:
- Attempts up to 5 retries with exponential backoff
- Initial retry after 1 second
- Doubles wait time for each retry (1s → 2s → 4s → 8s → 16s)
- Caps maximum retry delay at 30 seconds
- Retries transient network errors (timeout, connection, network, socket, unreachable, reset, refused)
- Retries 5xx server errors
- Does not retry 4xx client errors or permanent failures

### Background Location Permission
The implementation:
- Detects Android 10+ (API 29+) automatically
- Requests foreground location first (ACCESS_FINE_LOCATION or ACCESS_COARSE_LOCATION)
- After foreground permission is granted, requests background permission separately
- Shows AlertDialog with rationale for background location
- Allows user to grant or cancel background permission
- Gracefully degrades to foreground-only tracking if background is denied
- Provides user feedback via SnackBar messages

## Remaining Tasks

Only 2 TODOs remain:
1. Update BASE_URL for production (deployment-time change)
2. Replace notification icon (requires creating drawable resource)

All critical functionality has been implemented!
