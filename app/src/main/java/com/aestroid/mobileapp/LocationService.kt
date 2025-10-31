package com.aestroid.mobileapp

import android.app.Notification
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class LocationService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // The main client for getting location updates
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    // The callback object that will receive location updates
    private lateinit var locationCallback: LocationCallback

    override fun onCreate() {
        super.onCreate()
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        // Initialize the location callback
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                super.onLocationResult(locationResult)

                locationResult.lastLocation?.let { location ->
                    Log.d("LocationService", "New location: ${location.latitude}, ${location.longitude}")

                    // We got a location. Launch a coroutine in our service's scope
                    // to call the DataRepository. This is the main goal!
                    serviceScope.launch {
                        DataRepository.sendLocation(location.latitude, location.longitude)
                    }
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // This is called when the service is started (e.g., from the UI)

        // 1. Create the notification
        val notification = createNotification()

        // 2. Start the service in the foreground
        // The ID (1) must be a non-zero integer.
        startForeground(1, notification)

        // 3. Start requesting location updates
        startLocationUpdates()

        // START_STICKY tells the system to restart the service if it gets killed
        return START_STICKY
    }

    private fun startLocationUpdates() {
        // Check if we have permission. This is crucial.
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

            Log.e("LocationService", "Location permission not granted. Stopping service.")
            stopSelf() // Stop the service if permissions are missing
            return
        }

        // Configure how often we want location updates
        val locationRequest = LocationRequest.create().apply {
            interval = 10000 // 10 seconds
            fastestInterval = 5000 // 5 seconds
            priority = Priority.PRIORITY_HIGH_ACCURACY
        }

        // Start listening for updates
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            Looper.getMainLooper() // The thread to receive updates on
        )
    }

    private fun createNotification(): Notification {
        // Build the notification that will be shown to the user
        return NotificationCompat.Builder(this, "location") // "location" is the channel ID
            .setContentTitle("Location Tracking Active")
            .setContentText("Your location is being sent to the server.")
            .setSmallIcon(R.mipmap.ic_launcher) // TODO: Change to your app's icon
            .setOngoing(true) // Makes it non-dismissible
            .build()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Stop location updates
        fusedLocationClient.removeLocationUpdates(locationCallback)
        // Cancel all coroutines
        serviceScope.cancel()
    }

    // This is a "Started Service", not a "Bound Service", so we return null.
    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
}