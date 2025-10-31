package com.aestroid.mobileapp

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build

class MyApp : Application() {

    override fun onCreate() {
        super.onCreate()

        // This only runs on Android 8 (Oreo) or higher
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "location", // An ID for this channel
                "Location", // The user-visible name
                NotificationManager.IMPORTANCE_LOW // Low importance = no sound
            )

            val notificationManager =
                getSystemService(NOTIFICATION_SERVICE) as NotificationManager

            notificationManager.createNotificationChannel(channel)
        }
    }
}