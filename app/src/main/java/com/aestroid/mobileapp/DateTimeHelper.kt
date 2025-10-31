package com.aestroid.mobileapp

import java.text.SimpleDateFormat
import java.util.*

object DateTimeHelper {
    /**
     * Get current timestamp in IST (Indian Standard Time) format
     * Format: "yyyy-MM-dd HH:mm:ss"
     * Example: "2024-01-15 14:30:45"
     */
    fun getCurrentISTTimestamp(): String {
        val istTimeZone = TimeZone.getTimeZone("Asia/Kolkata")
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        dateFormat.timeZone = istTimeZone
        return dateFormat.format(Date())
    }
}

