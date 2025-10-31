package com.aestroid.mobileapp.dataclasses

import kotlinx.serialization.Serializable

@Serializable
data class LocationRequest(
    val unitId: String,
    val unitType: String,
    val latitude: Double,
    val longitude: Double,
    val timestamp: String // IST format: "yyyy-MM-dd HH:mm:ss"
)

@Serializable
data class ApiPost(
    val status: String
)

@Serializable
data class ApiResponse(
    val strategy: String
)