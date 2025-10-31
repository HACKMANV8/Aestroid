package com.aestroid.mobileapp.dataclasses

import kotlinx.serialization.Serializable

@Serializable
data class LocationRequest(
    val unitId: String,
    val unitType: String,
    val latitude: Double,
    val longitude: Double
)

@Serializable
data class ApiPost(
    val status: String
)

@Serializable
data class ApiResponse(
    val strategy: String
)