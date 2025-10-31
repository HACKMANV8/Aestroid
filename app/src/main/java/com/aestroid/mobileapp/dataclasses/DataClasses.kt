package com.aestroid.mobileapp.dataclasses

import kotlinx.serialization.Serializable

@Serializable
data class LocationRequest(
    val latitude: Double,
    val longitude: Double,
    val timeStamp: Long
)

@Serializable
data class ApiPost(
    val status: String
)

@Serializable
data class ApiResponse(
    val strategy: String
)