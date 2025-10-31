package com.aestroid.mobileapp

import android.util.Log
import com.aestroid.mobileapp.dataclasses.LocationRequest
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

object DataRepository {
    private val apiClient = ApiClient
    private val _locationUpdateFlow = MutableSharedFlow<LocationUpdateResult>()
    val locationUpdateFlow: SharedFlow<LocationUpdateResult> = _locationUpdateFlow.asSharedFlow()
    
    sealed class LocationUpdateResult {
        data class Success(val unitId: String, val lat: Double, val lon: Double) : LocationUpdateResult()
        data class Error(val message: String) : LocationUpdateResult()
    }
    
    suspend fun sendLocation(unitId: String, unitType: String, lat: Double, lon: Double) {
        // Get current timestamp in IST format
        val timestamp = DateTimeHelper.getCurrentISTTimestamp()
        
        val requestBody = LocationRequest(
            unitId = unitId,
            unitType = unitType,
            latitude = lat,
            longitude = lon,
            timestamp = timestamp
        )
        val result = apiClient.sendLocation(requestBody)
        result.onSuccess {
            Log.d("DataRepository", "Location sent successfully: $unitId at ($lat, $lon) at $timestamp IST")
            _locationUpdateFlow.emit(LocationUpdateResult.Success(unitId, lat, lon))
        }.onFailure { exception ->
            val errorMsg = "Network Error: ${exception.message}"
            Log.e("DataRepository", errorMsg, exception)
            _locationUpdateFlow.emit(LocationUpdateResult.Error(errorMsg))
        }
    }
}