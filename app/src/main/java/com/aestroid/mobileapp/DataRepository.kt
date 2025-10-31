package com.aestroid.mobileapp

import android.util.Log
import com.aestroid.mobileapp.dataclasses.ApiResponse
import com.aestroid.mobileapp.dataclasses.LocationRequest
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

object DataRepository {
    private val apiClient = ApiClient
    private val _dataFlow = MutableSharedFlow<ApiResponse>()
    val dataFlow: SharedFlow<ApiResponse> = _dataFlow.asSharedFlow()
    suspend fun sendLocation(lat: Double, lon: Double) {
        val requestBody = LocationRequest(
            latitude = lat,
            longitude = lon,
            timeStamp = System.currentTimeMillis()
        )
        val result = apiClient.sendLocation(requestBody)
        result.onSuccess { apiResponse ->
            _dataFlow.emit(apiResponse)
        }.onFailure { exception ->
            Log.e("DataRepository", "Network Error: ${exception.message}")
        }
    }
}