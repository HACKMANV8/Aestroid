package com.aestroid.mobileapp

import android.util.Log
// Import your data classes
import com.aestroid.mobileapp.dataclasses.ApiPost
import com.aestroid.mobileapp.dataclasses.ApiResponse
import com.aestroid.mobileapp.dataclasses.LocationRequest
import io.ktor.client.HttpClient
import io.ktor.client.call.*
import io.ktor.client.engine.android.Android
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.plugins.logging.Logger
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

object ApiClient {
    // Made this internal so DataRepository can access it
    internal const val BASE_URL = "" //TODO: Add base url for api

    val client: HttpClient = HttpClient(Android) {
        install(ContentNegotiation) {
            json(Json {
                prettyPrint = true
                isLenient = true
                ignoreUnknownKeys = true
            })
        }
        install(Logging) {
            level = LogLevel.ALL
            logger = object : Logger {
                override fun log(message: String) {
                    Log.d("KtorApiClient", message)
                }
            }
        }
    }

    suspend fun sendLocation(locationData: LocationRequest): Result<ApiResponse> {
        val endpoint = "$BASE_URL/location" // TODO: Fix endpoint

        return try {
            val response = client.post(endpoint) {
                contentType(ContentType.Application.Json)
                setBody(locationData)
            }.body<ApiResponse>()

            Result.success(response)
        } catch (e: Exception) {
            Log.e("ApiClient", "sendLocation failed: ${e.message}")
            Result.failure(e)
        }
    }

    suspend fun sendApiPost(postData: ApiPost): Result<ApiResponse> {
        val endpoint = "$BASE_URL/status" // TODO: Fix endpoint

        return try {
            val response = client.post(endpoint) {
                contentType(ContentType.Application.Json)
                setBody(postData)
            }.body<ApiResponse>()

            Result.success(response)
        } catch (e: Exception) {
            Log.e("ApiClient", "sendApiPost failed: ${e.message}")
            Result.failure(e)
        }
    }

    suspend fun getResponse(): Result<ApiResponse> {
        val endpoint = "$BASE_URL/strategy" // TODO: Fix endpoint

        return try {
            val response = client.get(endpoint) {
                contentType(ContentType.Application.Json)
            }.body<ApiResponse>()

            Result.success(response)
        } catch (e: Exception) {
            Log.e("ApiClient", "getResponse failed: ${e.message}")
            Result.failure(e)
        }
    }
}