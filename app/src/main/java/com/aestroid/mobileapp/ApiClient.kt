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
import kotlinx.coroutines.delay

object ApiClient {
    private const val MAX_RETRY_ATTEMPTS = 5
    private const val INITIAL_RETRY_DELAY_MS = 1000L // 1 second
    private const val MAX_RETRY_DELAY_MS = 30000L // 30 seconds
    private const val RETRY_MULTIPLIER = 2.0
    // Made this internal so DataRepository can access it
    // TODO: Update BASE_URL for production deployment
    // Change this to your backend server URL (e.g., http://your-server-ip:5000)
    // For local testing with emulator: http://10.0.2.2:5000
    // For physical device: use your computer's IP address, e.g., http://192.168.1.100:5000
    internal const val BASE_URL = "https://conjunctively-isoelastic-amiyah.ngrok-free.dev" // Default for Android emulator

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

    suspend fun sendLocation(locationData: LocationRequest): Result<Unit> {
        val endpoint = "$BASE_URL/api/location"
        
        var lastException: Exception? = null
        var retryDelay = INITIAL_RETRY_DELAY_MS
        
        for (attempt in 0..MAX_RETRY_ATTEMPTS) {
            try {
                val response = client.post(endpoint) {
                    contentType(ContentType.Application.Json)
                    setBody(locationData)
                }
                
                if (response.status.value in 200..299) {
                    if (attempt > 0) {
                        Log.d("ApiClient", "Location sent successfully after $attempt retry attempts")
                    } else {
                        Log.d("ApiClient", "Location sent successfully")
                    }
                    return Result.success(Unit)
                } else {
                    // Server error - only retry for 5xx errors
                    val isRetryable = response.status.value in 500..599
                    if (isRetryable && attempt < MAX_RETRY_ATTEMPTS) {
                        Log.w("ApiClient", "Server error ${response.status.value}, retrying in ${retryDelay}ms (attempt ${attempt + 1}/$MAX_RETRY_ATTEMPTS)")
                        delay(retryDelay)
                        retryDelay = (retryDelay * RETRY_MULTIPLIER).toLong().coerceAtMost(MAX_RETRY_DELAY_MS)
                        lastException = Exception("Server returned status: ${response.status}")
                        continue
                    } else {
                        Log.e("ApiClient", "Location send failed with status: ${response.status}")
                        return Result.failure(Exception("Server returned status: ${response.status}"))
                    }
                }
            } catch (e: Exception) {
                lastException = e
                val isTransientFailure = isTransientNetworkError(e)
                
                if (isTransientFailure && attempt < MAX_RETRY_ATTEMPTS) {
                    Log.w("ApiClient", "Transient network error: ${e.message}, retrying in ${retryDelay}ms (attempt ${attempt + 1}/$MAX_RETRY_ATTEMPTS)")
                    delay(retryDelay)
                    retryDelay = (retryDelay * RETRY_MULTIPLIER).toLong().coerceAtMost(MAX_RETRY_DELAY_MS)
                } else {
                    // Permanent failure or max retries reached
                    Log.e("ApiClient", "Location send failed after ${attempt + 1} attempts: ${e.message}", e)
                    return Result.failure(e)
                }
            }
        }
        
        // If we exhausted all retries
        Log.e("ApiClient", "Location send failed after $MAX_RETRY_ATTEMPTS retry attempts")
        return Result.failure(lastException ?: Exception("Unknown error after retries"))
    }
    
    /**
     * Determines if a network error is transient and worth retrying
     */
    private fun isTransientNetworkError(exception: Exception): Boolean {
        val message = exception.message?.lowercase() ?: return false
        return message.contains("timeout") ||
               message.contains("connection") ||
               message.contains("network") ||
               message.contains("socket") ||
               message.contains("unreachable") ||
               message.contains("reset") ||
               message.contains("refused")
    }

    suspend fun sendApiPost(postData: ApiPost): Result<ApiResponse> {
        val endpoint = "$BASE_URL/api/status"

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
        val endpoint = "$BASE_URL/api/strategy"

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