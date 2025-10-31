package com.aestroid.mobileapp.ViewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aestroid.mobileapp.ApiClient
import com.aestroid.mobileapp.DataRepository
import com.aestroid.mobileapp.dataclasses.ApiPost
import com.aestroid.mobileapp.dataclasses.ApiResponse
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class MainViewModel: ViewModel() {
    private val dataRepository = DataRepository
    private val _uiState = MutableStateFlow<StrategyUiState>(StrategyUiState.Loading)
    val uiState: StateFlow<StrategyUiState> = _uiState.asStateFlow()
    init {
        viewModelScope.launch {
            dataRepository.dataFlow
                .catch { exception ->
                    _uiState.value = StrategyUiState.Error(exception.message ?: "Unknown error")
                }
                .collect { response ->
                    processResponse(response)
                }
        }
    }
    fun onUserSendStatusClick() {
        // Launch a new coroutine to do the work
        viewModelScope.launch {
            // 1. Create the request object
            val postData = ApiPost(status = "UserClickedButton")

            // 2. Call the ApiClient's function
            // (We call ApiClient directly since DataRepository doesn't have this function)
            val result = ApiClient.sendApiPost(postData)

            // 3. Handle the result
            result.onSuccess { apiResponse ->
                // The POST was successful. Update the UI.
                processResponse(apiResponse)
            }.onFailure { exception ->
                // The POST failed. Update the UI to show an error.
                _uiState.value = StrategyUiState.Error(exception.message ?: "POST Error")
            }
        }
    }
    private fun processResponse(response: ApiResponse) {
        if (response.strategy.startsWith("Error:")) {
            _uiState.value = StrategyUiState.Error(response.strategy)
        } else {
            _uiState.value = StrategyUiState.Success(response.strategy)
        }
    }
}