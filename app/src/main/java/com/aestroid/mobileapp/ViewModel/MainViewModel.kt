package com.aestroid.mobileapp.ViewModel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.aestroid.mobileapp.DataRepository
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
    private fun processResponse(response: ApiResponse) {
        if (response.strategy.startsWith("Error:")) {
            _uiState.value = StrategyUiState.Error(response.strategy)
        } else {
            _uiState.value = StrategyUiState.Success(response.strategy)
        }
    }
}