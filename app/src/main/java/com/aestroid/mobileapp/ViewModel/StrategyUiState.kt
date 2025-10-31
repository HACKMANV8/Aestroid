package com.aestroid.mobileapp.ViewModel
sealed interface StrategyUiState {
    object Loading : StrategyUiState
    data class Success(val strategyMessage: String) : StrategyUiState
    data class Error(val errorMessage: String) : StrategyUiState
}
