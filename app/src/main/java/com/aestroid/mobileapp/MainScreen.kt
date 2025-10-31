package com.aestroid.mobileapp

import android.Manifest
import android.content.Context
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.aestroid.mobileapp.ViewModel.MainViewModel
import com.aestroid.mobileapp.ViewModel.StrategyUiState

@Composable
fun MainScreen(
    // This gets the ViewModel and ties it to the screen's lifecycle
    viewModel: MainViewModel = viewModel()
) {
    // 1. Get the Android Context for starting the service
    val context = LocalContext.current

    // 2. Observe the UI state from the ViewModel.
    // This 'by' keyword makes the Composable automatically
    // update when the state changes.
    val uiState by viewModel.uiState.collectAsState()

    // 3. Set up the Permission Launcher.
    // This handles asking the user for location permissions.
    val locationPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
        onResult = { permissions ->
            // Check if either fine or coarse location was granted
            if (permissions.getOrDefault(Manifest.permission.ACCESS_FINE_LOCATION, false) ||
                permissions.getOrDefault(Manifest.permission.ACCESS_COARSE_LOCATION, false)) {

                // Permission granted! Start the service.
                startLocationService(context)

            } else {
                // Permission was denied.
                // In a real app, you'd show a "permission denied" message.
            }
        }
    )

    // --- The UI Layout ---
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        // --- Data Display Area ---
        Text(
            text = "Server Strategy:",
            style = MaterialTheme.typography.titleLarge
        )
        Spacer(modifier = Modifier.height(16.dp))

        // This 'when' block automatically reacts to state changes
        when (val state = uiState) {
            is StrategyUiState.Loading -> {
                CircularProgressIndicator()
                Text(
                    text = "Waiting for location...",
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
            is StrategyUiState.Success -> {
                Text(
                    text = state.strategyMessage,
                    style = MaterialTheme.typography.bodyLarge
                )
            }
            is StrategyUiState.Error -> {
                Text(
                    text = state.errorMessage,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.error // Show errors in red
                )
            }
        }

        // This pushes the buttons to the bottom of the screen
        Spacer(modifier = Modifier.weight(1f))

        // --- Control Buttons ---
        Button(
            onClick = {
                // 4. This button's click launches the permission request
                locationPermissionLauncher.launch(
                    arrayOf(
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    )
                )
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Start Location Tracking")
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = {
                // 5. This button calls the ViewModel function
                viewModel.onUserSendStatusClick()
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Send 'Status' POST Request")
        }
    }
}

// A simple helper function to start your LocationService
private fun startLocationService(context: Context) {
    val intent = Intent(context, LocationService::class.java)
    context.startService(intent)
}