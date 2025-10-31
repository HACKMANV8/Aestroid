package com.aestroid.mobileapp

import android.Manifest
import android.content.Context
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.aestroid.mobileapp.ViewModel.MainViewModel
import com.aestroid.mobileapp.ViewModel.StrategyUiState

@OptIn(ExperimentalMaterial3Api::class) // Needed for TopAppBar
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel()
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()

    // --- Permission Launcher ---
    val locationPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
        onResult = { permissions ->
            if (permissions.getOrDefault(Manifest.permission.ACCESS_FINE_LOCATION, false) ||
                permissions.getOrDefault(Manifest.permission.ACCESS_COARSE_LOCATION, false)) {

                // Permission granted! Start the service.
                startLocationService(context)

            } else {
                // TODO: Show a SnackBar or dialog explaining permission was denied
            }
        }
    )

    // --- UI Layout ---
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Location Strategy App") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding) // Apply padding from the Scaffold
                .padding(16.dp), // Apply our own padding
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Spacer(modifier = Modifier.height(32.dp))

            // --- Data Display Card ---
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .defaultMinSize(minHeight = 150.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier
                        .padding(24.dp)
                        .fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    // This 'when' block automatically reacts to state changes
                    when (val state = uiState) {
                        is StrategyUiState.Loading -> {
                            CircularProgressIndicator()
                            Text(
                                text = "Waiting for first location update...",
                                style = MaterialTheme.typography.bodyMedium,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.padding(top = 16.dp)
                            )
                        }
                        is StrategyUiState.Success -> {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = "Success",
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(48.dp)
                            )
                            Text(
                                text = state.strategyMessage,
                                style = MaterialTheme.typography.headlineSmall,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.padding(top = 16.dp)
                            )
                        }
                        is StrategyUiState.Error -> {
                            Icon(
                                imageVector = Icons.Default.Error,
                                contentDescription = "Error",
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(48.dp)
                            )
                            Text(
                                text = state.errorMessage,
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.error,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.padding(top = 16.dp)
                            )
                        }
                    }
                }
            }

            // This spacer pushes the buttons to the bottom
            Spacer(modifier = Modifier.weight(1f))

            // --- Control Buttons ---
            Button(
                onClick = {
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

            OutlinedButton( // Use an OutlinedButton for secondary action
                onClick = {
                    viewModel.onUserSendStatusClick()
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Send 'Status' POST Request")
            }
        }
    }
}

// Helper function to start the service
private fun startLocationService(context: Context) {
    val intent = Intent(context, LocationService::class.java)
    context.startService(intent)
}