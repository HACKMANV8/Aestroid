package com.aestroid.mobileapp.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val DarkColorScheme = darkColorScheme(
    primary = OrangePrimary,
    secondary = OrangeSecondary,
    tertiary = OrangeTertiary,
    background = Black,
    surface = DarkGray,
    surfaceVariant = MediumGray,
    onPrimary = Black,
    onSecondary = Black,
    onTertiary = WhiteText,
    onBackground = WhiteText,
    onSurface = WhiteText,
    onSurfaceVariant = LightGrayText,
    error = Color(0xFFBA1A1A),
    onError = WhiteText
)

private val LightColorScheme = lightColorScheme(
    primary = OrangePrimary,
    secondary = OrangeSecondary,
    tertiary = OrangeTertiary,
    background = Color(0xFFFFFBFE),
    surface = Color(0xFFFFFBFE),
    surfaceVariant = Color(0xFFF5F5F5),
    onPrimary = WhiteText,
    onSecondary = WhiteText,
    onTertiary = WhiteText,
    onBackground = Black,
    onSurface = Black,
    onSurfaceVariant = DarkGray,
    error = Color(0xFFBA1A1A),
    onError = WhiteText
)

@Composable
fun AestroidTheme(
    darkTheme: Boolean = true, // Default to dark theme for orange/black aesthetic
    // Dynamic color is available on Android 12+
    dynamicColor: Boolean = false, // Disable dynamic color to use custom orange/black theme
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}