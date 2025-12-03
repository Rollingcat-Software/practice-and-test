package com.rollingcatsoftware.universalnfcreader.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.ColorLens
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Nfc
import androidx.compose.material.icons.filled.Security
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LargeTopAppBar
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.unit.dp

/**
 * Settings screen for app configuration.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onOpenNfcSettings: () -> Unit,
    modifier: Modifier = Modifier
) {
    val scrollBehavior = TopAppBarDefaults.exitUntilCollapsedScrollBehavior()
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    Scaffold(
        modifier = modifier.nestedScroll(scrollBehavior.nestedScrollConnection),
        topBar = {
            LargeTopAppBar(
                title = { Text("Settings") },
                scrollBehavior = scrollBehavior
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp)
        ) {
            // NFC Settings Section
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(300)) + slideInVertically(tween(300)) { 20 }
            ) {
                SettingsSection(title = "NFC") {
                    SettingsClickItem(
                        icon = Icons.Default.Nfc,
                        title = "NFC Settings",
                        subtitle = "Open device NFC settings",
                        onClick = onOpenNfcSettings
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Security Section
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(300, delayMillis = 100)) + slideInVertically(
                    tween(
                        300,
                        delayMillis = 100
                    )
                ) { 20 }
            ) {
                SettingsSection(title = "Security") {
                    var biometricEnabled by remember { mutableStateOf(true) }
                    SettingsSwitchItem(
                        icon = Icons.Default.Security,
                        title = "Biometric Lock",
                        subtitle = "Require authentication on launch",
                        checked = biometricEnabled,
                        onCheckedChange = { biometricEnabled = it }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Appearance Section
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(300, delayMillis = 200)) + slideInVertically(
                    tween(
                        300,
                        delayMillis = 200
                    )
                ) { 20 }
            ) {
                SettingsSection(title = "Appearance") {
                    var dynamicColors by remember { mutableStateOf(true) }
                    SettingsSwitchItem(
                        icon = Icons.Default.ColorLens,
                        title = "Dynamic Colors",
                        subtitle = "Use Material You wallpaper colors",
                        checked = dynamicColors,
                        onCheckedChange = { dynamicColors = it }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // About Section
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(300, delayMillis = 300)) + slideInVertically(
                    tween(
                        300,
                        delayMillis = 300
                    )
                ) { 20 }
            ) {
                SettingsSection(title = "About") {
                    SettingsInfoItem(
                        icon = Icons.Default.Info,
                        title = "Version",
                        value = "1.0.0"
                    )
                    HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                    SettingsInfoItem(
                        icon = Icons.Default.Code,
                        title = "Build",
                        value = "Debug"
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Supported Cards Info
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(300, delayMillis = 400)) + slideInVertically(
                    tween(
                        300,
                        delayMillis = 400
                    )
                ) { 20 }
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Supported Card Types",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        val cards = listOf(
                            "Turkish eID (BAC Authentication)",
                            "Istanbulkart (DESFire)",
                            "MIFARE Classic 1K/4K",
                            "MIFARE DESFire",
                            "MIFARE Ultralight",
                            "NDEF Tags",
                            "ISO 15693 (NfcV)"
                        )
                        cards.forEach { card ->
                            Text(
                                text = "• $card",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.8f),
                                modifier = Modifier.padding(vertical = 2.dp)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

@Composable
private fun SettingsSection(
    title: String,
    content: @Composable () -> Unit
) {
    Column {
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(start = 16.dp, bottom = 8.dp)
        )
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            content()
        }
    }
}

@Composable
private fun SettingsClickItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit
) {
    ListItem(
        headlineContent = { Text(title) },
        supportingContent = { Text(subtitle) },
        leadingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary
            )
        },
        trailingContent = {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.OpenInNew,
                contentDescription = null,
                modifier = Modifier.size(20.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        },
        modifier = Modifier.clickable(onClick = onClick)
    )
}

@Composable
private fun SettingsSwitchItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    ListItem(
        headlineContent = { Text(title) },
        supportingContent = { Text(subtitle) },
        leadingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary
            )
        },
        trailingContent = {
            Switch(
                checked = checked,
                onCheckedChange = onCheckedChange
            )
        }
    )
}

@Composable
private fun SettingsInfoItem(
    icon: ImageVector,
    title: String,
    value: String
) {
    ListItem(
        headlineContent = { Text(title) },
        leadingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary
            )
        },
        trailingContent = {
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    )
}
