package com.rollingcatsoftware.universalnfcreader.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationType
import com.rollingcatsoftware.universalnfcreader.ui.MainViewModel
import com.rollingcatsoftware.universalnfcreader.ui.components.MrzInputDialog
import com.rollingcatsoftware.universalnfcreader.ui.navigation.Screen
import com.rollingcatsoftware.universalnfcreader.ui.navigation.bottomNavItems

/**
 * Main app screen with bottom navigation.
 */
@Composable
fun AppScreen(
    viewModel: MainViewModel,
    onOpenNfcSettings: () -> Unit,
    onRefreshNfcStatus: () -> Unit = {},
    onResetNfcForRetap: () -> Unit = {},
    onRestoreNormalNfc: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val uiState by viewModel.uiState.collectAsState()
    var selectedItemIndex by rememberSaveable { mutableIntStateOf(0) }

    // Track previous pendingAuthData state to detect state transitions
    var previousPendingAuthData by remember { mutableStateOf(uiState.pendingAuthData) }

    // Handle NFC dispatch mode changes based on pendingAuthData state
    LaunchedEffect(uiState.pendingAuthData) {
        when {
            // MRZ was just provided - reset NFC for re-tap
            uiState.pendingAuthData != null && previousPendingAuthData == null -> {
                // Add delay to ensure camera resources are fully released
                kotlinx.coroutines.delay(300)
                onResetNfcForRetap()
            }
            // Read completed (success or cancelled) - restore normal NFC mode
            uiState.pendingAuthData == null && previousPendingAuthData != null -> {
                // Restore normal NFC mode after read completes
                onRestoreNormalNfc()
            }
        }
        previousPendingAuthData = uiState.pendingAuthData
    }

    // Show badge on history tab when there are items
    val historyBadgeCount = uiState.readHistory.size

    Scaffold(
        modifier = modifier.fillMaxSize(),
        bottomBar = {
            NavigationBar {
                bottomNavItems.forEachIndexed { index, screen ->
                    val selected = selectedItemIndex == index
                    NavigationBarItem(
                        selected = selected,
                        onClick = { selectedItemIndex = index },
                        icon = {
                            if (screen == Screen.History && historyBadgeCount > 0) {
                                BadgedBox(
                                    badge = {
                                        Badge {
                                            Text(
                                                text = if (historyBadgeCount > 9) "9+" else historyBadgeCount.toString(),
                                                style = MaterialTheme.typography.labelSmall
                                            )
                                        }
                                    }
                                ) {
                                    Icon(
                                        imageVector = if (selected) screen.selectedIcon else screen.unselectedIcon,
                                        contentDescription = screen.title
                                    )
                                }
                            } else {
                                Icon(
                                    imageVector = if (selected) screen.selectedIcon else screen.unselectedIcon,
                                    contentDescription = screen.title
                                )
                            }
                        },
                        label = { Text(screen.title) }
                    )
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Crossfade between screens
            when (selectedItemIndex) {
                0 -> {
                    AnimatedVisibility(
                        visible = true,
                        enter = fadeIn() + slideInVertically { -30 },
                        exit = fadeOut() + slideOutVertically { 30 }
                    ) {
                        ScanScreen(
                            uiState = uiState,
                            onCardDismissed = viewModel::onCardDismissed,
                            onPendingAuthCancelled = viewModel::onPendingAuthCancelled,
                            onRefreshNfcStatus = onRefreshNfcStatus,
                            onOpenNfcSettings = onOpenNfcSettings
                        )
                    }
                }

                1 -> {
                    AnimatedVisibility(
                        visible = true,
                        enter = fadeIn() + slideInVertically { -30 },
                        exit = fadeOut() + slideOutVertically { 30 }
                    ) {
                        HistoryScreen(
                            readHistory = uiState.readHistory,
                            onClearHistory = viewModel::clearHistory
                        )
                    }
                }

                2 -> {
                    AnimatedVisibility(
                        visible = true,
                        enter = fadeIn() + slideInVertically { -30 },
                        exit = fadeOut() + slideOutVertically { 30 }
                    ) {
                        SettingsScreen(
                            onOpenNfcSettings = onOpenNfcSettings
                        )
                    }
                }
            }
        }
    }

    // MRZ Authentication Dialog for Turkish eID
    uiState.authRequired?.let { authReq ->
        if (authReq.authType == AuthenticationType.MRZ_BAC) {
            MrzInputDialog(
                onDismiss = { viewModel.onAuthenticationCancelled() },
                onAuthenticate = { mrzData ->
                    viewModel.onAuthenticationProvided(mrzData)
                }
            )
        }
    }
}
