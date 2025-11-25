package com.rollingcatsoftware.universalnfcreader.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Contactless
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PortableWifiOff
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.ui.MainUiState
import com.rollingcatsoftware.universalnfcreader.ui.components.CardInfoCard
import com.rollingcatsoftware.universalnfcreader.ui.components.LoadingSkeleton
import com.rollingcatsoftware.universalnfcreader.ui.components.PulsingDotsIndicator

/**
 * Scan screen state for Crossfade transitions.
 */
private sealed class ScanState {
    data object Idle : ScanState()
    data object Reading : ScanState()
    data object WaitingForRetap : ScanState()
    data class Success(val card: CardData) : ScanState()
    data class Error(val error: CardError) : ScanState()
    data object NfcUnavailable : ScanState()
    data object NfcDisabled : ScanState()
}

/**
 * Scan screen - main NFC scanning interface with smooth transitions.
 */
@Composable
fun ScanScreen(
    uiState: MainUiState,
    onCardDismissed: () -> Unit,
    onPendingAuthCancelled: () -> Unit,
    modifier: Modifier = Modifier
) {
    val scanState = when {
        !uiState.isNfcAvailable -> ScanState.NfcUnavailable
        !uiState.isNfcEnabled -> ScanState.NfcDisabled
        uiState.isReading -> ScanState.Reading
        uiState.pendingAuthData != null -> ScanState.WaitingForRetap
        uiState.lastReadCard != null -> ScanState.Success(uiState.lastReadCard)
        uiState.error != null -> ScanState.Error(uiState.error)
        else -> ScanState.Idle
    }

    // Crossfade for smooth transitions
    Crossfade(
        targetState = scanState,
        animationSpec = tween(durationMillis = 300),
        label = "scan_state_transition",
        modifier = modifier.fillMaxSize()
    ) { state ->
        when (state) {
            is ScanState.Idle -> IdleContent(readHistory = uiState.readHistory)
            is ScanState.Reading -> ReadingContent()
            is ScanState.WaitingForRetap -> WaitingForRetapContent(onCancel = onPendingAuthCancelled)
            is ScanState.Success -> SuccessContent(
                cardData = state.card,
                readHistory = uiState.readHistory,
                onDismiss = onCardDismissed
            )
            is ScanState.Error -> ErrorContent(error = state.error, onRetry = onCardDismissed)
            is ScanState.NfcUnavailable -> NfcUnavailableContent()
            is ScanState.NfcDisabled -> NfcDisabledContent()
        }
    }
}

@Composable
private fun IdleContent(
    readHistory: List<CardData>,
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // NFC Scan prompt with animation
        item {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600)) + scaleIn(initialScale = 0.9f, animationSpec = tween(600))
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(32.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Icon(
                            imageVector = Icons.Default.Contactless,
                            contentDescription = null,
                            modifier = Modifier.size(80.dp),
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Ready to Scan",
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Hold an NFC card near the device to read it",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        }

        // Supported cards info with animation
        item {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(600, delayMillis = 150)) +
                    slideInVertically(tween(600, delayMillis = 150)) { 20 }
            ) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .semantics {
                            contentDescription = "Supported card types information card"
                        },
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Text(
                                text = "Supported Cards",
                                style = MaterialTheme.typography.titleSmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        SupportedCardsList()
                    }
                }
            }
        }

        // Recent reads
        if (readHistory.isNotEmpty()) {
            item {
                AnimatedVisibility(
                    visible = visible,
                    enter = fadeIn(tween(600, delayMillis = 300))
                ) {
                    Text(
                        text = "Recent Reads",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }
            }
            items(readHistory.take(3)) { card ->
                AnimatedVisibility(
                    visible = visible,
                    enter = fadeIn(tween(400, delayMillis = 350)) +
                        slideInVertically(tween(400, delayMillis = 350)) { 30 }
                ) {
                    CardInfoCard(
                        cardData = card,
                        isCompact = true
                    )
                }
            }
        }
    }
}

@Composable
private fun SupportedCardsList() {
    val cards = listOf(
        "Turkish eID" to "BAC authentication",
        "Istanbulkart" to "Transport card",
        "MIFARE Classic" to "1K/4K cards",
        "MIFARE DESFire" to "High security",
        "MIFARE Ultralight" to "Memory cards",
        "NDEF Tags" to "Formatted tags"
    )

    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        cards.forEach { (name, description) ->
            Text(
                text = "• $name - $description",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.8f)
            )
        }
    }
}

@Composable
private fun ReadingContent(
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp)
            .semantics {
                contentDescription = "Reading card in progress. Please keep your card near the device."
            },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated circular progress
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400)) + scaleIn(tween(400))
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(64.dp),
                strokeWidth = 6.dp
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Animated text with pulsing dots
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400, delayMillis = 100))
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "Reading card",
                    style = MaterialTheme.typography.titleLarge
                )
                Spacer(modifier = Modifier.width(8.dp))
                PulsingDotsIndicator()
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400, delayMillis = 200))
        ) {
            Text(
                text = "Keep your card near the device",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Loading skeleton
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400, delayMillis = 300)) + expandVertically(tween(400, delayMillis = 300))
        ) {
            Card(modifier = Modifier.fillMaxWidth()) {
                LoadingSkeleton()
            }
        }
    }
}

@Composable
private fun WaitingForRetapContent(
    onCancel: () -> Unit,
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp)
        ) {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400)) + scaleIn(initialScale = 0.8f, animationSpec = tween(400))
            ) {
                Icon(
                    imageVector = Icons.Default.Contactless,
                    contentDescription = null,
                    modifier = Modifier.size(100.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 100)) + slideInVertically(tween(400, delayMillis = 100)) { -20 }
            ) {
                Text(
                    text = "Tap Your Card Again",
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 200))
            ) {
                Text(
                    text = "MRZ credentials saved.\nPlease tap your Turkish eID card to read personal data.",
                    style = MaterialTheme.typography.bodyLarge,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 300)) + scaleIn(tween(400, delayMillis = 300))
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(48.dp),
                    strokeWidth = 4.dp
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 400))
            ) {
                OutlinedButton(onClick = onCancel) {
                    Text("Cancel")
                }
            }
        }
    }
}

@Composable
private fun SuccessContent(
    cardData: CardData,
    readHistory: List<CardData>,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Success header
        item {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400)) + slideInVertically(tween(400)) { -40 }
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = "Success",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(32.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = "Card Read Successfully",
                            style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                }
            }
        }

        // Card data
        item {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 150)) +
                    scaleIn(initialScale = 0.95f, animationSpec = tween(400, delayMillis = 150))
            ) {
                CardInfoCard(
                    cardData = cardData,
                    isCompact = false
                )
            }
        }

        // Info tip
        item {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 250)) + slideInVertically(tween(400, delayMillis = 250)) { 20 }
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSecondaryContainer,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = "Tap another card to read it, or swipe to see history.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                }
            }
        }

        // Read another button
        item {
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(tween(400, delayMillis = 350)) + slideInVertically(tween(400, delayMillis = 350)) { 20 }
            ) {
                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(imageVector = Icons.Default.Refresh, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Scan Another Card")
                }
            }
        }

        // Previous reads
        if (readHistory.size > 1) {
            item {
                AnimatedVisibility(
                    visible = visible,
                    enter = fadeIn(tween(400, delayMillis = 400))
                ) {
                    Text(
                        text = "Previous Reads",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            }
            items(readHistory.drop(1).take(3)) { card ->
                AnimatedVisibility(
                    visible = visible,
                    enter = fadeIn(tween(300, delayMillis = 450)) + slideInVertically(tween(300, delayMillis = 450)) { 30 }
                ) {
                    CardInfoCard(
                        cardData = card,
                        isCompact = true
                    )
                }
            }
        }
    }
}

@Composable
private fun ErrorContent(
    error: CardError,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated error icon
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400)) + scaleIn(initialScale = 0.7f, animationSpec = tween(400, easing = FastOutSlowInEasing))
        ) {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = "Error",
                modifier = Modifier.size(80.dp),
                tint = MaterialTheme.colorScheme.error
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Error title
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400, delayMillis = 150)) + slideInVertically(tween(400, delayMillis = 150)) { -20 }
        ) {
            Text(
                text = "Error Reading Card",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.error
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Error message card with guidance
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400, delayMillis = 250)) + expandVertically(tween(400, delayMillis = 250))
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = error.message,
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = getErrorGuidance(error),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.8f)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Try again button
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400, delayMillis = 350)) + slideInVertically(tween(400, delayMillis = 350)) { 40 }
        ) {
            Button(
                onClick = onRetry,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(imageVector = Icons.Default.Refresh, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Try Again")
            }
        }
    }
}

/**
 * Get contextual guidance for each error type.
 */
private fun getErrorGuidance(error: CardError): String {
    return when (error) {
        is CardError.ConnectionLost -> """
            ✓ Hold your card steady near the device
            ✓ Keep the card in place until reading completes
            ✓ Try different positioning
        """.trimIndent()
        is CardError.IoError -> """
            ✓ Remove any phone cases that may interfere
            ✓ Clean the card surface
            ✓ Try a different card position
        """.trimIndent()
        is CardError.AuthenticationFailed -> """
            ✓ Verify your MRZ data is correct
            ✓ Check document number, DOB, and DOE
            ✓ The card may be locked after too many attempts
        """.trimIndent()
        is CardError.UnsupportedCard -> """
            ✓ This card type may not be supported
            ✓ Check if the card has an NFC chip
            ✓ Some cards require specific authentication
        """.trimIndent()
        is CardError.NfcNotAvailable -> """
            ✓ This device does not have NFC hardware
            ✓ NFC is required to read cards
        """.trimIndent()
        is CardError.NfcDisabled -> """
            ✓ Enable NFC in device settings
            ✓ Go to Settings → Connected devices → NFC
        """.trimIndent()
        is CardError.CardBlocked -> """
            ✓ Card has been locked due to too many failed attempts
            ✓ You may need to contact the card issuer
            ✓ Some cards can be unblocked with a PUK code
        """.trimIndent()
        is CardError.Timeout -> """
            ✓ Keep the card near the device longer
            ✓ Try again with card held steady
            ✓ Ensure card is properly positioned
        """.trimIndent()
        else -> """
            ✓ Try again with the card held steady
            ✓ Make sure NFC is enabled
            ✓ Remove any obstructions between phone and card
        """.trimIndent()
    }
}

@Composable
private fun NfcUnavailableContent(
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400)) + scaleIn(tween(400))
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(32.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp),
                    tint = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "NFC Not Available",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "This device does not support NFC",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
private fun NfcDisabledContent(
    modifier: Modifier = Modifier
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(tween(400)) + scaleIn(tween(400))
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(32.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.PortableWifiOff,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp),
                    tint = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "NFC Disabled",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Please enable NFC in your device settings",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}
