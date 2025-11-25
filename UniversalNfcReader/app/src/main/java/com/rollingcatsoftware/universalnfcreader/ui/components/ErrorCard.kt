package com.rollingcatsoftware.universalnfcreader.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PortableWifiOff
import androidx.compose.material.icons.filled.SignalWifiOff
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError

/**
 * Card displaying an error message with appropriate icon.
 */
@Composable
fun ErrorCard(
    error: CardError,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = getErrorIcon(error),
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.error
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = getErrorTitle(error),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onErrorContainer
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = error.message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.8f),
                textAlign = TextAlign.Center
            )

            if (error.isRecoverable) {
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = getRecoveryHint(error),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.6f),
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

private fun getErrorIcon(error: CardError): ImageVector {
    return when (error) {
        is CardError.ConnectionLost -> Icons.Default.SignalWifiOff
        is CardError.Timeout -> Icons.Default.Timer
        is CardError.AuthenticationRequired,
        is CardError.AuthenticationFailed,
        is CardError.CardBlocked -> Icons.Default.Lock
        is CardError.NfcNotAvailable,
        is CardError.NfcDisabled -> Icons.Default.PortableWifiOff
        is CardError.SecurityValidationFailed -> Icons.Default.Warning
        else -> Icons.Default.Error
    }
}

private fun getErrorTitle(error: CardError): String {
    return when (error) {
        is CardError.ConnectionLost -> "Connection Lost"
        is CardError.Timeout -> "Timeout"
        is CardError.AuthenticationRequired -> "Authentication Required"
        is CardError.AuthenticationFailed -> "Authentication Failed"
        is CardError.CardBlocked -> "Card Blocked"
        is CardError.UnsupportedCard -> "Unsupported Card"
        is CardError.NfcNotAvailable -> "NFC Not Available"
        is CardError.NfcDisabled -> "NFC Disabled"
        is CardError.InvalidResponse -> "Invalid Response"
        is CardError.ParseError -> "Parse Error"
        is CardError.SecurityValidationFailed -> "Security Error"
        is CardError.IoError -> "Communication Error"
        is CardError.Unknown -> "Error"
    }
}

private fun getRecoveryHint(error: CardError): String {
    return when (error) {
        is CardError.ConnectionLost -> "Try holding the card steady on the device"
        is CardError.Timeout -> "Move the card closer and try again"
        is CardError.AuthenticationRequired -> "Provide authentication credentials"
        is CardError.AuthenticationFailed -> "Check your credentials and try again"
        is CardError.NfcDisabled -> "Enable NFC in Settings > Connections"
        is CardError.InvalidResponse -> "The card may be damaged. Try again."
        is CardError.IoError -> "Move the card closer and try again"
        else -> "Please try again"
    }
}
