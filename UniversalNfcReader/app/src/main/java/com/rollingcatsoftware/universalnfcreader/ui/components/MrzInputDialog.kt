package com.rollingcatsoftware.universalnfcreader.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Numbers
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData

/**
 * Dialog for entering MRZ (Machine Readable Zone) data for BAC authentication.
 *
 * Collects:
 * - Document Number (up to 9 characters)
 * - Date of Birth (YYMMDD format)
 * - Date of Expiry (YYMMDD format)
 *
 * Security Note: Input data is only held in memory during dialog display.
 * After authentication, the data should be cleared.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MrzInputDialog(
    onDismiss: () -> Unit,
    onAuthenticate: (AuthenticationData.MrzData) -> Unit,
    modifier: Modifier = Modifier
) {
    var documentNumber by rememberSaveable { mutableStateOf("") }
    var dateOfBirth by rememberSaveable { mutableStateOf("") }
    var dateOfExpiry by rememberSaveable { mutableStateOf("") }

    var documentNumberError by remember { mutableStateOf<String?>(null) }
    var dateOfBirthError by remember { mutableStateOf<String?>(null) }
    var dateOfExpiryError by remember { mutableStateOf<String?>(null) }

    val focusRequester = remember { FocusRequester() }

    // Request focus on first field
    LaunchedEffect(Unit) {
        focusRequester.requestFocus()
    }

    fun validateAndSubmit() {
        var isValid = true

        // Validate document number
        if (documentNumber.isBlank()) {
            documentNumberError = "Document number is required"
            isValid = false
        } else if (documentNumber.length > 9) {
            documentNumberError = "Document number must be up to 9 characters"
            isValid = false
        } else {
            documentNumberError = null
        }

        // Validate date of birth
        if (dateOfBirth.isBlank()) {
            dateOfBirthError = "Date of birth is required"
            isValid = false
        } else if (!dateOfBirth.matches(Regex("\\d{6}"))) {
            dateOfBirthError = "Must be YYMMDD format (e.g., 901231)"
            isValid = false
        } else if (!isValidDate(dateOfBirth)) {
            dateOfBirthError = "Invalid date"
            isValid = false
        } else {
            dateOfBirthError = null
        }

        // Validate date of expiry
        if (dateOfExpiry.isBlank()) {
            dateOfExpiryError = "Date of expiry is required"
            isValid = false
        } else if (!dateOfExpiry.matches(Regex("\\d{6}"))) {
            dateOfExpiryError = "Must be YYMMDD format (e.g., 301231)"
            isValid = false
        } else if (!isValidDate(dateOfExpiry)) {
            dateOfExpiryError = "Invalid date"
            isValid = false
        } else {
            dateOfExpiryError = null
        }

        if (isValid) {
            try {
                val mrzData = AuthenticationData.MrzData(
                    documentNumber = documentNumber.uppercase().trim(),
                    dateOfBirth = dateOfBirth.trim(),
                    dateOfExpiry = dateOfExpiry.trim()
                )
                onAuthenticate(mrzData)
            } catch (e: IllegalArgumentException) {
                // Handle validation error from MrzData init
                documentNumberError = e.message
            }
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            dismissOnBackPress = true,
            dismissOnClickOutside = false,
            usePlatformDefaultWidth = false
        )
    ) {
        Card(
            modifier = modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Badge,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Turkish eID Authentication",
                            style = MaterialTheme.typography.titleLarge
                        )
                    }
                    IconButton(onClick = onDismiss) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close"
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Enter the MRZ data from the back of your ID card:",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(24.dp))

                // Document Number Input
                OutlinedTextField(
                    value = documentNumber,
                    onValueChange = {
                        documentNumber = it.take(9).uppercase()
                        documentNumberError = null
                    },
                    label = { Text("Document Number") },
                    placeholder = { Text("e.g., A12345678") },
                    leadingIcon = {
                        Icon(Icons.Default.Numbers, contentDescription = null)
                    },
                    isError = documentNumberError != null,
                    supportingText = documentNumberError?.let { { Text(it) } },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Characters,
                        keyboardType = KeyboardType.Text,
                        imeAction = ImeAction.Next
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(focusRequester),
                    textStyle = MaterialTheme.typography.bodyLarge.copy(
                        fontFamily = FontFamily.Monospace
                    )
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Date of Birth Input
                OutlinedTextField(
                    value = dateOfBirth,
                    onValueChange = {
                        dateOfBirth = it.filter { c -> c.isDigit() }.take(6)
                        dateOfBirthError = null
                    },
                    label = { Text("Date of Birth") },
                    placeholder = { Text("YYMMDD (e.g., 901231)") },
                    leadingIcon = {
                        Icon(Icons.Default.CalendarMonth, contentDescription = null)
                    },
                    isError = dateOfBirthError != null,
                    supportingText = {
                        if (dateOfBirthError != null) {
                            Text(dateOfBirthError!!)
                        } else {
                            Text("Format: YYMMDD (Year-Month-Day)")
                        }
                    },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number,
                        imeAction = ImeAction.Next
                    ),
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = MaterialTheme.typography.bodyLarge.copy(
                        fontFamily = FontFamily.Monospace
                    )
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Date of Expiry Input
                OutlinedTextField(
                    value = dateOfExpiry,
                    onValueChange = {
                        dateOfExpiry = it.filter { c -> c.isDigit() }.take(6)
                        dateOfExpiryError = null
                    },
                    label = { Text("Date of Expiry") },
                    placeholder = { Text("YYMMDD (e.g., 301231)") },
                    leadingIcon = {
                        Icon(Icons.Default.CalendarMonth, contentDescription = null)
                    },
                    isError = dateOfExpiryError != null,
                    supportingText = {
                        if (dateOfExpiryError != null) {
                            Text(dateOfExpiryError!!)
                        } else {
                            Text("Format: YYMMDD (Year-Month-Day)")
                        }
                    },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number,
                        imeAction = ImeAction.Done
                    ),
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = MaterialTheme.typography.bodyLarge.copy(
                        fontFamily = FontFamily.Monospace
                    )
                )

                Spacer(modifier = Modifier.height(24.dp))

                // Security Note
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Text(
                        text = "Your MRZ data is used only for card authentication and is not stored.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(12.dp)
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Action Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    TextButton(onClick = onDismiss) {
                        Text("Cancel")
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(
                        onClick = { validateAndSubmit() },
                        enabled = documentNumber.isNotBlank() &&
                                dateOfBirth.length == 6 &&
                                dateOfExpiry.length == 6
                    ) {
                        Text("Authenticate")
                    }
                }
            }
        }
    }
}

/**
 * Validates a date string in YYMMDD format.
 */
private fun isValidDate(yymmdd: String): Boolean {
    if (yymmdd.length != 6 || !yymmdd.all { it.isDigit() }) return false

    val month = yymmdd.substring(2, 4).toIntOrNull() ?: return false
    val day = yymmdd.substring(4, 6).toIntOrNull() ?: return false

    if (month < 1 || month > 12) return false
    if (day < 1 || day > 31) return false

    // Basic validation - not accounting for month-specific days
    return true
}
