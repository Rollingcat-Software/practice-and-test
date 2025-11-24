package com.turkey.eidnfc.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.PersonalData

/**
 * Main screen of the Turkish eID NFC Reader app.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    uiState: MainViewModel.UiState,
    pin: String,
    onPinChanged: (String) -> Unit,
    onResetClick: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Turkish eID NFC Reader") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when (uiState) {
                MainViewModel.UiState.Idle -> {
                    IdleScreen(
                        pin = pin,
                        onPinChanged = onPinChanged
                    )
                }
                MainViewModel.UiState.Reading -> {
                    ReadingScreen()
                }
                is MainViewModel.UiState.Success -> {
                    SuccessScreen(
                        cardData = uiState.cardData,
                        onResetClick = onResetClick
                    )
                }
                is MainViewModel.UiState.Error -> {
                    ErrorScreen(
                        message = uiState.message,
                        onResetClick = onResetClick
                    )
                }
            }
        }
    }
}

/**
 * Idle screen - shows instructions and PIN input.
 */
@Composable
fun IdleScreen(
    pin: String,
    onPinChanged: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.Nfc,
            contentDescription = "NFC Icon",
            modifier = Modifier.size(120.dp),
            tint = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            text = "Turkish eID Card Reader",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Enter your 6-digit PIN and hold your ID card near the device",
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(32.dp))

        PinInputField(
            pin = pin,
            onPinChanged = onPinChanged
        )

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer
            )
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = "Info",
                    tint = MaterialTheme.colorScheme.onSecondaryContainer
                )
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "Make sure NFC is enabled in your device settings",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer
                )
            }
        }
    }
}

/**
 * PIN input field with mask toggle.
 */
@Composable
fun PinInputField(
    pin: String,
    onPinChanged: (String) -> Unit
) {
    var showPin by remember { mutableStateOf(false) }

    OutlinedTextField(
        value = pin,
        onValueChange = onPinChanged,
        label = { Text("PIN (6 digits)") },
        placeholder = { Text("Enter your PIN") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
        visualTransformation = if (showPin) {
            VisualTransformation.None
        } else {
            PasswordVisualTransformation()
        },
        trailingIcon = {
            IconButton(onClick = { showPin = !showPin }) {
                Icon(
                    imageVector = if (showPin) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                    contentDescription = if (showPin) "Hide PIN" else "Show PIN"
                )
            }
        },
        modifier = Modifier.fillMaxWidth(),
        supportingText = {
            Text("${pin.length}/6")
        },
        isError = pin.isNotEmpty() && pin.length != 6
    )
}

/**
 * Reading screen - shows loading indicator.
 */
@Composable
fun ReadingScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(64.dp)
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Reading card...",
            style = MaterialTheme.typography.titleLarge
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Keep your card near the device",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
    }
}

/**
 * Success screen - displays card data.
 */
@Composable
fun SuccessScreen(
    cardData: CardData,
    onResetClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Success header
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

        Spacer(modifier = Modifier.height(16.dp))

        // Photo
        cardData.photo?.let { photo ->
            PhotoCard(photo)
            Spacer(modifier = Modifier.height(16.dp))
        }

        // Personal data
        cardData.personalData?.let { personalData ->
            PersonalDataCard(personalData)
            Spacer(modifier = Modifier.height(16.dp))
        }

        // Reset button
        Button(
            onClick = onResetClick,
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(imageVector = Icons.Default.Refresh, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Read Another Card")
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

/**
 * Photo card component.
 */
@Composable
fun PhotoCard(photo: Bitmap) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Photo",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.align(Alignment.Start)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Image(
                bitmap = photo.asImageBitmap(),
                contentDescription = "ID Photo",
                modifier = Modifier.size(200.dp)
            )
        }
    }
}

/**
 * Personal data card component.
 */
@Composable
fun PersonalDataCard(personalData: PersonalData) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Personal Information",
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(16.dp))

            DataRow("Name", "${personalData.firstName} ${personalData.lastName}")
            DataRow("TCKN", personalData.tckn)
            DataRow("Birth Date", personalData.birthDate)
            DataRow("Gender", personalData.gender)
            DataRow("Nationality", personalData.nationality)
            DataRow("Document Number", personalData.documentNumber)
            DataRow("Issue Date", personalData.issueDate)
            DataRow("Expiry Date", personalData.expiryDate)
            DataRow("Place of Birth", personalData.placeOfBirth)
        }
    }
}

/**
 * Data row component for displaying key-value pairs.
 */
@Composable
fun DataRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Text(
            text = "$label:",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1.5f)
        )
    }
}

/**
 * Error screen - displays error message.
 */
@Composable
fun ErrorScreen(
    message: String,
    onResetClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.Error,
            contentDescription = "Error",
            modifier = Modifier.size(80.dp),
            tint = MaterialTheme.colorScheme.error
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Error Reading Card",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.error
        )

        Spacer(modifier = Modifier.height(16.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer
            )
        ) {
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onErrorContainer,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(16.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = onResetClick,
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(imageVector = Icons.Default.Refresh, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Try Again")
        }
    }
}
