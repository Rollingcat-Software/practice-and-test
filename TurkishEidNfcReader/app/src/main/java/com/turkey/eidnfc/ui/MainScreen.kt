package com.turkey.eidnfc.ui

import android.graphics.Bitmap
import androidx.compose.animation.*
import androidx.compose.animation.core.*
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
import com.turkey.eidnfc.ui.components.LoadingSkeleton
import com.turkey.eidnfc.ui.components.PulsingDotsIndicator

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
            // Crossfade animation for smooth state transitions
            Crossfade(
                targetState = uiState,
                animationSpec = tween(durationMillis = 300),
                label = "state_transition"
            ) { state ->
                when (state) {
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
                            cardData = state.cardData,
                            onResetClick = onResetClick
                        )
                    }
                    is MainViewModel.UiState.Error -> {
                        ErrorScreen(
                            message = state.message,
                            onResetClick = onResetClick
                        )
                    }
                }
            }
        }
    }
}

/**
 * Idle screen - shows instructions and PIN input with animations.
 */
@Composable
fun IdleScreen(
    pin: String,
    onPinChanged: (String) -> Unit
) {
    // Animated entrance for elements
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        visible = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated NFC icon with scale and fade
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600)) +
                    scaleIn(initialScale = 0.8f, animationSpec = tween(600))
        ) {
            Icon(
                imageVector = Icons.Default.Nfc,
                contentDescription = "NFC Icon",
                modifier = Modifier.size(120.dp),
                tint = MaterialTheme.colorScheme.primary
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Animated title with slide
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 100)) +
                    slideInVertically(initialOffsetY = { -20 }, animationSpec = tween(600, delayMillis = 100))
        ) {
            Text(
                text = "Turkish eID Card Reader",
                style = MaterialTheme.typography.headlineMedium,
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Animated subtitle
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 200)) +
                    slideInVertically(initialOffsetY = { -20 }, animationSpec = tween(600, delayMillis = 200))
        ) {
            Text(
                text = "Enter your 6-digit PIN and hold your ID card near the device",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Animated PIN input field
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 300)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(600, delayMillis = 300))
        ) {
            PinInputField(
                pin = pin,
                onPinChanged = onPinChanged
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Animated info card
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 400)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(600, delayMillis = 400))
        ) {
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
 * Reading screen - shows loading indicator with animations and skeleton.
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
        // Animated circular progress
        CircularProgressIndicator(
            modifier = Modifier.size(64.dp),
            strokeWidth = 6.dp
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Animated text with pulsing dots
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

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Keep your card near the device",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Loading skeleton
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            LoadingSkeleton()
        }
    }
}

/**
 * Success screen - displays card data with animations.
 */
@Composable
fun SuccessScreen(
    cardData: CardData,
    onResetClick: () -> Unit
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        visible = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Animated success header
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400)) +
                    slideInVertically(initialOffsetY = { -40 }, animationSpec = tween(400))
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

        Spacer(modifier = Modifier.height(16.dp))

        // Animated photo
        cardData.photo?.let { photo ->
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(animationSpec = tween(400, delayMillis = 150)) +
                        scaleIn(initialScale = 0.9f, animationSpec = tween(400, delayMillis = 150))
            ) {
                Column {
                    PhotoCard(photo)
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
        }

        // Animated personal data
        cardData.personalData?.let { personalData ->
            AnimatedVisibility(
                visible = visible,
                enter = fadeIn(animationSpec = tween(400, delayMillis = 250)) +
                        slideInVertically(initialOffsetY = { 40 }, animationSpec = tween(400, delayMillis = 250))
            ) {
                Column {
                    PersonalDataCard(personalData)
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }
        }

        // Animated reset button
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400, delayMillis = 350)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(400, delayMillis = 350))
        ) {
            Button(
                onClick = onResetClick,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(imageVector = Icons.Default.Refresh, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Read Another Card")
            }
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
 * Error screen - displays error message with animations.
 */
@Composable
fun ErrorScreen(
    message: String,
    onResetClick: () -> Unit
) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        visible = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated error icon with shake effect
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400)) +
                    scaleIn(initialScale = 0.7f, animationSpec = tween(400, easing = FastOutSlowInEasing))
        ) {
            Icon(
                imageVector = Icons.Default.Error,
                contentDescription = "Error",
                modifier = Modifier.size(80.dp),
                tint = MaterialTheme.colorScheme.error
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Animated error title
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400, delayMillis = 150)) +
                    slideInVertically(initialOffsetY = { -20 }, animationSpec = tween(400, delayMillis = 150))
        ) {
            Text(
                text = "Error Reading Card",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.error
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Animated error message card
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400, delayMillis = 250)) +
                    expandVertically(animationSpec = tween(400, delayMillis = 250))
        ) {
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
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Animated try again button
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400, delayMillis = 350)) +
                    slideInVertically(initialOffsetY = { 40 }, animationSpec = tween(400, delayMillis = 350))
        ) {
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
}
