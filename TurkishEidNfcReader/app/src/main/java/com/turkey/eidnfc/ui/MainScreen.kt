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
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
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
 * MRZ data holder for UI state.
 */
data class MrzInputData(
    val documentNumber: String = "",
    val dateOfBirth: String = "",
    val dateOfExpiry: String = ""
) {
    /**
     * Converts to pipe-separated format for backward compatibility.
     */
    fun toFormattedString(): String {
        return "$documentNumber|$dateOfBirth|$dateOfExpiry"
    }

    /**
     * Checks if all required fields are filled.
     */
    fun isComplete(): Boolean {
        return documentNumber.isNotBlank() &&
                dateOfBirth.length == 6 &&
                dateOfExpiry.length == 6
    }
}

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
 * Idle screen - shows instructions and MRZ input with animations.
 */
@Composable
fun IdleScreen(
    pin: String,
    onPinChanged: (String) -> Unit
) {
    // Parse existing pin to MRZ data
    val parts = pin.split("|")
    var mrzData by remember {
        mutableStateOf(
            if (parts.size == 3) {
                MrzInputData(parts[0], parts[1], parts[2])
            } else {
                MrzInputData()
            }
        )
    }

    // Update parent when MRZ data changes
    LaunchedEffect(mrzData) {
        onPinChanged(mrzData.toFormattedString())
    }

    // Animated entrance for elements
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        visible = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(16.dp))

        // Animated NFC icon with scale and fade
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600)) +
                    scaleIn(initialScale = 0.8f, animationSpec = tween(600))
        ) {
            Icon(
                imageVector = Icons.Default.Nfc,
                contentDescription = "NFC Icon",
                modifier = Modifier.size(80.dp),
                tint = MaterialTheme.colorScheme.primary
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

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

        Spacer(modifier = Modifier.height(8.dp))

        // Animated subtitle
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 200)) +
                    slideInVertically(initialOffsetY = { -20 }, animationSpec = tween(600, delayMillis = 200))
        ) {
            Text(
                text = "Enter your MRZ data from the back of your ID card, then hold it near the device",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Animated MRZ input fields
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 300)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(600, delayMillis = 300))
        ) {
            MrzInputFields(
                mrzData = mrzData,
                onMrzDataChanged = { mrzData = it }
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
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Information card. MRZ Data Location: Find the MRZ data on the back of your Turkish ID card. Document Number is the first 9 characters. Date of Birth and Date of Expiry are in YYMMDD format."
                    },
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = null, // Decorative, card has semantic description
                            tint = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            text = "MRZ Data Location",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Find the MRZ data on the back of your Turkish ID card:\n" +
                                "- Document Number: First 9 characters\n" +
                                "- Date of Birth: YYMMDD format\n" +
                                "- Date of Expiry: YYMMDD format",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // NFC info card
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(600, delayMillis = 500)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(600, delayMillis = 500))
        ) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = if (mrzData.isComplete()) {
                            "NFC status: Ready. You can now hold your ID card near the device to read it."
                        } else {
                            "NFC status: Not ready. Please fill in all MRZ fields before scanning your card."
                        }
                    },
                colors = CardDefaults.cardColors(
                    containerColor = if (mrzData.isComplete()) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    }
                )
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Nfc,
                        contentDescription = null, // Decorative, card has semantic description
                        tint = if (mrzData.isComplete()) {
                            MaterialTheme.colorScheme.onPrimaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = if (mrzData.isComplete()) {
                            "Ready! Hold your ID card near the device"
                        } else {
                            "Fill in all MRZ fields to read your card"
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (mrzData.isComplete()) {
                            MaterialTheme.colorScheme.onPrimaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}

/**
 * MRZ input fields for BAC authentication.
 */
@Composable
fun MrzInputFields(
    mrzData: MrzInputData,
    onMrzDataChanged: (MrzInputData) -> Unit
) {
    val clipboardManager = LocalClipboardManager.current
    var showCopySnackbar by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = false) {
                contentDescription = "MRZ Authentication Data input section. Enter document number, date of birth, and date of expiry from your ID card."
            }
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "MRZ Authentication Data",
                    style = MaterialTheme.typography.titleMedium
                )

                // Action buttons (Copy and Clear)
                if (mrzData.isComplete()) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        // Copy button
                        TextButton(
                            onClick = {
                                clipboardManager.setText(AnnotatedString(mrzData.toFormattedString()))
                                showCopySnackbar = true
                            },
                            modifier = Modifier.semantics {
                                contentDescription = "Copy MRZ data to clipboard button"
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.ContentCopy,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Copy")
                        }

                        // Clear All button
                        TextButton(
                            onClick = {
                                onMrzDataChanged(MrzInputData())
                            },
                            modifier = Modifier.semantics {
                                contentDescription = "Clear all MRZ fields button"
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Clear,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Clear")
                        }
                    }
                } else if (mrzData.documentNumber.isNotEmpty() ||
                    mrzData.dateOfBirth.isNotEmpty() ||
                    mrzData.dateOfExpiry.isNotEmpty()) {
                    // Show only clear if incomplete
                    TextButton(
                        onClick = {
                            onMrzDataChanged(MrzInputData())
                        },
                        modifier = Modifier.semantics {
                            contentDescription = "Clear all MRZ fields button"
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.Clear,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Clear")
                    }
                }
            }

            // Success feedback for copy action
            if (showCopySnackbar) {
                LaunchedEffect(Unit) {
                    kotlinx.coroutines.delay(2000)
                    showCopySnackbar = false
                }
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = MaterialTheme.shapes.small
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            "MRZ data copied to clipboard",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Document Number
            OutlinedTextField(
                value = mrzData.documentNumber,
                onValueChange = { value ->
                    // Allow alphanumeric characters, max 9 chars
                    if (value.length <= 9 && value.all { it.isLetterOrDigit() }) {
                        onMrzDataChanged(mrzData.copy(documentNumber = value.uppercase()))
                    }
                },
                label = { Text("Document Number") },
                placeholder = { Text("e.g., A12345678") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text),
                trailingIcon = {
                    if (mrzData.documentNumber.isNotEmpty()) {
                        IconButton(
                            onClick = {
                                onMrzDataChanged(mrzData.copy(documentNumber = ""))
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Clear,
                                contentDescription = "Clear document number"
                            )
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Document number input field. Enter 1 to 9 alphanumeric characters from your ID card."
                    },
                supportingText = {
                    Text("${mrzData.documentNumber.length}/9 characters")
                },
                isError = mrzData.documentNumber.isNotEmpty() && mrzData.documentNumber.length < 1
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Date of Birth
            OutlinedTextField(
                value = mrzData.dateOfBirth,
                onValueChange = { value ->
                    // Only digits, max 6
                    if (value.length <= 6 && value.all { it.isDigit() }) {
                        onMrzDataChanged(mrzData.copy(dateOfBirth = value))
                    }
                },
                label = { Text("Date of Birth") },
                placeholder = { Text("YYMMDD (e.g., 900115)") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                trailingIcon = {
                    if (mrzData.dateOfBirth.isNotEmpty()) {
                        IconButton(
                            onClick = {
                                onMrzDataChanged(mrzData.copy(dateOfBirth = ""))
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Clear,
                                contentDescription = "Clear date of birth"
                            )
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Date of birth input field. Enter 6 digits in YYMMDD format."
                    },
                supportingText = {
                    val formatted = formatDatePreview(mrzData.dateOfBirth)
                    when {
                        formatted != null -> Text("Preview: $formatted", color = MaterialTheme.colorScheme.primary)
                        mrzData.dateOfBirth.length == 6 && !isValidDate(mrzData.dateOfBirth) ->
                            Text("Invalid date. Check month (01-12) and day (01-31)", color = MaterialTheme.colorScheme.error)
                        else -> Text("${mrzData.dateOfBirth.length}/6 digits")
                    }
                },
                isError = mrzData.dateOfBirth.isNotEmpty() &&
                         (mrzData.dateOfBirth.length != 6 || !isValidDate(mrzData.dateOfBirth))
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Date of Expiry
            OutlinedTextField(
                value = mrzData.dateOfExpiry,
                onValueChange = { value ->
                    // Only digits, max 6
                    if (value.length <= 6 && value.all { it.isDigit() }) {
                        onMrzDataChanged(mrzData.copy(dateOfExpiry = value))
                    }
                },
                label = { Text("Date of Expiry") },
                placeholder = { Text("YYMMDD (e.g., 301231)") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                trailingIcon = {
                    if (mrzData.dateOfExpiry.isNotEmpty()) {
                        IconButton(
                            onClick = {
                                onMrzDataChanged(mrzData.copy(dateOfExpiry = ""))
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Clear,
                                contentDescription = "Clear date of expiry"
                            )
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Date of expiry input field. Enter 6 digits in YYMMDD format."
                    },
                supportingText = {
                    val formatted = formatDatePreview(mrzData.dateOfExpiry)
                    when {
                        formatted != null -> Text("Preview: $formatted", color = MaterialTheme.colorScheme.primary)
                        mrzData.dateOfExpiry.length == 6 && !isValidDate(mrzData.dateOfExpiry) ->
                            Text("Invalid date. Check month (01-12) and day (01-31)", color = MaterialTheme.colorScheme.error)
                        else -> Text("${mrzData.dateOfExpiry.length}/6 digits")
                    }
                },
                isError = mrzData.dateOfExpiry.isNotEmpty() &&
                         (mrzData.dateOfExpiry.length != 6 || !isValidDate(mrzData.dateOfExpiry))
            )
        }
    }
}

/**
 * Validates if a YYMMDD date string contains valid month and day values.
 */
private fun isValidDate(date: String): Boolean {
    if (date.length != 6) return false
    return try {
        val month = date.substring(2, 4).toInt()
        val day = date.substring(4, 6).toInt()
        month in 1..12 && day in 1..31
    } catch (e: Exception) {
        false
    }
}

/**
 * Formats YYMMDD date to human-readable format.
 */
private fun formatDatePreview(date: String): String? {
    if (date.length != 6) return null
    if (!isValidDate(date)) return null
    return try {
        val yy = date.substring(0, 2)
        val mm = date.substring(2, 4)
        val dd = date.substring(4, 6)
        val year = if (yy.toInt() > 50) "19$yy" else "20$yy"
        "$dd/$mm/$year"
    } catch (e: Exception) {
        null
    }
}

/**
 * Legacy PIN input field with mask toggle (kept for backward compatibility).
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
            .padding(24.dp)
            .semantics {
                contentDescription = "Reading card in progress. Please keep your card near the device."
            },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated circular progress
        CircularProgressIndicator(
            modifier = Modifier
                .size(64.dp)
                .semantics {
                    contentDescription = "Loading indicator. Card reading in progress."
                },
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
            modifier = Modifier
                .fillMaxWidth()
                .semantics {
                    contentDescription = "Loading placeholder for card data."
                }
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

        // Helpful tip about MRZ data preservation
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400, delayMillis = 350)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(400, delayMillis = 350))
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
                        text = "Your MRZ data is saved. You can read another card without re-entering.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Animated reset button
        AnimatedVisibility(
            visible = visible,
            enter = fadeIn(animationSpec = tween(400, delayMillis = 400)) +
                    slideInVertically(initialOffsetY = { 20 }, animationSpec = tween(400, delayMillis = 400))
        ) {
            Button(
                onClick = onResetClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Read another card button. Tap to scan a new ID card."
                    }
            ) {
                Icon(imageVector = Icons.Default.Refresh, contentDescription = null) // Decorative
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
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "ID card photo section displaying the cardholder's photograph."
            }
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
                contentDescription = "ID card photograph of the cardholder",
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
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = true) {
                contentDescription = "Personal Information. " +
                    "Name: ${personalData.firstName} ${personalData.lastName}. " +
                    "TCKN: ${personalData.tckn}. " +
                    "Birth Date: ${personalData.birthDate}. " +
                    "Gender: ${personalData.gender}. " +
                    "Nationality: ${personalData.nationality}. " +
                    "Document Number: ${personalData.documentNumber}. " +
                    "Issue Date: ${personalData.issueDate}. " +
                    "Expiry Date: ${personalData.expiryDate}. " +
                    "Place of Birth: ${personalData.placeOfBirth}."
            }
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
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Try again button. Tap to retry reading your ID card."
                    }
            ) {
                Icon(imageVector = Icons.Default.Refresh, contentDescription = null) // Decorative
                Spacer(modifier = Modifier.width(8.dp))
                Text("Try Again")
            }
        }
    }
}
