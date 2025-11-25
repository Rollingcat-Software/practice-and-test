package com.turkey.eidnfc.ui

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.scaleIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.Cake
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Event
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Nfc
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.OffsetMapping
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.TransformedText
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.PersonalData
import com.turkey.eidnfc.ui.components.LoadingSkeleton
import com.turkey.eidnfc.ui.components.PulsingDotsIndicator
import com.turkey.eidnfc.ui.scanner.MrzScannerScreen
import java.util.Calendar

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
                        contentDescription =
                            "Information card. MRZ Data Location: Find the MRZ data on the back of your Turkish ID card. Document Number is the first 9 characters. Date of Birth and Date of Expiry are in YYMMDD format."
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
 * MRZ input fields for BAC authentication with improved UX.
 * Features: Date pickers, visual formatting, auto-advance, and MRZ scanner.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MrzInputFields(
    mrzData: MrzInputData,
    onMrzDataChanged: (MrzInputData) -> Unit
) {
    val clipboardManager = LocalClipboardManager.current
    val focusManager = LocalFocusManager.current

    var showCopySnackbar by remember { mutableStateOf(false) }
    var showMrzScanner by remember { mutableStateOf(false) }
    var showDobDatePicker by remember { mutableStateOf(false) }
    var showDoeDatePicker by remember { mutableStateOf(false) }

    // Focus requesters for auto-advance
    val dobFocusRequester = remember { FocusRequester() }
    val doeFocusRequester = remember { FocusRequester() }

    // MRZ Scanner Dialog
    if (showMrzScanner) {
        MrzScannerScreen(
            onMrzScanned = { scannedData ->
                onMrzDataChanged(
                    MrzInputData(
                        documentNumber = scannedData.documentNumber,
                        dateOfBirth = scannedData.dateOfBirth,
                        dateOfExpiry = scannedData.dateOfExpiry
                    )
                )
            },
            onDismiss = { showMrzScanner = false }
        )
    }

    // Date of Birth Date Picker
    if (showDobDatePicker) {
        MrzDatePickerDialog(
            title = "Select Date of Birth",
            initialDate = mrzData.dateOfBirth,
            isBirthDate = true,
            onDateSelected = { yymmdd ->
                onMrzDataChanged(mrzData.copy(dateOfBirth = yymmdd))
            },
            onDismiss = { showDobDatePicker = false }
        )
    }

    // Date of Expiry Date Picker
    if (showDoeDatePicker) {
        MrzDatePickerDialog(
            title = "Select Date of Expiry",
            initialDate = mrzData.dateOfExpiry,
            isBirthDate = false,
            onDateSelected = { yymmdd ->
                onMrzDataChanged(mrzData.copy(dateOfExpiry = yymmdd))
            },
            onDismiss = { showDoeDatePicker = false }
        )
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics(mergeDescendants = false) {
                contentDescription =
                    "MRZ Authentication Data input section. Enter document number, date of birth, and date of expiry from your ID card."
            }
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Header with title and action buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "MRZ Authentication Data",
                    style = MaterialTheme.typography.titleMedium
                )

                // Action buttons
                Row(
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    // Scan MRZ button
                    FilledTonalIconButton(
                        onClick = { showMrzScanner = true },
                        modifier = Modifier.semantics {
                            contentDescription = "Scan MRZ with camera"
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.CameraAlt,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    if (mrzData.isComplete()) {
                        // Copy button
                        IconButton(
                            onClick = {
                                clipboardManager.setText(AnnotatedString(mrzData.toFormattedString()))
                                showCopySnackbar = true
                            }
                        ) {
                            Icon(
                                imageVector = Icons.Default.ContentCopy,
                                contentDescription = "Copy MRZ data",
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    if (mrzData.documentNumber.isNotEmpty() ||
                        mrzData.dateOfBirth.isNotEmpty() ||
                        mrzData.dateOfExpiry.isNotEmpty()
                    ) {
                        // Clear All button
                        IconButton(
                            onClick = { onMrzDataChanged(MrzInputData()) }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Clear,
                                contentDescription = "Clear all fields",
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }
            }

            // Success feedback for copy action
            AnimatedVisibility(visible = showCopySnackbar) {
                LaunchedEffect(Unit) {
                    kotlinx.coroutines.delay(2000)
                    showCopySnackbar = false
                }
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = MaterialTheme.shapes.small,
                    modifier = Modifier.padding(top = 8.dp)
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

            // Document Number with auto-advance
            OutlinedTextField(
                value = mrzData.documentNumber,
                onValueChange = { value ->
                    if (value.length <= 9 && value.all { it.isLetterOrDigit() }) {
                        val newValue = value.uppercase()
                        onMrzDataChanged(mrzData.copy(documentNumber = newValue))
                        // Auto-advance when max length reached
                        if (newValue.length == 9) {
                            dobFocusRequester.requestFocus()
                        }
                    }
                },
                label = { Text("Document Number") },
                placeholder = { Text("A12345678") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Text,
                    imeAction = ImeAction.Next
                ),
                keyboardActions = KeyboardActions(
                    onNext = { dobFocusRequester.requestFocus() }
                ),
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.Badge,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                },
                trailingIcon = {
                    if (mrzData.documentNumber.isNotEmpty()) {
                        IconButton(onClick = { onMrzDataChanged(mrzData.copy(documentNumber = "")) }) {
                            Icon(Icons.Default.Clear, "Clear document number")
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Document number. Enter 9 alphanumeric characters from your ID card."
                    },
                supportingText = {
                    Text(
                        "${mrzData.documentNumber.length}/9 characters",
                        color = if (mrzData.documentNumber.length == 9)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.onSurfaceVariant
                    )
                },
                isError = mrzData.documentNumber.isNotEmpty() && mrzData.documentNumber.length < 1
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Date of Birth with visual formatting and date picker
            OutlinedTextField(
                value = mrzData.dateOfBirth,
                onValueChange = { value ->
                    val digitsOnly = value.filter { it.isDigit() }
                    if (digitsOnly.length <= 6) {
                        onMrzDataChanged(mrzData.copy(dateOfBirth = digitsOnly))
                        // Auto-advance when complete
                        if (digitsOnly.length == 6 && isValidDate(digitsOnly)) {
                            doeFocusRequester.requestFocus()
                        }
                    }
                },
                label = { Text("Date of Birth") },
                placeholder = { Text("YY/MM/DD") },
                singleLine = true,
                visualTransformation = DateVisualTransformation(),
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Next
                ),
                keyboardActions = KeyboardActions(
                    onNext = { doeFocusRequester.requestFocus() }
                ),
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.Cake,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                },
                trailingIcon = {
                    Row {
                        // Date picker button
                        IconButton(onClick = { showDobDatePicker = true }) {
                            Icon(
                                Icons.Default.CalendarMonth,
                                contentDescription = "Open date picker"
                            )
                        }
                        if (mrzData.dateOfBirth.isNotEmpty()) {
                            IconButton(onClick = { onMrzDataChanged(mrzData.copy(dateOfBirth = "")) }) {
                                Icon(Icons.Default.Clear, "Clear date of birth")
                            }
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .focusRequester(dobFocusRequester)
                    .semantics {
                        contentDescription = "Date of birth. Enter 6 digits or tap calendar to select."
                    },
                supportingText = {
                    val formatted = formatDatePreview(mrzData.dateOfBirth)
                    when {
                        formatted != null -> Text(
                            formatted,
                            color = MaterialTheme.colorScheme.primary
                        )

                        mrzData.dateOfBirth.length == 6 && !isValidDate(mrzData.dateOfBirth) ->
                            Text(
                                "Invalid date",
                                color = MaterialTheme.colorScheme.error
                            )

                        else -> Text("${mrzData.dateOfBirth.length}/6 digits")
                    }
                },
                isError = mrzData.dateOfBirth.isNotEmpty() &&
                    mrzData.dateOfBirth.length == 6 && !isValidDate(mrzData.dateOfBirth)
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Date of Expiry with visual formatting and date picker
            OutlinedTextField(
                value = mrzData.dateOfExpiry,
                onValueChange = { value ->
                    val digitsOnly = value.filter { it.isDigit() }
                    if (digitsOnly.length <= 6) {
                        onMrzDataChanged(mrzData.copy(dateOfExpiry = digitsOnly))
                        // Hide keyboard when complete
                        if (digitsOnly.length == 6 && isValidDate(digitsOnly)) {
                            focusManager.clearFocus()
                        }
                    }
                },
                label = { Text("Date of Expiry") },
                placeholder = { Text("YY/MM/DD") },
                singleLine = true,
                visualTransformation = DateVisualTransformation(),
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Done
                ),
                keyboardActions = KeyboardActions(
                    onDone = { focusManager.clearFocus() }
                ),
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.Event,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                },
                trailingIcon = {
                    Row {
                        // Date picker button
                        IconButton(onClick = { showDoeDatePicker = true }) {
                            Icon(
                                Icons.Default.CalendarMonth,
                                contentDescription = "Open date picker"
                            )
                        }
                        if (mrzData.dateOfExpiry.isNotEmpty()) {
                            IconButton(onClick = { onMrzDataChanged(mrzData.copy(dateOfExpiry = "")) }) {
                                Icon(Icons.Default.Clear, "Clear date of expiry")
                            }
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .focusRequester(doeFocusRequester)
                    .semantics {
                        contentDescription = "Date of expiry. Enter 6 digits or tap calendar to select."
                    },
                supportingText = {
                    val formatted = formatDatePreview(mrzData.dateOfExpiry)
                    when {
                        formatted != null -> Text(
                            formatted,
                            color = MaterialTheme.colorScheme.primary
                        )

                        mrzData.dateOfExpiry.length == 6 && !isValidDate(mrzData.dateOfExpiry) ->
                            Text(
                                "Invalid date",
                                color = MaterialTheme.colorScheme.error
                            )

                        else -> Text("${mrzData.dateOfExpiry.length}/6 digits")
                    }
                },
                isError = mrzData.dateOfExpiry.isNotEmpty() &&
                    mrzData.dateOfExpiry.length == 6 && !isValidDate(mrzData.dateOfExpiry)
            )

            // Quick scan tip
            if (mrzData.documentNumber.isEmpty() &&
                mrzData.dateOfBirth.isEmpty() &&
                mrzData.dateOfExpiry.isEmpty()
            ) {
                Spacer(modifier = Modifier.height(12.dp))
                Surface(
                    onClick = { showMrzScanner = true },
                    color = MaterialTheme.colorScheme.secondaryContainer,
                    shape = MaterialTheme.shapes.small
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.CameraAlt,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSecondaryContainer,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        Column {
                            Text(
                                "Quick Entry: Scan MRZ",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                            Text(
                                "Use camera to auto-fill all fields",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Visual transformation that formats YYMMDD as YY/MM/DD.
 */
private class DateVisualTransformation : VisualTransformation {
    override fun filter(text: AnnotatedString): TransformedText {
        val trimmed = text.text.take(6)
        val formatted = buildString {
            for (i in trimmed.indices) {
                append(trimmed[i])
                if (i == 1 || i == 3) {
                    if (i < trimmed.length - 1) append("/")
                }
            }
        }

        val offsetMapping = object : OffsetMapping {
            override fun originalToTransformed(offset: Int): Int {
                return when {
                    offset <= 2 -> offset
                    offset <= 4 -> offset + 1
                    offset <= 6 -> offset + 2
                    else -> formatted.length
                }
            }

            override fun transformedToOriginal(offset: Int): Int {
                return when {
                    offset <= 2 -> offset
                    offset <= 5 -> offset - 1
                    offset <= 8 -> offset - 2
                    else -> trimmed.length
                }
            }
        }

        return TransformedText(AnnotatedString(formatted), offsetMapping)
    }
}

/**
 * Date picker dialog for MRZ date selection.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MrzDatePickerDialog(
    title: String,
    initialDate: String,
    isBirthDate: Boolean,
    onDateSelected: (String) -> Unit,
    onDismiss: () -> Unit
) {
    val calendar = Calendar.getInstance()

    // Parse initial date if valid
    if (initialDate.length == 6) {
        try {
            val yy = initialDate.substring(0, 2).toInt()
            val mm = initialDate.substring(2, 4).toInt()
            val dd = initialDate.substring(4, 6).toInt()
            val year = if (yy > 50) 1900 + yy else 2000 + yy
            calendar.set(year, mm - 1, dd)
        } catch (e: Exception) {
            // Use current date as fallback
        }
    }

    val datePickerState = rememberDatePickerState(
        initialSelectedDateMillis = calendar.timeInMillis,
        yearRange = if (isBirthDate) {
            1920..Calendar.getInstance().get(Calendar.YEAR)
        } else {
            Calendar.getInstance().get(Calendar.YEAR)..2099
        }
    )

    DatePickerDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(
                onClick = {
                    datePickerState.selectedDateMillis?.let { millis ->
                        val selectedCal = Calendar.getInstance().apply {
                            timeInMillis = millis
                        }
                        val yy = String.format("%02d", selectedCal.get(Calendar.YEAR) % 100)
                        val mm = String.format("%02d", selectedCal.get(Calendar.MONTH) + 1)
                        val dd = String.format("%02d", selectedCal.get(Calendar.DAY_OF_MONTH))
                        onDateSelected("$yy$mm$dd")
                    }
                    onDismiss()
                }
            ) {
                Text("OK")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    ) {
        DatePicker(
            state = datePickerState,
            title = {
                Text(
                    text = title,
                    modifier = Modifier.padding(start = 24.dp, top = 16.dp)
                )
            },
            headline = {
                Text(
                    text = if (isBirthDate) "Select your birth date" else "Select expiry date",
                    modifier = Modifier.padding(start = 24.dp)
                )
            }
        )
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
