package com.turkey.eidnfc.ui

import android.nfc.Tag
import android.nfc.TagLostException
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.onError
import com.turkey.eidnfc.domain.model.onSuccess
import com.turkey.eidnfc.domain.usecase.ReadEidCardUseCase
import com.turkey.eidnfc.domain.usecase.ValidatePinUseCase
import com.turkey.eidnfc.util.Constants
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.IOException
import javax.inject.Inject

/**
 * ViewModel for the main screen.
 *
 * Manages UI state and coordinates NFC card reading operations through use cases.
 * This follows Clean Architecture by separating:
 * - Presentation (ViewModel)
 * - Domain (Use Cases)
 * - Data (Repository)
 */
@HiltViewModel
class MainViewModel @Inject constructor(
    private val readEidCardUseCase: ReadEidCardUseCase,
    private val validatePinUseCase: ValidatePinUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _pin = MutableStateFlow("")
    val pin: StateFlow<String> = _pin.asStateFlow()

    /**
     * Updates the MRZ data value.
     * Accepts format: documentNumber|dateOfBirth|dateOfExpiry
     * Example: A12345678|900115|301231
     */
    fun onPinChanged(newValue: String) {
        // Allow alphanumeric characters, digits, and pipe separator for MRZ format
        _pin.value = newValue
    }

    /**
     * Handles NFC tag detection.
     * Uses ReadEidCardUseCase to encapsulate business logic.
     */
    fun onTagDetected(tag: Tag) {
        val currentMrzData = _pin.value

        // Quick validation for immediate UI feedback
        if (!ValidatePinUseCase.validateQuick(currentMrzData)) {
            _uiState.value = UiState.Error(
                "Please enter valid MRZ data:\n" +
                "- Document Number (1-9 characters)\n" +
                "- Date of Birth (YYMMDD)\n" +
                "- Date of Expiry (YYMMDD)"
            )
            return
        }

        Timber.d("Tag detected, starting card reading...")
        _uiState.value = UiState.Reading

        viewModelScope.launch {
            // Use case handles validation and reading
            val params = ReadEidCardUseCase.Params(tag, currentMrzData)
            readEidCardUseCase(params)
                .onSuccess { cardData ->
                    Timber.d("Card read successfully via use case")
                    _uiState.value = UiState.Success(cardData)
                    // Clear PIN for security
                    clearPin()
                }
                .onError { exception ->
                    Timber.e("Card reading failed: ${exception.message}")
                    _uiState.value = UiState.Error(getActionableErrorMessage(exception))
                }
        }
    }

    /**
     * Resets the UI state to idle.
     */
    fun resetState() {
        _uiState.value = UiState.Idle
        clearPin()
    }

    /**
     * Clears the MRZ data from memory for security.
     */
    private fun clearPin() {
        _pin.value = ""
    }

    /**
     * Maps exceptions to user-friendly, actionable error messages.
     *
     * Provides contextual guidance based on the error type to help users
     * understand what went wrong and how to fix it.
     */
    private fun getActionableErrorMessage(exception: Exception): String {
        // Check exception type first
        return when (exception) {
            is TagLostException -> {
                "Card connection lost.\n\n" +
                "✓ Hold your card steady against the device\n" +
                "✓ Don't move the card until reading completes\n" +
                "✓ Try again"
            }

            is IOException -> {
                "Communication error with card.\n\n" +
                "✓ Ensure your card is touching the NFC area\n" +
                "✓ Remove any metal cases or thick covers\n" +
                "✓ Try cleaning the card surface\n" +
                "✓ Try again"
            }

            is IllegalArgumentException -> {
                // These usually come from validation errors with good messages
                exception.message ?: "Invalid input data. Please check your MRZ data."
            }

            is SecurityException -> {
                "Security error.\n\n" +
                "✓ Verify your MRZ data is correct\n" +
                "✓ Check document number, birth date, and expiry date\n" +
                "✓ Ensure dates match your ID card exactly"
            }

            else -> {
                // Check message content for known patterns
                val message = exception.message ?: ""
                when {
                    message.contains("MRZ", ignoreCase = true) ||
                    message.contains("BAC", ignoreCase = true) -> {
                        "Authentication failed.\n\n" +
                        "✓ Double-check your MRZ data from the card\n" +
                        "✓ Verify document number (9 characters)\n" +
                        "✓ Verify birth date and expiry date (YYMMDD)\n" +
                        "✓ Make sure dates are entered correctly"
                    }

                    message.contains("connection", ignoreCase = true) ||
                    message.contains("lost", ignoreCase = true) -> {
                        "Card connection interrupted.\n\n" +
                        "✓ Keep the card steady on the device\n" +
                        "✓ Don't move until reading completes\n" +
                        "✓ Try again"
                    }

                    message.contains("timeout", ignoreCase = true) -> {
                        "Reading timed out.\n\n" +
                        "✓ Keep the card on the device longer\n" +
                        "✓ Ensure good contact with NFC area\n" +
                        "✓ Try again"
                    }

                    message.contains("NFC", ignoreCase = true) -> {
                        "NFC error.\n\n" +
                        "✓ Ensure NFC is enabled in Settings\n" +
                        "✓ Check if your device supports NFC\n" +
                        "✓ Try restarting NFC in device settings"
                    }

                    message.isNotEmpty() -> {
                        // Use the original message if it's descriptive enough
                        "$message\n\n" +
                        "✓ Verify your MRZ data is correct\n" +
                        "✓ Try again with the card held steady"
                    }

                    else -> {
                        "Unable to read card.\n\n" +
                        "✓ Check that your MRZ data is correct\n" +
                        "✓ Ensure NFC is enabled\n" +
                        "✓ Hold the card steady on the device\n" +
                        "✓ Try again"
                    }
                }
            }
        }
    }

    /**
     * UI state sealed class.
     */
    sealed class UiState {
        data object Idle : UiState()
        data object Reading : UiState()
        data class Success(val cardData: CardData) : UiState()
        data class Error(val message: String) : UiState()
    }
}
