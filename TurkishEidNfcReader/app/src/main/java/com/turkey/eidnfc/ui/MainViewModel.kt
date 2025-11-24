package com.turkey.eidnfc.ui

import android.nfc.Tag
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.turkey.eidnfc.data.nfc.NfcCardReader
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.NfcError
import com.turkey.eidnfc.domain.model.NfcResult
import com.turkey.eidnfc.domain.model.toUserMessage
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import timber.log.Timber

/**
 * ViewModel for the main screen.
 *
 * Manages UI state and coordinates NFC card reading operations.
 */
class MainViewModel : ViewModel() {

    private val cardReader = NfcCardReader()

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _pin = MutableStateFlow("")
    val pin: StateFlow<String> = _pin.asStateFlow()

    /**
     * Updates the PIN value.
     */
    fun onPinChanged(newPin: String) {
        // Only allow digits and max 6 characters
        if (newPin.all { it.isDigit() } && newPin.length <= 6) {
            _pin.value = newPin
        }
    }

    /**
     * Handles NFC tag detection.
     */
    fun onTagDetected(tag: Tag) {
        val currentPin = _pin.value

        if (currentPin.length != 6) {
            _uiState.value = UiState.Error("Please enter a 6-digit PIN")
            return
        }

        Timber.d("Tag detected, starting card reading...")
        _uiState.value = UiState.Reading

        viewModelScope.launch {
            when (val result = cardReader.readCard(tag, currentPin)) {
                is NfcResult.Success -> {
                    Timber.d("Card read successfully")
                    _uiState.value = UiState.Success(result.data)
                    // Clear PIN for security
                    clearPin()
                }
                is NfcResult.Error -> {
                    Timber.e("Card reading failed: ${result.error}")
                    _uiState.value = UiState.Error(result.error.toUserMessage())
                }
                NfcResult.Loading -> {
                    // Should not happen in this flow
                    _uiState.value = UiState.Reading
                }
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
     * Clears the PIN from memory for security.
     */
    private fun clearPin() {
        _pin.value = ""
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
