package com.turkey.eidnfc.ui

import android.nfc.Tag
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
     * Updates the PIN value.
     * Only allows digits and max 6 characters.
     */
    fun onPinChanged(newPin: String) {
        if (newPin.all { it.isDigit() } && newPin.length <= Constants.Nfc.PIN_LENGTH) {
            _pin.value = newPin
        }
    }

    /**
     * Handles NFC tag detection.
     * Uses ReadEidCardUseCase to encapsulate business logic.
     */
    fun onTagDetected(tag: Tag) {
        val currentPin = _pin.value

        // Quick validation for immediate UI feedback
        if (!ValidatePinUseCase.validateQuick(currentPin)) {
            _uiState.value = UiState.Error("Please enter a valid 6-digit PIN")
            return
        }

        Timber.d("Tag detected, starting card reading...")
        _uiState.value = UiState.Reading

        viewModelScope.launch {
            // Use case handles validation and reading
            val params = ReadEidCardUseCase.Params(tag, currentPin)
            readEidCardUseCase(params)
                .onSuccess { cardData ->
                    Timber.d("Card read successfully via use case")
                    _uiState.value = UiState.Success(cardData)
                    // Clear PIN for security
                    clearPin()
                }
                .onError { exception ->
                    Timber.e("Card reading failed: ${exception.message}")
                    _uiState.value = UiState.Error(exception.message ?: "Unknown error occurred")
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
