package com.rollingcatsoftware.universalnfcreader.ui

import android.nfc.Tag
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rollingcatsoftware.universalnfcreader.data.nfc.CardReadResult
import com.rollingcatsoftware.universalnfcreader.data.nfc.NfcCardReadingService
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationType
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * ViewModel for the main NFC reading screen.
 *
 * Manages UI state and coordinates with [NfcCardReadingService].
 */
@HiltViewModel
class MainViewModel @Inject constructor(
    private val nfcCardReadingService: NfcCardReadingService
) : ViewModel() {

    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    /**
     * Handle a detected NFC tag.
     */
    fun onTagDetected(tag: Tag) {
        // Check if we have pending auth data waiting for a fresh tag
        val pendingAuth = _uiState.value.pendingAuthData
        if (pendingAuth != null) {
            // User has already entered MRZ - use fresh tag for authenticated read
            performAuthenticatedRead(tag, pendingAuth.authData)
            return
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isReading = true,
                    error = null,
                    lastReadCard = null
                )
            }

            when (val result = nfcCardReadingService.readCard(tag)) {
                is CardReadResult.Success -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            lastReadCard = result.cardData,
                            readHistory = listOf(result.cardData) + it.readHistory.take(9)
                        )
                    }
                }

                is CardReadResult.Failure -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            error = result.error
                        )
                    }
                }

                is CardReadResult.AuthenticationRequired -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            authRequired = AuthRequirement(
                                cardType = result.cardType,
                                authType = result.authType,
                                pendingTag = tag
                            )
                        )
                    }
                }

                is CardReadResult.UnsupportedCard -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            error = CardError.UnsupportedCard(
                                detectedTechnologies = result.technologies
                            )
                        )
                    }
                }

                is CardReadResult.Exception -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            error = CardError.Unknown(
                                message = result.exception.message ?: "Unknown error"
                            )
                        )
                    }
                }
            }
        }
    }

    /**
     * Store authentication data and wait for card re-tap.
     *
     * NFC tags become stale after a short timeout, so we cannot use the
     * original tag reference after user enters MRZ data. Instead, we store
     * the MRZ credentials and prompt user to re-tap the card.
     */
    fun onAuthenticationProvided(authData: AuthenticationData) {
        val auth = _uiState.value.authRequired ?: return

        // Store auth data and wait for fresh tag tap
        _uiState.update {
            it.copy(
                authRequired = null,
                pendingAuthData = PendingAuthData(
                    cardType = auth.cardType,
                    authData = authData
                )
            )
        }
    }

    /**
     * Clear pending authentication and return to normal state.
     */
    fun onPendingAuthCancelled() {
        _uiState.update { it.copy(pendingAuthData = null) }
    }

    /**
     * Perform authenticated read with a fresh tag.
     */
    private fun performAuthenticatedRead(tag: Tag, authData: AuthenticationData) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isReading = true,
                    pendingAuthData = null
                )
            }

            when (val result = nfcCardReadingService.readCardWithAuth(tag, authData)) {
                is CardReadResult.Success -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            lastReadCard = result.cardData,
                            readHistory = listOf(result.cardData) + it.readHistory.take(9)
                        )
                    }
                }

                is CardReadResult.Failure -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            error = result.error
                        )
                    }
                }

                else -> {
                    _uiState.update {
                        it.copy(
                            isReading = false,
                            error = CardError.Unknown()
                        )
                    }
                }
            }
        }
    }

    /**
     * Dismiss authentication dialog.
     */
    fun onAuthenticationCancelled() {
        _uiState.update { it.copy(authRequired = null) }
    }

    /**
     * Clear the current error.
     */
    fun onErrorDismissed() {
        _uiState.update { it.copy(error = null) }
    }

    /**
     * Clear the last read card.
     */
    fun onCardDismissed() {
        _uiState.update { it.copy(lastReadCard = null) }
    }

    /**
     * Clear read history.
     */
    fun clearHistory() {
        _uiState.update { it.copy(readHistory = emptyList()) }
    }

    /**
     * Update NFC availability status.
     */
    fun updateNfcStatus(isAvailable: Boolean, isEnabled: Boolean) {
        _uiState.update {
            it.copy(
                isNfcAvailable = isAvailable,
                isNfcEnabled = isEnabled,
                error = when {
                    !isAvailable -> CardError.NfcNotAvailable()
                    !isEnabled -> CardError.NfcDisabled()
                    else -> it.error?.takeUnless { e ->
                        e is CardError.NfcNotAvailable || e is CardError.NfcDisabled
                    }
                }
            )
        }
    }

    /**
     * Request NFC status refresh.
     * This is called when user manually wants to check if NFC was enabled.
     * The actual status update happens through [updateNfcStatus] called from MainActivity.
     */
    fun refreshNfcStatus() {
        // This triggers a recomposition check
        // The actual status is updated by MainActivity.updateNfcStatus()
        // which is called from the BroadcastReceiver or onResume
        _uiState.update { it.copy() }
    }
}

/**
 * UI state for the main screen.
 */
data class MainUiState(
    val isReading: Boolean = false,
    val isNfcAvailable: Boolean = true,
    val isNfcEnabled: Boolean = true,
    val lastReadCard: CardData? = null,
    val readHistory: List<CardData> = emptyList(),
    val error: CardError? = null,
    val authRequired: AuthRequirement? = null,
    val pendingAuthData: PendingAuthData? = null
)

/**
 * Pending authentication requirement (waiting for MRZ input).
 */
data class AuthRequirement(
    val cardType: CardType,
    val authType: AuthenticationType,
    val pendingTag: Tag?
)

/**
 * Pending authentication data (MRZ entered, waiting for card re-tap).
 */
data class PendingAuthData(
    val cardType: CardType,
    val authData: AuthenticationData
)
