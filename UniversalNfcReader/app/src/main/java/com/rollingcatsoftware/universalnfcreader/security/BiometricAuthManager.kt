package com.rollingcatsoftware.universalnfcreader.security

import android.content.Context
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricManager.Authenticators.BIOMETRIC_STRONG
import androidx.biometric.BiometricManager.Authenticators.BIOMETRIC_WEAK
import androidx.biometric.BiometricManager.Authenticators.DEVICE_CREDENTIAL
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Manager for biometric authentication (fingerprint, face, etc.)
 *
 * Handles:
 * - Checking biometric availability
 * - Showing biometric prompt
 * - Fallback to device credentials if biometrics unavailable
 * - Graceful handling when no authentication is available
 */
class BiometricAuthManager(private val context: Context) {

    companion object {
        private const val TAG = "BiometricAuthManager"

        // Allow strong biometrics, weak biometrics, or device credential (PIN/pattern/password)
        private const val AUTHENTICATORS = BIOMETRIC_STRONG or BIOMETRIC_WEAK or DEVICE_CREDENTIAL
    }

    private val biometricManager = BiometricManager.from(context)

    private val _authenticationState =
        MutableStateFlow<AuthenticationState>(AuthenticationState.Idle)
    val authenticationState: StateFlow<AuthenticationState> = _authenticationState.asStateFlow()

    /**
     * Result of biometric capability check
     */
    sealed class BiometricCapability {
        /** Device supports biometric authentication */
        data object Available : BiometricCapability()

        /** No biometric hardware available */
        data object NoHardware : BiometricCapability()

        /** Biometric hardware unavailable (maybe disabled) */
        data object HardwareUnavailable : BiometricCapability()

        /** No biometrics enrolled */
        data object NoneEnrolled : BiometricCapability()

        /** Unknown status */
        data object Unknown : BiometricCapability()
    }

    /**
     * State of authentication
     */
    sealed class AuthenticationState {
        data object Idle : AuthenticationState()
        data object Authenticating : AuthenticationState()
        data object Authenticated : AuthenticationState()
        data class Failed(val message: String) : AuthenticationState()
        data object Cancelled : AuthenticationState()
    }

    /**
     * Check if biometric authentication is available on this device.
     */
    fun checkBiometricCapability(): BiometricCapability {
        return when (biometricManager.canAuthenticate(AUTHENTICATORS)) {
            BiometricManager.BIOMETRIC_SUCCESS -> {
                SecureLogger.d(TAG, "Biometric authentication is available")
                BiometricCapability.Available
            }

            BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE -> {
                SecureLogger.d(TAG, "No biometric hardware available")
                BiometricCapability.NoHardware
            }

            BiometricManager.BIOMETRIC_ERROR_HW_UNAVAILABLE -> {
                SecureLogger.d(TAG, "Biometric hardware unavailable")
                BiometricCapability.HardwareUnavailable
            }

            BiometricManager.BIOMETRIC_ERROR_NONE_ENROLLED -> {
                SecureLogger.d(TAG, "No biometrics enrolled")
                BiometricCapability.NoneEnrolled
            }

            else -> {
                SecureLogger.d(TAG, "Unknown biometric status")
                BiometricCapability.Unknown
            }
        }
    }

    /**
     * Check if authentication is required (i.e., device has biometric capability)
     */
    fun isAuthenticationRequired(): Boolean {
        return checkBiometricCapability() == BiometricCapability.Available
    }

    /**
     * Show biometric authentication prompt.
     *
     * @param activity The FragmentActivity to show the prompt in
     * @param onSuccess Called when authentication succeeds
     * @param onError Called when authentication fails
     * @param onCancel Called when user cancels authentication
     */
    fun authenticate(
        activity: FragmentActivity,
        onSuccess: () -> Unit,
        onError: (String) -> Unit,
        onCancel: () -> Unit
    ) {
        _authenticationState.value = AuthenticationState.Authenticating

        val executor = ContextCompat.getMainExecutor(context)

        val callback = object : BiometricPrompt.AuthenticationCallback() {
            override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                super.onAuthenticationSucceeded(result)
                SecureLogger.d(TAG, "Authentication succeeded")
                _authenticationState.value = AuthenticationState.Authenticated
                onSuccess()
            }

            override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                super.onAuthenticationError(errorCode, errString)
                SecureLogger.e(TAG, "Authentication error: $errorCode - $errString")

                when (errorCode) {
                    BiometricPrompt.ERROR_USER_CANCELED,
                    BiometricPrompt.ERROR_NEGATIVE_BUTTON,
                    BiometricPrompt.ERROR_CANCELED -> {
                        _authenticationState.value = AuthenticationState.Cancelled
                        onCancel()
                    }

                    BiometricPrompt.ERROR_LOCKOUT,
                    BiometricPrompt.ERROR_LOCKOUT_PERMANENT -> {
                        _authenticationState.value =
                            AuthenticationState.Failed("Too many failed attempts. Try again later.")
                        onError("Too many failed attempts. Try again later.")
                    }

                    else -> {
                        _authenticationState.value =
                            AuthenticationState.Failed(errString.toString())
                        onError(errString.toString())
                    }
                }
            }

            override fun onAuthenticationFailed() {
                super.onAuthenticationFailed()
                SecureLogger.w(TAG, "Authentication failed (biometric not recognized)")
                // Don't update state here - user can try again
            }
        }

        val biometricPrompt = BiometricPrompt(activity, executor, callback)

        val promptInfo = BiometricPrompt.PromptInfo.Builder()
            .setTitle("Authenticate to continue")
            .setSubtitle("Universal NFC Reader")
            .setDescription("Verify your identity to access the app")
            .setAllowedAuthenticators(AUTHENTICATORS)
            .build()

        try {
            biometricPrompt.authenticate(promptInfo)
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to show biometric prompt", e)
            _authenticationState.value = AuthenticationState.Failed(e.message ?: "Unknown error")
            onError(e.message ?: "Failed to show authentication prompt")
        }
    }

    /**
     * Reset authentication state to idle.
     */
    fun resetState() {
        _authenticationState.value = AuthenticationState.Idle
    }

    /**
     * Mark as authenticated (for when biometrics not available)
     */
    fun markAsAuthenticated() {
        _authenticationState.value = AuthenticationState.Authenticated
    }
}
