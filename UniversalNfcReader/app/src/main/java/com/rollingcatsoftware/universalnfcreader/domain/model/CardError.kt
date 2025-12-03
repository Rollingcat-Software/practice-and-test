package com.rollingcatsoftware.universalnfcreader.domain.model

/**
 * Sealed class representing all possible card reading errors.
 *
 * Security Note: Error messages are sanitized to avoid leaking sensitive information.
 * Internal error details are logged separately (without PII).
 */
sealed class CardError(
    open val message: String,
    open val code: String,
    open val isRecoverable: Boolean
) {
    /**
     * NFC connection was lost during reading.
     * User action: Hold card steady on device.
     */
    data class ConnectionLost(
        override val message: String = "Connection to card was lost. Please hold the card steady.",
        override val code: String = "ERR_CONNECTION_LOST",
        override val isRecoverable: Boolean = true
    ) : CardError(message, code, isRecoverable)

    /**
     * Operation timed out.
     * User action: Try again, hold card closer.
     */
    data class Timeout(
        override val message: String = "Operation timed out. Please try again.",
        override val code: String = "ERR_TIMEOUT",
        override val isRecoverable: Boolean = true,
        val timeoutMs: Long = 0
    ) : CardError(message, code, isRecoverable)

    /**
     * Card requires authentication to read data.
     * User action: Provide credentials (MRZ, PIN, or key).
     */
    data class AuthenticationRequired(
        override val message: String = "This card requires authentication to read data.",
        override val code: String = "ERR_AUTH_REQUIRED",
        override val isRecoverable: Boolean = true,
        val authType: AuthenticationType = AuthenticationType.UNKNOWN
    ) : CardError(message, code, isRecoverable)

    /**
     * Authentication failed (wrong credentials).
     * User action: Verify credentials and try again.
     */
    data class AuthenticationFailed(
        override val message: String = "Authentication failed. Please verify your credentials.",
        override val code: String = "ERR_AUTH_FAILED",
        override val isRecoverable: Boolean = true,
        val attemptsRemaining: Int = -1
    ) : CardError(message, code, isRecoverable)

    /**
     * Card is blocked (too many wrong attempts).
     * User action: Card may need to be unblocked with PUK.
     */
    data class CardBlocked(
        override val message: String = "Card is blocked due to too many failed attempts.",
        override val code: String = "ERR_CARD_BLOCKED",
        override val isRecoverable: Boolean = false
    ) : CardError(message, code, isRecoverable)

    /**
     * Card type is not supported by this reader.
     * User action: Check supported card types.
     */
    data class UnsupportedCard(
        override val message: String = "This card type is not supported.",
        override val code: String = "ERR_UNSUPPORTED",
        override val isRecoverable: Boolean = false,
        val detectedTechnologies: List<String> = emptyList()
    ) : CardError(message, code, isRecoverable)

    /**
     * Device doesn't have NFC capability.
     * User action: Use a device with NFC.
     */
    data class NfcNotAvailable(
        override val message: String = "NFC is not available on this device.",
        override val code: String = "ERR_NFC_NOT_AVAILABLE",
        override val isRecoverable: Boolean = false
    ) : CardError(message, code, isRecoverable)

    /**
     * NFC is disabled in settings.
     * User action: Enable NFC in device settings.
     */
    data class NfcDisabled(
        override val message: String = "NFC is disabled. Please enable it in settings.",
        override val code: String = "ERR_NFC_DISABLED",
        override val isRecoverable: Boolean = true
    ) : CardError(message, code, isRecoverable)

    /**
     * Invalid APDU response from card.
     * User action: Try again, card may be damaged.
     */
    data class InvalidResponse(
        override val message: String = "Card returned an invalid response.",
        override val code: String = "ERR_INVALID_RESPONSE",
        override val isRecoverable: Boolean = true,
        val statusWord: Int = 0
    ) : CardError(message, code, isRecoverable)

    /**
     * Failed to parse card data.
     * User action: Card data may be corrupted.
     */
    data class ParseError(
        override val message: String = "Failed to parse card data.",
        override val code: String = "ERR_PARSE",
        override val isRecoverable: Boolean = false,
        val field: String = ""
    ) : CardError(message, code, isRecoverable)

    /**
     * Security validation failed (e.g., SOD signature).
     * User action: Card may be tampered or counterfeit.
     */
    data class SecurityValidationFailed(
        override val message: String = "Security validation failed. Card data may be compromised.",
        override val code: String = "ERR_SECURITY",
        override val isRecoverable: Boolean = false,
        val validationType: String = ""
    ) : CardError(message, code, isRecoverable)

    /**
     * Generic IO error during NFC operations.
     */
    data class IoError(
        override val message: String = "Communication error with the card.",
        override val code: String = "ERR_IO",
        override val isRecoverable: Boolean = true
    ) : CardError(message, code, isRecoverable)

    /**
     * Unknown error occurred.
     */
    data class Unknown(
        override val message: String = "An unexpected error occurred.",
        override val code: String = "ERR_UNKNOWN",
        override val isRecoverable: Boolean = false
    ) : CardError(message, code, isRecoverable)

    companion object {
        /**
         * Map APDU status word to appropriate error.
         *
         * Reference: ISO 7816-4 Status Words
         */
        fun fromApduStatusWord(sw: Int): CardError {
            return when {
                sw == 0x9000 -> throw IllegalArgumentException("0x9000 is success, not an error")
                sw and 0xFFF0 == 0x63C0 -> {
                    // 63CX - Wrong PIN, X attempts remaining
                    val remaining = sw and 0x000F
                    if (remaining == 0) {
                        CardBlocked()
                    } else {
                        AuthenticationFailed(
                            message = "Wrong PIN. $remaining attempts remaining.",
                            attemptsRemaining = remaining
                        )
                    }
                }

                sw == 0x6982 -> AuthenticationRequired(
                    message = "Security status not satisfied. Authentication required."
                )

                sw == 0x6983 -> CardBlocked(
                    message = "Authentication method blocked."
                )

                sw == 0x6984 -> InvalidResponse(
                    message = "Referenced data invalidated.",
                    statusWord = sw
                )

                sw == 0x6A82 -> InvalidResponse(
                    message = "File or application not found.",
                    statusWord = sw
                )

                sw == 0x6A86 -> InvalidResponse(
                    message = "Incorrect parameters P1-P2.",
                    statusWord = sw
                )

                sw == 0x6700 -> InvalidResponse(
                    message = "Wrong length.",
                    statusWord = sw
                )

                sw == 0x6E00 -> InvalidResponse(
                    message = "Class not supported.",
                    statusWord = sw
                )

                sw == 0x6D00 -> InvalidResponse(
                    message = "Instruction not supported.",
                    statusWord = sw
                )

                else -> InvalidResponse(
                    message = "Card error: ${String.format("0x%04X", sw)}",
                    statusWord = sw
                )
            }
        }
    }
}

/**
 * Types of authentication supported by different cards.
 */
enum class AuthenticationType {
    /** MRZ-based BAC authentication (eID, passport) */
    MRZ_BAC,

    /** PIN authentication */
    PIN,

    /** MIFARE key authentication (Key A or Key B) */
    MIFARE_KEY,

    /** DESFire authentication */
    DESFIRE_KEY,

    /** 3DES authentication (Ultralight C) */
    THREE_DES,

    /** Unknown authentication type */
    UNKNOWN
}
