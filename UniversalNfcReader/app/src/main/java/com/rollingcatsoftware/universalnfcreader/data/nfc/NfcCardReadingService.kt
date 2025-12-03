package com.rollingcatsoftware.universalnfcreader.data.nfc

import android.nfc.Tag
import android.util.Log
import com.rollingcatsoftware.universalnfcreader.data.nfc.detector.CardDetector
import com.rollingcatsoftware.universalnfcreader.data.nfc.detector.UniversalCardDetector
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationType
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Main service for orchestrating NFC card reading operations.
 *
 * This service coordinates:
 * 1. Card type detection using [CardDetector]
 * 2. Reader selection using [CardReaderFactory]
 * 3. Card reading with appropriate reader
 * 4. Result handling and error mapping
 *
 * Usage:
 * ```kotlin
 * val service = NfcCardReadingService(detector, factory)
 *
 * // Read card (auto-detect type)
 * val result = service.readCard(tag)
 *
 * // Read with authentication
 * val authResult = service.readCardWithAuth(tag, mrzData)
 * ```
 */
@Singleton
class NfcCardReadingService @Inject constructor(
    private val detector: CardDetector,
    private val factory: CardReaderFactory
) {

    companion object {
        private const val TAG = "NfcCardReadingService"
    }

    /**
     * Secondary constructor for manual instantiation.
     */
    constructor() : this(UniversalCardDetector(), CardReaderFactory())

    /**
     * Read card data without authentication.
     *
     * Automatically detects card type and uses appropriate reader.
     *
     * @param tag The NFC tag to read
     * @return [CardReadResult] containing card data or error
     */
    suspend fun readCard(tag: Tag): CardReadResult = withContext(Dispatchers.IO) {
        Log.d(TAG, "Starting card read operation")

        try {
            // Step 1: Detect card type
            val cardType = detector.detectCardType(tag)
            val technologies = detector.getSupportedTechnologies(tag)
            Log.d(TAG, "Detected card type: $cardType")

            // Step 2: Get appropriate reader
            val reader = factory.createReader(cardType)
            if (reader == null) {
                Log.w(TAG, "No reader available for card type: $cardType")
                return@withContext CardReadResult.UnsupportedCard(
                    cardType = cardType,
                    technologies = technologies.map { it.substringAfterLast('.') }
                )
            }

            // Step 3: Check if authentication is required
            if (reader.requiresAuthentication()) {
                Log.d(TAG, "Card requires authentication")
                return@withContext CardReadResult.AuthenticationRequired(
                    cardType = cardType,
                    authType = getAuthType(cardType)
                )
            }

            // Step 4: Read card
            return@withContext when (val result = reader.readCard(tag)) {
                is Result.Success -> {
                    Log.d(TAG, "Card read successful: ${result.data.uid}")
                    CardReadResult.Success(result.data)
                }

                is Result.Error -> {
                    Log.e(TAG, "Card read failed: ${result.error.message}")
                    CardReadResult.Failure(cardType, result.error)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during card read: ${e.message}", e)
            CardReadResult.Exception(e)
        }
    }

    /**
     * Read card data with authentication.
     *
     * @param tag The NFC tag to read
     * @param authData Authentication credentials
     * @return [CardReadResult] containing card data or error
     */
    suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): CardReadResult = withContext(Dispatchers.IO) {
        Log.d(TAG, "Starting authenticated card read")

        try {
            // Step 1: Detect card type
            val cardType = detector.detectCardType(tag)
            Log.d(TAG, "Detected card type: $cardType")

            // Step 2: Get appropriate reader
            val reader = factory.createReader(cardType)
            if (reader == null) {
                Log.w(TAG, "No reader available for card type: $cardType")
                return@withContext CardReadResult.UnsupportedCard(
                    cardType = cardType,
                    technologies = detector.getTechnologyNames(tag)
                )
            }

            // Step 3: Read with authentication
            return@withContext when (val result = reader.readCardWithAuth(tag, authData)) {
                is Result.Success -> {
                    Log.d(TAG, "Authenticated read successful: ${result.data.uid}")
                    CardReadResult.Success(result.data)
                }

                is Result.Error -> {
                    Log.e(TAG, "Authenticated read failed: ${result.error.message}")
                    CardReadResult.Failure(cardType, result.error)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during authenticated read: ${e.message}", e)
            CardReadResult.Exception(e)
        } finally {
            // Clear sensitive data from authentication object
            authData.clear()
        }
    }

    /**
     * Detect card type without reading.
     *
     * Useful for showing UI before reading.
     *
     * @param tag The NFC tag to analyze
     * @return Detected [CardType]
     */
    suspend fun detectCardType(tag: Tag): CardType = withContext(Dispatchers.IO) {
        detector.detectCardType(tag)
    }

    /**
     * Get supported technologies from tag.
     */
    fun getSupportedTechnologies(tag: Tag): List<String> {
        return detector.getTechnologyNames(tag)
    }

    /**
     * Check if a card type is supported by this service.
     */
    fun isCardTypeSupported(cardType: CardType): Boolean {
        return factory.isSupported(cardType)
    }

    /**
     * Get the authentication type required for a card type.
     */
    private fun getAuthType(cardType: CardType): AuthenticationType {
        return when (cardType) {
            CardType.TURKISH_EID -> AuthenticationType.MRZ_BAC
            CardType.MIFARE_CLASSIC_1K,
            CardType.MIFARE_CLASSIC_4K,
            CardType.STUDENT_CARD_CLASSIC -> AuthenticationType.MIFARE_KEY

            CardType.MIFARE_DESFIRE,
            CardType.ISTANBULKART,
            CardType.STUDENT_CARD_DESFIRE -> AuthenticationType.DESFIRE_KEY

            CardType.MIFARE_ULTRALIGHT_C -> AuthenticationType.THREE_DES
            else -> AuthenticationType.UNKNOWN
        }
    }
}

/**
 * Sealed class representing all possible outcomes of a card read operation.
 */
sealed class CardReadResult {
    /**
     * Card was read successfully.
     */
    data class Success(val cardData: CardData) : CardReadResult()

    /**
     * Card read failed with a recoverable or non-recoverable error.
     */
    data class Failure(
        val cardType: CardType,
        val error: CardError
    ) : CardReadResult()

    /**
     * Card requires authentication to read.
     * UI should prompt for credentials.
     */
    data class AuthenticationRequired(
        val cardType: CardType,
        val authType: AuthenticationType
    ) : CardReadResult()

    /**
     * Card type is not supported by any reader.
     */
    data class UnsupportedCard(
        val cardType: CardType,
        val technologies: List<String>
    ) : CardReadResult()

    /**
     * Unexpected exception occurred.
     */
    data class Exception(val exception: kotlin.Exception) : CardReadResult()

    /**
     * Check if the result is a success.
     */
    val isSuccess: Boolean get() = this is Success

    /**
     * Get card data if successful, null otherwise.
     */
    fun getDataOrNull(): CardData? = (this as? Success)?.cardData

    /**
     * Get error if failure, null otherwise.
     */
    fun getErrorOrNull(): CardError? = (this as? Failure)?.error
}
