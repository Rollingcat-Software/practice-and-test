package com.turkey.eidnfc.data.repository

import android.nfc.Tag
import com.turkey.eidnfc.data.nfc.NfcCardReader
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.NfcResult
import com.turkey.eidnfc.domain.model.Result
import com.turkey.eidnfc.util.isValidPin
import timber.log.Timber
import javax.inject.Inject

/**
 * Implementation of EidRepository interface.
 *
 * This class acts as a single source of truth for eID card data operations.
 * It coordinates between the NFC card reader and the rest of the application,
 * handling data transformation and error mapping.
 *
 * @param cardReader The NFC card reader for communicating with physical cards
 */
class EidRepositoryImpl @Inject constructor(
    private val cardReader: NfcCardReader
) : EidRepository {

    /**
     * Reads card data from the NFC tag with PIN authentication.
     *
     * This method:
     * 1. Validates PIN format
     * 2. Calls NFC card reader
     * 3. Transforms NfcResult to Result
     * 4. Handles errors appropriately
     *
     * @param tag The NFC tag detected by the system
     * @param pin The 6-digit PIN for authentication
     * @return Result<CardData> containing the card data on success, or an error
     */
    override suspend fun readCard(tag: Tag, pin: String): Result<CardData> {
        return try {
            // Validate PIN before attempting to read
            if (!validatePin(pin)) {
                Timber.e("Invalid PIN format provided")
                return Result.Error(
                    IllegalArgumentException("PIN must be exactly 6 digits")
                )
            }

            Timber.d("Reading card with validated PIN...")

            // Call NFC card reader
            when (val nfcResult = cardReader.readCard(tag, pin)) {
                is NfcResult.Success -> {
                    Timber.d("Successfully read card data")
                    Result.Success(nfcResult.data)
                }

                is NfcResult.Error -> {
                    val errorMessage = mapNfcErrorToMessage(nfcResult.error)
                    Timber.e("Failed to read card: $errorMessage")
                    Result.Error(Exception(errorMessage))
                }

                NfcResult.Loading -> {
                    // This shouldn't happen in the read flow, but handle it gracefully
                    Timber.w("Unexpected Loading state from card reader")
                    Result.Loading
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Exception occurred while reading card")
            Result.Error(e)
        }
    }

    /**
     * Validates if the provided PIN format is correct.
     *
     * @param pin The PIN to validate
     * @return True if PIN is valid (6 digits), false otherwise
     */
    override fun validatePin(pin: String): Boolean {
        return pin.isValidPin()
    }

    /**
     * Maps NFC error types to user-friendly error messages.
     *
     * @param error The NFC error to map
     * @return A descriptive error message
     */
    private fun mapNfcErrorToMessage(error: com.turkey.eidnfc.domain.model.NfcError): String {
        return when (error) {
            is com.turkey.eidnfc.domain.model.NfcError.WrongPin ->
                "Wrong PIN. ${error.attemptsRemaining} attempt(s) remaining."

            com.turkey.eidnfc.domain.model.NfcError.CardLocked ->
                "Card is locked. Contact authorities to unlock."

            com.turkey.eidnfc.domain.model.NfcError.SecurityNotSatisfied ->
                "Security condition not satisfied. Please verify PIN."

            com.turkey.eidnfc.domain.model.NfcError.FileNotFound ->
                "Required file not found on card."

            com.turkey.eidnfc.domain.model.NfcError.InvalidCard ->
                "This is not a valid Turkish eID card."

            com.turkey.eidnfc.domain.model.NfcError.NfcNotAvailable ->
                "NFC is not available on this device."

            com.turkey.eidnfc.domain.model.NfcError.NfcDisabled ->
                "Please enable NFC in device settings."

            com.turkey.eidnfc.domain.model.NfcError.ConnectionLost ->
                "Connection to card lost. Please try again."

            com.turkey.eidnfc.domain.model.NfcError.Timeout ->
                "Operation timed out. Please try again."

            com.turkey.eidnfc.domain.model.NfcError.ParseError ->
                "Failed to parse card data."

            is com.turkey.eidnfc.domain.model.NfcError.UnknownError ->
                "Error: ${error.message}"
        }
    }
}
