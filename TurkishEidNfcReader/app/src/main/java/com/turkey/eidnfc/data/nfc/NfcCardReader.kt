package com.turkey.eidnfc.data.nfc

import android.nfc.Tag
import android.nfc.tech.IsoDep
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.NfcError
import com.turkey.eidnfc.domain.model.NfcResult
import com.turkey.eidnfc.domain.model.PersonalData
import com.turkey.eidnfc.util.Dg1Parser
import com.turkey.eidnfc.util.Dg2Parser
import com.turkey.eidnfc.util.SodValidator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import timber.log.Timber
import java.io.IOException

/**
 * NFC card reader for Turkish eID cards.
 *
 * This class handles:
 * 1. IsoDep connection to the card
 * 2. Application selection (AID)
 * 3. PIN verification
 * 4. Reading data groups (DG1, DG2)
 * 5. Reading public files (EF.CardAccess, EF.SOD)
 * 6. Parsing and validating card data
 *
 * All operations are performed on IO dispatcher and with timeout protection.
 */
class NfcCardReader {

    companion object {
        private const val TIMEOUT_MS = 30000L // 30 seconds timeout
        private const val CONNECTION_TIMEOUT = 5000 // 5 seconds for connection
    }

    /**
     * Reads card data with PIN authentication.
     *
     * @param tag NFC tag from Android
     * @param pin 6-digit PIN for authentication
     * @return NfcResult with CardData or error
     */
    suspend fun readCard(tag: Tag, pin: String): NfcResult<CardData> {
        return withContext(Dispatchers.IO) {
            try {
                withTimeout(TIMEOUT_MS) {
                    readCardInternal(tag, pin)
                }
            } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
                Timber.e("Card reading timed out")
                NfcResult.Error(NfcError.Timeout)
            } catch (e: IOException) {
                Timber.e(e, "IO error during card reading")
                NfcResult.Error(NfcError.ConnectionLost)
            } catch (e: Exception) {
                Timber.e(e, "Unexpected error during card reading")
                NfcResult.Error(NfcError.UnknownError(e.message ?: "Unknown error"))
            }
        }
    }

    /**
     * Internal implementation of card reading.
     */
    private suspend fun readCardInternal(tag: Tag, pin: String): NfcResult<CardData> {
        var isoDep: IsoDep? = null

        try {
            // Get IsoDep interface
            isoDep = IsoDep.get(tag)
            if (isoDep == null) {
                Timber.e("IsoDep not supported by this tag")
                return NfcResult.Error(NfcError.InvalidCard)
            }

            // Connect to the card
            Timber.d("Connecting to card...")
            isoDep.timeout = CONNECTION_TIMEOUT
            isoDep.connect()
            Timber.d("Connected to card")

            // Select Turkish eID application
            val selectResult = selectEidApplication(isoDep)
            if (selectResult !is NfcResult.Success) {
                return selectResult as NfcResult.Error
            }

            // Read public files (no PIN required)
            val cardAccessData = readCardAccess(isoDep)
            val sodData = readSod(isoDep)

            // Verify PIN
            val pinResult = verifyPin(isoDep, pin)
            if (pinResult !is NfcResult.Success) {
                return pinResult as NfcResult.Error
            }

            // Read DG1 (personal data)
            val personalData = readDg1(isoDep)

            // Read DG2 (photo)
            val photo = readDg2(isoDep)

            // Construct CardData
            val cardData = CardData(
                personalData = personalData,
                photo = photo,
                cardAccessData = cardAccessData,
                sodData = sodData,
                isAuthenticated = true
            )

            // Validate SOD if available
            if (sodData != null) {
                validateSod(sodData)
            }

            Timber.d("Card reading completed successfully")
            NfcResult.Success(cardData)

        } catch (e: IOException) {
            Timber.e(e, "Connection lost during card reading")
            NfcResult.Error(NfcError.ConnectionLost)
        } catch (e: Exception) {
            Timber.e(e, "Error during card reading")
            NfcResult.Error(NfcError.UnknownError(e.message ?: "Unknown error"))
        } finally {
            // Always close the connection
            try {
                isoDep?.close()
                Timber.d("IsoDep connection closed")
            } catch (e: Exception) {
                Timber.e(e, "Error closing IsoDep connection")
            }
        }
    }

    /**
     * Selects the Turkish eID application by AID.
     */
    private suspend fun selectEidApplication(isoDep: IsoDep): NfcResult<Unit> {
        return try {
            val command = ApduHelper.selectEidAid()
            ApduHelper.logCommand(command)

            val response = isoDep.transceive(command)
            ApduHelper.logResponse(response)

            val (_, statusWord) = ApduHelper.parseResponse(response)

            if (ApduHelper.isSuccess(statusWord)) {
                Timber.d("Turkish eID application selected successfully")
                NfcResult.Success(Unit)
            } else {
                Timber.e("Failed to select Turkish eID application: ${ApduHelper.getStatusDescription(statusWord)}")
                NfcResult.Error(NfcError.InvalidCard)
            }
        } catch (e: Exception) {
            Timber.e(e, "Error selecting eID application")
            NfcResult.Error(NfcError.ConnectionLost)
        }
    }

    /**
     * Verifies PIN1 for authentication.
     */
    private suspend fun verifyPin(isoDep: IsoDep, pin: String): NfcResult<Unit> {
        return try {
            val command = ApduHelper.verifyPinCommand(pin)
            if (command == null) {
                return NfcResult.Error(NfcError.UnknownError("Invalid PIN format"))
            }

            ApduHelper.logCommand(command)

            val response = isoDep.transceive(command)
            ApduHelper.logResponse(response)

            val (_, statusWord) = ApduHelper.parseResponse(response)

            when {
                ApduHelper.isSuccess(statusWord) -> {
                    Timber.d("PIN verified successfully")
                    NfcResult.Success(Unit)
                }
                statusWord == ApduHelper.StatusWord.AUTH_METHOD_BLOCKED -> {
                    Timber.e("Card is blocked")
                    NfcResult.Error(NfcError.CardLocked)
                }
                else -> {
                    val remainingAttempts = ApduHelper.getRemainingPinAttempts(statusWord)
                    if (remainingAttempts >= 0) {
                        Timber.e("Wrong PIN, $remainingAttempts attempts remaining")
                        NfcResult.Error(NfcError.WrongPin(remainingAttempts))
                    } else {
                        Timber.e("PIN verification failed: ${ApduHelper.getStatusDescription(statusWord)}")
                        NfcResult.Error(NfcError.SecurityNotSatisfied)
                    }
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Error verifying PIN")
            NfcResult.Error(NfcError.ConnectionLost)
        }
    }

    /**
     * Reads EF.CardAccess file (public, no PIN required).
     */
    private suspend fun readCardAccess(isoDep: IsoDep): ByteArray? {
        return try {
            Timber.d("Reading EF.CardAccess...")
            selectAndReadFile(isoDep, ApduHelper.FileIds.EF_CARD_ACCESS)
        } catch (e: Exception) {
            Timber.w(e, "Failed to read EF.CardAccess")
            null
        }
    }

    /**
     * Reads EF.SOD (Security Object Document) file.
     */
    private suspend fun readSod(isoDep: IsoDep): ByteArray? {
        return try {
            Timber.d("Reading EF.SOD...")
            selectAndReadFile(isoDep, ApduHelper.FileIds.EF_SOD)
        } catch (e: Exception) {
            Timber.w(e, "Failed to read EF.SOD")
            null
        }
    }

    /**
     * Reads DG1 (personal data) and parses it.
     */
    private suspend fun readDg1(isoDep: IsoDep): PersonalData? {
        return try {
            Timber.d("Reading DG1 (personal data)...")
            val dg1Data = selectAndReadFile(isoDep, ApduHelper.FileIds.DG1)

            if (dg1Data != null) {
                Dg1Parser.parse(dg1Data)
            } else {
                null
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to read DG1")
            null
        }
    }

    /**
     * Reads DG2 (photo) and decodes it.
     */
    private suspend fun readDg2(isoDep: IsoDep): android.graphics.Bitmap? {
        return try {
            Timber.d("Reading DG2 (photo)...")
            val dg2Data = selectAndReadFile(isoDep, ApduHelper.FileIds.DG2)

            if (dg2Data != null) {
                Dg2Parser.parse(dg2Data)
            } else {
                null
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to read DG2")
            null
        }
    }

    /**
     * Selects a file and reads its content.
     */
    private suspend fun selectAndReadFile(isoDep: IsoDep, fileId: ByteArray): ByteArray? {
        try {
            // Select file
            val selectCommand = ApduHelper.selectCommand(fileId, isAid = false)
            ApduHelper.logCommand(selectCommand)

            val selectResponse = isoDep.transceive(selectCommand)
            ApduHelper.logResponse(selectResponse)

            val (_, selectStatus) = ApduHelper.parseResponse(selectResponse)

            if (!ApduHelper.isSuccess(selectStatus)) {
                Timber.e("Failed to select file: ${ApduHelper.getStatusDescription(selectStatus)}")
                return null
            }

            // Read file content
            val fileData = ApduHelper.readCompleteFile { command ->
                isoDep.transceive(command)
            }

            return fileData

        } catch (e: Exception) {
            Timber.e(e, "Error selecting/reading file")
            return null
        }
    }

    /**
     * Validates SOD signature and logs results.
     */
    private fun validateSod(sodData: ByteArray) {
        try {
            Timber.d("Validating SOD...")
            val result = SodValidator.validate(sodData)

            if (result.isSignatureValid) {
                Timber.d("✓ SOD signature is valid")
            } else {
                Timber.w("✗ SOD signature is invalid")
            }

            if (result.isCertificateValid) {
                Timber.d("✓ Document Signer certificate is valid")
            } else {
                Timber.w("✗ Document Signer certificate is invalid or expired")
            }

            if (result.errorMessage != null) {
                Timber.e("SOD validation error: ${result.errorMessage}")
            }
        } catch (e: Exception) {
            Timber.e(e, "Error validating SOD")
        }
    }
}
