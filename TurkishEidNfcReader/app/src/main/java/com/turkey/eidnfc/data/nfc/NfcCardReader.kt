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
 * NFC card reader for Turkish eID cards (ICAO MRTD compliant).
 *
 * This class handles:
 * 1. IsoDep connection to the card
 * 2. Application selection (AID)
 * 3. BAC (Basic Access Control) authentication using MRZ data
 * 4. Secure messaging for encrypted communication
 * 5. Reading data groups (DG1, DG2)
 * 6. Reading public files (EF.CardAccess, EF.SOD)
 * 7. Parsing and validating card data
 *
 * All operations are performed on IO dispatcher and with timeout protection.
 */
class NfcCardReader {

    companion object {
        private const val TIMEOUT_MS = 30000L // 30 seconds timeout
        private const val CONNECTION_TIMEOUT = 5000 // 5 seconds for connection

        // File IDs for MRTD (short file identifiers)
        val EF_COM: Byte = 0x1E
        val EF_SOD: Byte = 0x1D
        val DG1: Byte = 0x01
        val DG2: Byte = 0x02
    }

    private val bacAuthentication = BacAuthentication()

    /**
     * Reads card data with MRZ-based BAC authentication.
     *
     * @param tag NFC tag from Android
     * @param mrzData MRZ data for BAC authentication (document number, DOB, DOE)
     * @return NfcResult with CardData or error
     */
    suspend fun readCard(tag: Tag, mrzData: BacAuthentication.MrzData): NfcResult<CardData> {
        return withContext(Dispatchers.IO) {
            try {
                withTimeout(TIMEOUT_MS) {
                    readCardInternal(tag, mrzData)
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
     * Legacy method for backward compatibility.
     * Parses PIN as MRZ data in format: documentNumber|dateOfBirth|dateOfExpiry
     *
     * @param tag NFC tag from Android
     * @param pin MRZ data formatted as "docNo|YYMMDD|YYMMDD" or legacy 6-digit PIN
     * @return NfcResult with CardData or error
     */
    suspend fun readCard(tag: Tag, pin: String): NfcResult<CardData> {
        // Try to parse as MRZ data (format: docNo|dob|doe)
        val parts = pin.split("|")
        return if (parts.size == 3) {
            try {
                val mrzData = BacAuthentication.MrzData(
                    documentNumber = parts[0],
                    dateOfBirth = parts[1],
                    dateOfExpiry = parts[2]
                )
                readCard(tag, mrzData)
            } catch (e: IllegalArgumentException) {
                Timber.e(e, "Invalid MRZ data format")
                NfcResult.Error(NfcError.UnknownError("Invalid MRZ data: ${e.message}"))
            }
        } else {
            Timber.e("PIN format not supported. Use MRZ format: documentNumber|YYMMDD|YYMMDD")
            NfcResult.Error(NfcError.UnknownError(
                "Invalid format. Please enter MRZ data as: documentNumber|dateOfBirth|dateOfExpiry (e.g., A12345678|900101|301231)"
            ))
        }
    }

    /**
     * Internal implementation of card reading with BAC authentication.
     */
    private suspend fun readCardInternal(tag: Tag, mrzData: BacAuthentication.MrzData): NfcResult<CardData> {
        var isoDep: IsoDep? = null

        return try {
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

            // Select MRTD application
            val selectResult = selectMrtdApplication(isoDep)
            if (selectResult !is NfcResult.Success) {
                return selectResult as NfcResult.Error
            }

            // Perform BAC authentication
            val secureMessaging = performBacAuthentication(isoDep, mrzData)
            if (secureMessaging == null) {
                Timber.e("BAC authentication failed")
                return NfcResult.Error(NfcError.SecurityNotSatisfied)
            }

            Timber.d("BAC authentication successful, reading data with secure messaging...")

            // Read EF.SOD with secure messaging
            val sodData = readFileSecure(isoDep, secureMessaging, EF_SOD)

            // Read DG1 (personal data) with secure messaging
            val personalData = readDg1Secure(isoDep, secureMessaging)

            // Read DG2 (photo) with secure messaging
            val photo = readDg2Secure(isoDep, secureMessaging)

            // Construct CardData
            val cardData = CardData(
                personalData = personalData,
                photo = photo,
                cardAccessData = null, // EF.CardAccess not present for BAC-only cards
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
     * Selects the MRTD application by AID.
     */
    private fun selectMrtdApplication(isoDep: IsoDep): NfcResult<Unit> {
        return try {
            val command = ApduHelper.selectEidAid()
            ApduHelper.logCommand(command)

            val response = isoDep.transceive(command)
            ApduHelper.logResponse(response)

            val (_, statusWord) = ApduHelper.parseResponse(response)

            if (ApduHelper.isSuccess(statusWord)) {
                Timber.d("MRTD application selected successfully")
                NfcResult.Success(Unit)
            } else {
                Timber.e("Failed to select MRTD application: ${ApduHelper.getStatusDescription(statusWord)}")
                NfcResult.Error(NfcError.InvalidCard)
            }
        } catch (e: Exception) {
            Timber.e(e, "Error selecting MRTD application")
            NfcResult.Error(NfcError.ConnectionLost)
        }
    }

    /**
     * Performs BAC (Basic Access Control) authentication.
     *
     * @return SecureMessaging instance if successful, null otherwise
     */
    private suspend fun performBacAuthentication(
        isoDep: IsoDep,
        mrzData: BacAuthentication.MrzData
    ): SecureMessaging? {
        try {
            Timber.d("Starting BAC authentication...")

            // Derive keys from MRZ data
            val (kEnc, kMac) = bacAuthentication.deriveKeys(mrzData)

            // Get challenge from card (GET CHALLENGE command)
            val challengeCommand = byteArrayOf(
                0x00,              // CLA
                0x84.toByte(),    // INS: GET CHALLENGE
                0x00,              // P1
                0x00,              // P2
                0x08               // Le: 8 bytes
            )

            ApduHelper.logCommand(challengeCommand)
            val challengeResponse = isoDep.transceive(challengeCommand)
            ApduHelper.logResponse(challengeResponse)

            val (rndIcc, challengeStatus) = ApduHelper.parseResponse(challengeResponse)

            if (!ApduHelper.isSuccess(challengeStatus) || rndIcc.size != 8) {
                Timber.e("Failed to get challenge: ${ApduHelper.getStatusDescription(challengeStatus)}")
                return null
            }

            // Perform mutual authentication
            val sessionKeys = bacAuthentication.performMutualAuthentication(
                kEnc, kMac, rndIcc
            ) { command ->
                isoDep.transceive(command)
            }

            if (sessionKeys == null) {
                Timber.e("Mutual authentication failed")
                return null
            }

            Timber.d("BAC authentication completed successfully")
            return SecureMessaging(
                sessionKeys.encryptionKey,
                sessionKeys.macKey,
                sessionKeys.sendSequenceCounter
            )

        } catch (e: Exception) {
            Timber.e(e, "BAC authentication error")
            return null
        }
    }

    /**
     * Reads a file using secure messaging.
     */
    private suspend fun readFileSecure(
        isoDep: IsoDep,
        secureMessaging: SecureMessaging,
        shortFileId: Byte
    ): ByteArray? {
        try {
            Timber.d("Reading file with SFI: 0x${String.format("%02X", shortFileId)}")

            // Select file by short file identifier using READ BINARY with SFI
            // P1 = 0x80 | SFI, P2 = offset
            val allData = mutableListOf<Byte>()
            var offset = 0

            while (true) {
                // Build READ BINARY command with short file identifier
                val p1 = if (offset == 0) (0x80 or shortFileId.toInt()).toByte() else (offset shr 8).toByte()
                val p2 = (offset and 0xFF).toByte()

                val readCommand = if (offset == 0) {
                    // First read: use SFI
                    byteArrayOf(0x00, 0xB0.toByte(), p1, p2, 0x00)
                } else {
                    // Subsequent reads: use offset
                    byteArrayOf(0x00, 0xB0.toByte(), (offset shr 8).toByte(), (offset and 0xFF).toByte(), 0x00)
                }

                // Wrap with secure messaging
                val protectedCommand = secureMessaging.wrapCommand(readCommand)
                ApduHelper.logCommand(protectedCommand)

                val response = isoDep.transceive(protectedCommand)
                ApduHelper.logResponse(response)

                // Unwrap response
                val unwrapped = secureMessaging.unwrapResponse(response)
                if (unwrapped == null) {
                    Timber.e("Failed to unwrap response")
                    if (offset == 0) return null
                    break
                }

                val (data, statusWord) = unwrapped

                if (statusWord == 0x6982) {
                    Timber.e("Security condition not satisfied")
                    return null
                }

                if (statusWord == 0x6A82 || statusWord == 0x6A83) {
                    // File not found or end of file
                    if (offset == 0) {
                        Timber.w("File not found")
                        return null
                    }
                    break
                }

                if (!ApduHelper.isSuccess(statusWord) && statusWord != 0x6282) {
                    if (offset == 0) {
                        Timber.e("Failed to read file: ${ApduHelper.getStatusDescription(statusWord)}")
                        return null
                    }
                    break
                }

                if (data.isEmpty()) {
                    break
                }

                allData.addAll(data.toList())
                offset += data.size

                // Check if we reached end of file
                if (data.size < 200 || statusWord == 0x6282) {
                    break
                }
            }

            Timber.d("Read ${allData.size} bytes from file")
            return if (allData.isNotEmpty()) allData.toByteArray() else null

        } catch (e: Exception) {
            Timber.e(e, "Error reading file with secure messaging")
            return null
        }
    }

    /**
     * Reads DG1 (personal data) with secure messaging and parses it.
     */
    private suspend fun readDg1Secure(isoDep: IsoDep, secureMessaging: SecureMessaging): PersonalData? {
        return try {
            Timber.d("Reading DG1 (personal data) with secure messaging...")
            val dg1Data = readFileSecure(isoDep, secureMessaging, DG1)

            if (dg1Data != null) {
                Timber.d("DG1 data: ${ApduHelper.toHexString(dg1Data.take(50).toByteArray())}...")
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
     * Reads DG2 (photo) with secure messaging and decodes it.
     */
    private suspend fun readDg2Secure(isoDep: IsoDep, secureMessaging: SecureMessaging): android.graphics.Bitmap? {
        return try {
            Timber.d("Reading DG2 (photo) with secure messaging...")
            val dg2Data = readFileSecure(isoDep, secureMessaging, DG2)

            if (dg2Data != null) {
                Timber.d("DG2 data size: ${dg2Data.size} bytes")
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
     * Validates SOD signature and logs results.
     */
    private fun validateSod(sodData: ByteArray) {
        try {
            Timber.d("Validating SOD...")
            val result = SodValidator.validate(sodData)

            if (result.isSignatureValid) {
                Timber.d("SOD signature is valid")
            } else {
                Timber.w("SOD signature is invalid")
            }

            if (result.isCertificateValid) {
                Timber.d("Document Signer certificate is valid")
            } else {
                Timber.w("Document Signer certificate is invalid or expired")
            }

            if (result.errorMessage != null) {
                Timber.e("SOD validation error: ${result.errorMessage}")
            }
        } catch (e: Exception) {
            Timber.e(e, "Error validating SOD")
        }
    }
}
