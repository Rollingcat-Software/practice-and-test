package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.graphics.Bitmap
import android.nfc.Tag
import android.nfc.tech.IsoDep
import android.util.Log
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.BacAuthentication
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.Dg1Parser
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.Dg2Parser
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.EidApduHelper
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.SecureMessaging
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import com.rollingcatsoftware.universalnfcreader.domain.model.TurkishEidData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.IOException

/**
 * Reader for Turkish eID cards (ICAO MRTD compliant).
 *
 * This reader handles:
 * 1. IsoDep connection to the card
 * 2. MRTD Application selection
 * 3. BAC (Basic Access Control) authentication using MRZ data
 * 4. Secure messaging for encrypted communication
 * 5. Reading data groups (DG1, DG2)
 * 6. Parsing personal data and photo
 *
 * Security Note: BAC authentication requires MRZ data (document number, DOB, DOE)
 * from the physical card. Session keys are cleared after reading.
 */
class TurkishEidReader : BaseCardReader() {

    companion object {
        private const val TAG = "TurkishEidReader"
        private const val TIMEOUT_MS = 30000L // 30 seconds
        private const val CONNECTION_TIMEOUT = 5000 // 5 seconds
    }

    private val bacAuthentication = BacAuthentication()

    override val supportedCardTypes = listOf(CardType.TURKISH_EID)

    override fun requiresAuthentication(): Boolean = true

    /**
     * Basic card read without authentication - returns minimal data.
     */
    override suspend fun readCard(tag: Tag): Result<CardData> {
        val basicInfo = readBasicInfo(tag)

        return Result.success(
            TurkishEidData(
                uid = basicInfo.uid.joinToString("") { "%02X".format(it) },
                cardType = CardType.TURKISH_EID,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = emptyMap(),
                bacSuccessful = false
            )
        )
    }

    /**
     * Read card with MRZ-based BAC authentication.
     */
    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> = withContext(Dispatchers.IO) {
        if (authData !is AuthenticationData.MrzData) {
            return@withContext Result.error(
                CardError.AuthenticationRequired(
                    message = "Turkish eID requires MRZ authentication data"
                )
            )
        }

        try {
            withTimeout(TIMEOUT_MS) {
                readCardInternal(tag, authData)
            }
        } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
            Log.e(TAG, "Card reading timed out")
            Result.error(CardError.Timeout())
        } catch (e: IOException) {
            Log.e(TAG, "IO error during card reading", e)
            Result.error(CardError.ConnectionLost())
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during card reading", e)
            Result.error(CardError.Unknown(message = e.message ?: "Unknown error"))
        }
    }

    /**
     * Internal implementation of card reading with BAC authentication.
     */
    private suspend fun readCardInternal(
        tag: Tag,
        authData: AuthenticationData.MrzData
    ): Result<CardData> {
        var isoDep: IsoDep? = null
        var secureMessaging: SecureMessaging? = null

        return try {
            val basicInfo = readBasicInfo(tag)

            // Get IsoDep interface
            isoDep = IsoDep.get(tag)
            if (isoDep == null) {
                Log.e(TAG, "IsoDep not supported by this tag")
                return Result.error(
                    CardError.UnsupportedCard(
                        detectedTechnologies = basicInfo.technologies
                    )
                )
            }

            // Connect to the card
            Log.d(TAG, "Connecting to card...")
            isoDep.timeout = CONNECTION_TIMEOUT
            isoDep.connect()
            Log.d(TAG, "Connected to card")

            // Select MRTD application
            val selectResult = selectMrtdApplication(isoDep)
            if (selectResult != null) {
                return selectResult
            }

            // Create MRZ data for BAC authentication
            val mrzData = try {
                BacAuthentication.MrzData(
                    documentNumber = authData.getDocumentNumber(),
                    dateOfBirth = authData.getDateOfBirth(),
                    dateOfExpiry = authData.getDateOfExpiry()
                )
            } catch (e: IllegalArgumentException) {
                Log.e(TAG, "Invalid MRZ data: ${e.message}")
                return Result.error(
                    CardError.AuthenticationFailed(
                        message = "Invalid MRZ data: ${e.message}"
                    )
                )
            }

            // Perform BAC authentication
            secureMessaging = performBacAuthentication(isoDep, mrzData)
            if (secureMessaging == null) {
                Log.e(TAG, "BAC authentication failed")
                return Result.error(
                    CardError.AuthenticationFailed(
                        message = "BAC authentication failed. Check MRZ data."
                    )
                )
            }

            Log.d(TAG, "BAC authentication successful, reading data with secure messaging...")

            // Read DG1 (personal data) with secure messaging
            val personalData = readDg1Secure(isoDep, secureMessaging)

            // Read DG2 (photo) with secure messaging
            val photo = readDg2Secure(isoDep, secureMessaging)

            // Construct TurkishEidData
            val cardData = TurkishEidData(
                uid = basicInfo.uid.joinToString("") { "%02X".format(it) },
                cardType = CardType.TURKISH_EID,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildRawDataMap(personalData),
                documentNumber = personalData?.documentNumber ?: "",
                surname = personalData?.lastName ?: "",
                givenNames = personalData?.firstName ?: "",
                nationality = personalData?.nationality ?: "",
                dateOfBirth = personalData?.birthDate ?: "",
                sex = personalData?.gender ?: "",
                dateOfExpiry = personalData?.expiryDate ?: "",
                personalNumber = personalData?.tckn ?: "",
                photo = photo,
                bacSuccessful = true
            )

            Log.d(TAG, "Card reading completed successfully")
            Result.success(cardData)

        } catch (e: IOException) {
            Log.e(TAG, "Connection lost during card reading", e)
            Result.error(CardError.ConnectionLost())
        } catch (e: Exception) {
            Log.e(TAG, "Error during card reading", e)
            Result.error(CardError.Unknown(message = e.message ?: "Unknown error"))
        } finally {
            // Clear sensitive key material
            secureMessaging?.clear()

            // Always close the connection
            try {
                isoDep?.close()
                Log.d(TAG, "IsoDep connection closed")
            } catch (e: Exception) {
                Log.e(TAG, "Error closing IsoDep connection", e)
            }
        }
    }

    /**
     * Selects the MRTD application by AID.
     */
    private fun selectMrtdApplication(isoDep: IsoDep): Result<CardData>? {
        return try {
            val command = EidApduHelper.selectMrtdAid()
            EidApduHelper.logCommand(command)

            val response = isoDep.transceive(command)
            EidApduHelper.logResponse(response)

            val (_, statusWord) = EidApduHelper.parseResponse(response)

            if (EidApduHelper.isSuccess(statusWord)) {
                Log.d(TAG, "MRTD application selected successfully")
                null // Success - continue
            } else {
                Log.e(
                    TAG,
                    "Failed to select MRTD application: ${
                        EidApduHelper.getStatusDescription(statusWord)
                    }"
                )
                Result.error(
                    CardError.UnsupportedCard(
                        message = "Not a valid Turkish eID card"
                    )
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error selecting MRTD application", e)
            Result.error(CardError.ConnectionLost())
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
            Log.d(TAG, "Starting BAC authentication...")

            // Derive keys from MRZ data
            val (kEnc, kMac) = bacAuthentication.deriveKeys(mrzData)

            // Get challenge from card
            val challengeCommand = EidApduHelper.getChallengeCommand()
            EidApduHelper.logCommand(challengeCommand)

            val challengeResponse = isoDep.transceive(challengeCommand)
            EidApduHelper.logResponse(challengeResponse)

            val (rndIcc, challengeStatus) = EidApduHelper.parseResponse(challengeResponse)

            if (!EidApduHelper.isSuccess(challengeStatus) || rndIcc.size != 8) {
                Log.e(
                    TAG,
                    "Failed to get challenge: ${EidApduHelper.getStatusDescription(challengeStatus)}"
                )
                return null
            }

            // Perform mutual authentication
            val sessionKeys = bacAuthentication.performMutualAuthentication(
                kEnc, kMac, rndIcc
            ) { command ->
                isoDep.transceive(command)
            }

            // Clear base keys
            kEnc.fill(0)
            kMac.fill(0)

            if (sessionKeys == null) {
                Log.e(TAG, "Mutual authentication failed")
                return null
            }

            Log.d(TAG, "BAC authentication completed successfully")
            return SecureMessaging(
                sessionKeys.encryptionKey,
                sessionKeys.macKey,
                sessionKeys.sendSequenceCounter
            )

        } catch (e: Exception) {
            Log.e(TAG, "BAC authentication error", e)
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
            Log.d(TAG, "Reading file with SFI: 0x${String.format("%02X", shortFileId)}")

            val allData = mutableListOf<Byte>()
            var offset = 0

            while (true) {
                // Build READ BINARY command
                val readCommand = EidApduHelper.readBinaryCommand(
                    offset = offset,
                    length = 0,
                    useSfi = offset == 0,
                    sfi = shortFileId
                )

                // Wrap with secure messaging
                val protectedCommand = secureMessaging.wrapCommand(readCommand)
                EidApduHelper.logCommand(protectedCommand)

                val response = isoDep.transceive(protectedCommand)
                EidApduHelper.logResponse(response)

                // Unwrap response
                val unwrapped = secureMessaging.unwrapResponse(response)
                if (unwrapped == null) {
                    Log.e(TAG, "Failed to unwrap response")
                    if (offset == 0) return null
                    break
                }

                val (data, statusWord) = unwrapped

                if (statusWord == 0x6982) {
                    Log.e(TAG, "Security condition not satisfied")
                    return null
                }

                if (statusWord == 0x6A82 || statusWord == 0x6A83) {
                    if (offset == 0) {
                        Log.w(TAG, "File not found")
                        return null
                    }
                    break
                }

                if (!EidApduHelper.isSuccess(statusWord) && statusWord != 0x6282) {
                    if (offset == 0) {
                        Log.e(
                            TAG,
                            "Failed to read file: ${EidApduHelper.getStatusDescription(statusWord)}"
                        )
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

            Log.d(TAG, "Read ${allData.size} bytes from file")
            return if (allData.isNotEmpty()) allData.toByteArray() else null

        } catch (e: Exception) {
            Log.e(TAG, "Error reading file with secure messaging", e)
            return null
        }
    }

    /**
     * Reads DG1 (personal data) with secure messaging and parses it.
     */
    private suspend fun readDg1Secure(
        isoDep: IsoDep,
        secureMessaging: SecureMessaging
    ): Dg1Parser.PersonalData? {
        return try {
            Log.d(TAG, "Reading DG1 (personal data) with secure messaging...")
            val dg1Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG1)

            if (dg1Data != null) {
                Log.d(TAG, "DG1 data size: ${dg1Data.size} bytes")
                Dg1Parser.parse(dg1Data)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read DG1", e)
            null
        }
    }

    /**
     * Reads DG2 (photo) with secure messaging and decodes it.
     */
    private suspend fun readDg2Secure(
        isoDep: IsoDep,
        secureMessaging: SecureMessaging
    ): Bitmap? {
        return try {
            Log.d(TAG, "Reading DG2 (photo) with secure messaging...")
            val dg2Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG2)

            if (dg2Data != null) {
                Log.d(TAG, "DG2 data size: ${dg2Data.size} bytes")
                Dg2Parser.parse(dg2Data)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read DG2", e)
            null
        }
    }

    /**
     * Build raw data map for debugging.
     */
    private fun buildRawDataMap(personalData: Dg1Parser.PersonalData?): Map<String, Any> {
        return buildMap {
            personalData?.let {
                put("parsed", true)
                put("tckn", it.tckn.take(3) + "***" + it.tckn.takeLast(2)) // Masked
            } ?: put("parsed", false)
        }
    }
}
