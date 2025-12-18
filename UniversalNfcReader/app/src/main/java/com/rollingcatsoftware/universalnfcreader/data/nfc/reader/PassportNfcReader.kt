package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.graphics.Bitmap
import android.nfc.Tag
import android.nfc.tech.IsoDep
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.BacAuthentication
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.Dg1Parser
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.Dg2Parser
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.EidApduHelper
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.MrzParser
import com.rollingcatsoftware.universalnfcreader.data.nfc.eid.SecureMessaging
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod.HashVerifier
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod.LdsSecurityObjectParser
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod.SodValidator
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.PassportData
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.IOException

/**
 * Reader for e-Passports (ICAO Doc 9303 TD3 format).
 *
 * This reader handles:
 * 1. IsoDep connection to the passport chip
 * 2. MRTD Application selection
 * 3. BAC (Basic Access Control) authentication using TD3 MRZ data
 * 4. Secure messaging for encrypted communication
 * 5. Reading data groups (DG1, DG2, SOD)
 * 6. SOD validation and hash verification
 *
 * COMPLIANCE: se-checklist.md
 * - Section 5.2: "Log errors securely (no PII in logs)" - Uses SecureLogger
 * - Section 3.1: "Implement timeout handling" - Uses withTimeout()
 * - Section 1.1: "Clear PIN bytes from memory" - Clears session keys in finally
 *
 * Security Note: BAC authentication requires MRZ data (document number, DOB, DOE)
 * from the physical passport. Session keys are cleared after reading.
 *
 * TD3 MRZ Format (2 lines × 44 characters):
 * Line 1: Type(2) + Country(3) + Surname<<GivenNames (39)
 * Line 2: DocNo(9) + Check(1) + Nationality(3) + DOB(6) + Check(1) + Sex(1) + DOE(6) + Check(1) + Personal(14) + Check(1) + Composite(1)
 */
class PassportNfcReader : BaseCardReader() {

    companion object {
        private const val TAG = "PassportNfcReader"
        private const val TIMEOUT_MS = 45000L // 45 seconds (passports may have larger photos)
        private const val CONNECTION_TIMEOUT = 5000 // 5 seconds
    }

    private val bacAuthentication = BacAuthentication()
    private val sodValidator = SodValidator()

    override val supportedCardTypes = listOf(CardType.PASSPORT)

    override fun requiresAuthentication(): Boolean = true

    /**
     * Basic card read without authentication - returns minimal data.
     */
    override suspend fun readCard(tag: Tag): Result<CardData> {
        val basicInfo = readBasicInfo(tag)

        return Result.success(
            PassportData(
                uid = basicInfo.uid.joinToString("") { "%02X".format(it) },
                cardType = CardType.PASSPORT,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = emptyMap(),
                bacSuccessful = false
            )
        )
    }

    /**
     * Read passport with MRZ-based BAC authentication.
     */
    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> = withContext(Dispatchers.IO) {
        if (authData !is AuthenticationData.MrzData) {
            return@withContext Result.error(
                CardError.AuthenticationRequired(
                    message = "Passport requires MRZ authentication data"
                )
            )
        }

        try {
            withTimeout(TIMEOUT_MS) {
                readPassportInternal(tag, authData)
            }
        } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
            SecureLogger.e(TAG, "Passport reading timed out")
            Result.error(CardError.Timeout())
        } catch (e: IOException) {
            SecureLogger.e(TAG, "IO error during passport reading", e)
            Result.error(CardError.ConnectionLost())
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Unexpected error during passport reading", e)
            Result.error(CardError.Unknown(message = e.message ?: "Unknown error"))
        }
    }

    /**
     * Internal implementation of passport reading with BAC authentication.
     */
    private suspend fun readPassportInternal(
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
                SecureLogger.e(TAG, "IsoDep not supported by this tag")
                return Result.error(
                    CardError.UnsupportedCard(
                        detectedTechnologies = basicInfo.technologies
                    )
                )
            }

            // Connect to the passport chip
            SecureLogger.d(TAG, "Connecting to passport chip...")
            isoDep.timeout = CONNECTION_TIMEOUT
            isoDep.connect()
            SecureLogger.d(TAG, "Connected to passport")

            // Select MRTD application
            val selectResult = selectMrtdApplication(isoDep)
            if (selectResult != null) {
                return selectResult
            }

            // Parse MRZ data for BAC authentication
            val mrzData = try {
                BacAuthentication.MrzData(
                    documentNumber = authData.getDocumentNumber(),
                    dateOfBirth = authData.getDateOfBirth(),
                    dateOfExpiry = authData.getDateOfExpiry()
                )
            } catch (e: IllegalArgumentException) {
                SecureLogger.e(TAG, "Invalid MRZ data: ${e.message}")
                return Result.error(
                    CardError.AuthenticationFailed(
                        message = "Invalid MRZ data: ${e.message}"
                    )
                )
            }

            // Perform BAC authentication
            secureMessaging = performBacAuthentication(isoDep, mrzData)
            if (secureMessaging == null) {
                SecureLogger.e(TAG, "BAC authentication failed")
                return Result.error(
                    CardError.AuthenticationFailed(
                        message = "BAC authentication failed. Please verify MRZ data is correct."
                    )
                )
            }

            SecureLogger.d(TAG, "BAC authentication successful, reading passport data...")

            // Read EF.COM to get available data groups
            val efComData = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.EF_COM)
            val availableDataGroups = parseEfCom(efComData)
            SecureLogger.d(TAG, "Available data groups: $availableDataGroups")

            // Read SOD (Security Object Document) for validation
            val sodData = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.EF_SOD)
            var sodValidationResult: SodValidator.SodValidationResult? = null
            if (sodData != null) {
                SecureLogger.d(TAG, "SOD data size: ${sodData.size} bytes")
                sodValidationResult = sodValidator.validate(sodData)
                SecureLogger.d(TAG, "SOD validation result: signature=${sodValidationResult?.isSignatureValid}")
            }

            // Read DG1 (MRZ data)
            val dg1Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG1)

            // Parse MRZ - try MrzParser first (handles TD3/TD1), fall back to Dg1Parser
            var mrzParserResult: MrzParser.MrzData? = null
            var dg1ParserResult: Dg1Parser.PersonalData? = null

            if (dg1Data != null) {
                // Extract raw MRZ string from DG1 TLV structure
                val rawMrz = extractMrzFromDg1(dg1Data)
                if (rawMrz != null) {
                    SecureLogger.d(TAG, "Extracted MRZ (${rawMrz.length} chars)")
                    mrzParserResult = MrzParser.parse(rawMrz)
                    SecureLogger.d(TAG, "MrzParser result: docType=${mrzParserResult?.documentType}, name=${mrzParserResult?.surname}")
                }

                // Also try Dg1Parser for TD1 cards (like Turkish eID)
                if (mrzParserResult == null) {
                    dg1ParserResult = Dg1Parser.parse(dg1Data)
                    SecureLogger.d(TAG, "Dg1Parser result: name=${dg1ParserResult?.lastName}")
                }
            }

            // Verify DG1 hash
            var dg1HashValid: Boolean? = null
            if (dg1Data != null && sodValidationResult?.ldsSecurityObject != null) {
                val dg1Result = HashVerifier.verifyDataGroup(
                    1, dg1Data, sodValidationResult.ldsSecurityObject
                )
                dg1HashValid = dg1Result.isValid
                SecureLogger.d(TAG, "DG1 hash verification: $dg1HashValid")
            }

            // Read DG2 (Photo)
            val photo = readDg2Secure(isoDep, secureMessaging)

            // Verify DG2 hash
            var dg2HashValid: Boolean? = null
            val dg2Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG2)
            if (dg2Data != null && sodValidationResult?.ldsSecurityObject != null) {
                val dg2Result = HashVerifier.verifyDataGroup(
                    2, dg2Data, sodValidationResult.ldsSecurityObject
                )
                dg2HashValid = dg2Result.isValid
                SecureLogger.d(TAG, "DG2 hash verification: $dg2HashValid")
            }

            // Optionally read DG11 (Additional Personal Data) if available
            val dg11Data = if (availableDataGroups.contains(11)) {
                readDg11Secure(isoDep, secureMessaging)
            } else null

            // Optionally read DG12 (Document Details) if available
            val dg12Data = if (availableDataGroups.contains(12)) {
                readDg12Secure(isoDep, secureMessaging)
            } else null

            // Construct PassportData - prefer MrzParser (TD3) over Dg1Parser (TD1)
            val passportData = if (mrzParserResult != null) {
                // TD3 passport format from MrzParser
                PassportData(
                    uid = basicInfo.uid.joinToString("") { "%02X".format(it) },
                    cardType = CardType.PASSPORT,
                    readTimestamp = System.currentTimeMillis(),
                    technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                    rawData = buildRawDataMapFromMrzParser(mrzParserResult, availableDataGroups),
                    documentType = mrzParserResult.documentCode,
                    issuingCountry = mrzParserResult.issuingCountry,
                    documentNumber = mrzParserResult.documentNumber,
                    surname = mrzParserResult.surname,
                    givenNames = mrzParserResult.givenNames,
                    nationality = mrzParserResult.nationality,
                    dateOfBirth = mrzParserResult.getFormattedDateOfBirth(),
                    sex = mrzParserResult.sex,
                    dateOfExpiry = mrzParserResult.getFormattedDateOfExpiry(),
                    personalNumber = mrzParserResult.personalNumber,
                    photo = photo,
                    dg11Data = dg11Data,
                    dg12Data = dg12Data,
                    bacSuccessful = true,
                    paceSuccessful = false,
                    sodValid = sodValidationResult?.isSignatureValid,
                    dg1HashValid = dg1HashValid,
                    dg2HashValid = dg2HashValid,
                    activeAuthenticationSupported = availableDataGroups.contains(15),
                    chipAuthenticationSupported = availableDataGroups.contains(14)
                )
            } else {
                // TD1 format from Dg1Parser (fallback)
                PassportData(
                    uid = basicInfo.uid.joinToString("") { "%02X".format(it) },
                    cardType = CardType.PASSPORT,
                    readTimestamp = System.currentTimeMillis(),
                    technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                    rawData = buildRawDataMap(dg1ParserResult, availableDataGroups),
                    documentType = "P",
                    issuingCountry = dg1ParserResult?.nationality ?: "",
                    documentNumber = dg1ParserResult?.documentNumber ?: "",
                    surname = dg1ParserResult?.lastName ?: "",
                    givenNames = dg1ParserResult?.firstName ?: "",
                    nationality = dg1ParserResult?.nationality ?: "",
                    dateOfBirth = dg1ParserResult?.birthDate ?: "",
                    sex = dg1ParserResult?.gender ?: "",
                    dateOfExpiry = dg1ParserResult?.expiryDate ?: "",
                    personalNumber = dg1ParserResult?.tckn ?: "",
                    photo = photo,
                    dg11Data = dg11Data,
                    dg12Data = dg12Data,
                    bacSuccessful = true,
                    paceSuccessful = false,
                    sodValid = sodValidationResult?.isSignatureValid,
                    dg1HashValid = dg1HashValid,
                    dg2HashValid = dg2HashValid,
                    activeAuthenticationSupported = availableDataGroups.contains(15),
                    chipAuthenticationSupported = availableDataGroups.contains(14)
                )
            }

            SecureLogger.d(TAG, "Passport reading completed successfully")
            Result.success(passportData)

        } catch (e: IOException) {
            SecureLogger.e(TAG, "Connection lost during passport reading", e)
            Result.error(CardError.ConnectionLost())
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error during passport reading", e)
            Result.error(CardError.Unknown(message = e.message ?: "Unknown error"))
        } finally {
            // Clear sensitive key material
            secureMessaging?.clear()

            // Always close the connection
            try {
                isoDep?.close()
                SecureLogger.d(TAG, "IsoDep connection closed")
            } catch (e: Exception) {
                SecureLogger.e(TAG, "Error closing IsoDep connection", e)
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
                SecureLogger.d(TAG, "MRTD application selected successfully")
                null // Success - continue
            } else {
                SecureLogger.e(
                    TAG,
                    "Failed to select MRTD application: ${
                        EidApduHelper.getStatusDescription(statusWord)
                    }"
                )
                Result.error(
                    CardError.UnsupportedCard(
                        message = "Not a valid e-Passport or MRTD document"
                    )
                )
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error selecting MRTD application", e)
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
            SecureLogger.d(TAG, "Starting BAC authentication...")

            // Derive keys from MRZ data
            val (kEnc, kMac) = bacAuthentication.deriveKeys(mrzData)

            // Get challenge from card
            val challengeCommand = EidApduHelper.getChallengeCommand()
            EidApduHelper.logCommand(challengeCommand)

            val challengeResponse = isoDep.transceive(challengeCommand)
            EidApduHelper.logResponse(challengeResponse)

            val (rndIcc, challengeStatus) = EidApduHelper.parseResponse(challengeResponse)

            if (!EidApduHelper.isSuccess(challengeStatus) || rndIcc.size != 8) {
                SecureLogger.e(
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
                SecureLogger.e(TAG, "Mutual authentication failed")
                return null
            }

            SecureLogger.d(TAG, "BAC authentication completed successfully")
            return SecureMessaging(
                sessionKeys.encryptionKey,
                sessionKeys.macKey,
                sessionKeys.sendSequenceCounter
            )

        } catch (e: Exception) {
            SecureLogger.e(TAG, "BAC authentication error", e)
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
            SecureLogger.d(TAG, "Reading file with SFI: 0x${String.format("%02X", shortFileId)}")

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
                    SecureLogger.e(TAG, "Failed to unwrap response")
                    if (offset == 0) return null
                    break
                }

                val (data, statusWord) = unwrapped

                if (statusWord == 0x6982) {
                    SecureLogger.e(TAG, "Security condition not satisfied")
                    return null
                }

                if (statusWord == 0x6A82 || statusWord == 0x6A83) {
                    if (offset == 0) {
                        SecureLogger.w(TAG, "File not found")
                        return null
                    }
                    break
                }

                if (!EidApduHelper.isSuccess(statusWord) && statusWord != 0x6282) {
                    if (offset == 0) {
                        SecureLogger.e(
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

            SecureLogger.d(TAG, "Read ${allData.size} bytes from file")
            return if (allData.isNotEmpty()) allData.toByteArray() else null

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading file with secure messaging", e)
            return null
        }
    }

    /**
     * Parses EF.COM to get list of available data groups.
     *
     * @return List of available data group numbers (1, 2, 11, 12, etc.)
     */
    private fun parseEfCom(data: ByteArray?): List<Int> {
        if (data == null || data.isEmpty()) {
            return emptyList()
        }

        val dataGroups = mutableListOf<Int>()

        try {
            // EF.COM structure: 60 LL [5C LL tag-list] [5F01 LL LDS-version] [5F36 LL unicode-version]
            // Tag list contains DG tags: 61=DG1, 75=DG2, etc.
            var i = 0
            while (i < data.size) {
                val tag = data[i].toInt() and 0xFF
                i++
                if (i >= data.size) break

                val length = data[i].toInt() and 0xFF
                i++

                when (tag) {
                    0x60 -> continue // Application template, continue parsing
                    0x5C -> {
                        // Tag list
                        val endPos = i + length
                        while (i < endPos && i < data.size) {
                            val dgTag = data[i].toInt() and 0xFF
                            val dgNumber = when (dgTag) {
                                0x61 -> 1   // DG1
                                0x75 -> 2   // DG2
                                0x63 -> 3   // DG3
                                0x76 -> 4   // DG4
                                0x65 -> 5   // DG5
                                0x66 -> 6   // DG6
                                0x67 -> 7   // DG7
                                0x68 -> 8   // DG8
                                0x69 -> 9   // DG9
                                0x6A -> 10  // DG10
                                0x6B -> 11  // DG11
                                0x6C -> 12  // DG12
                                0x6D -> 13  // DG13
                                0x6E -> 14  // DG14 (Chip Authentication)
                                0x6F -> 15  // DG15 (Active Authentication)
                                0x70 -> 16  // DG16
                                else -> null
                            }
                            dgNumber?.let { dataGroups.add(it) }
                            i++
                        }
                    }
                    else -> {
                        i += length // Skip other tags
                    }
                }
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse EF.COM", e)
        }

        return dataGroups
    }

    /**
     * Reads DG2 (photo) with secure messaging and decodes it.
     */
    private suspend fun readDg2Secure(
        isoDep: IsoDep,
        secureMessaging: SecureMessaging
    ): Bitmap? {
        return try {
            SecureLogger.d(TAG, "Reading DG2 (photo) with secure messaging...")
            val dg2Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG2)

            if (dg2Data != null) {
                SecureLogger.d(TAG, "DG2 data size: ${dg2Data.size} bytes")
                Dg2Parser.parse(dg2Data)
            } else {
                null
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to read DG2", e)
            null
        }
    }

    /**
     * Reads DG11 (Additional Personal Details) with secure messaging.
     *
     * DG11 may contain:
     * - Full name
     * - Personal number (national ID)
     * - Place of birth
     * - Address
     * - Telephone
     * - Profession
     * - Title
     * - Personal summary
     */
    private suspend fun readDg11Secure(
        isoDep: IsoDep,
        secureMessaging: SecureMessaging
    ): Map<String, String>? {
        return try {
            SecureLogger.d(TAG, "Reading DG11 (additional personal data) with secure messaging...")
            val dg11Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG11)

            if (dg11Data != null) {
                SecureLogger.d(TAG, "DG11 data size: ${dg11Data.size} bytes")
                parseDg11(dg11Data)
            } else {
                null
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to read DG11", e)
            null
        }
    }

    /**
     * Parses DG11 TLV structure.
     */
    private fun parseDg11(data: ByteArray): Map<String, String>? {
        val result = mutableMapOf<String, String>()

        try {
            // DG11 is TLV encoded with tag 6B
            var i = 0
            while (i < data.size - 1) {
                val tag = data[i].toInt() and 0xFF
                i++

                // Handle two-byte tags
                val fullTag = if (tag == 0x5F) {
                    val tag2 = data[i].toInt() and 0xFF
                    i++
                    (tag shl 8) or tag2
                } else {
                    tag
                }

                if (i >= data.size) break
                val length = data[i].toInt() and 0xFF
                i++

                if (i + length > data.size) break
                val value = String(data.sliceArray(i until i + length), Charsets.UTF_8)
                i += length

                when (fullTag) {
                    0x5F0E -> result["fullNameOfHolder"] = value
                    0x5F11 -> result["personalNumber"] = SecureLogger.maskDocumentNumber(value) // Redacted
                    0x5F42 -> result["placeOfBirth"] = value
                    0x5F12 -> result["dateOfBirth"] = value
                    0x5F13 -> result["permanentAddress"] = value
                    0x5F14 -> result["telephone"] = value
                    0x5F15 -> result["profession"] = value
                    0x5F16 -> result["title"] = value
                    0x5F17 -> result["personalSummary"] = value
                    0x6B -> continue // DG11 tag itself
                    else -> {
                        // Unknown tag, skip
                    }
                }
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse DG11", e)
            return null
        }

        return if (result.isNotEmpty()) result else null
    }

    /**
     * Reads DG12 (Document Details) with secure messaging.
     *
     * DG12 may contain:
     * - Issuing authority
     * - Date of issue
     * - Other persons
     * - Endorsements
     * - Tax/exit requirements
     */
    private suspend fun readDg12Secure(
        isoDep: IsoDep,
        secureMessaging: SecureMessaging
    ): Map<String, String>? {
        return try {
            SecureLogger.d(TAG, "Reading DG12 (document details) with secure messaging...")
            val dg12Data = readFileSecure(isoDep, secureMessaging, EidApduHelper.FileIds.DG12)

            if (dg12Data != null) {
                SecureLogger.d(TAG, "DG12 data size: ${dg12Data.size} bytes")
                parseDg12(dg12Data)
            } else {
                null
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to read DG12", e)
            null
        }
    }

    /**
     * Parses DG12 TLV structure.
     */
    private fun parseDg12(data: ByteArray): Map<String, String>? {
        val result = mutableMapOf<String, String>()

        try {
            var i = 0
            while (i < data.size - 1) {
                val tag = data[i].toInt() and 0xFF
                i++

                // Handle two-byte tags
                val fullTag = if (tag == 0x5F) {
                    val tag2 = data[i].toInt() and 0xFF
                    i++
                    (tag shl 8) or tag2
                } else {
                    tag
                }

                if (i >= data.size) break
                val length = data[i].toInt() and 0xFF
                i++

                if (i + length > data.size) break
                val value = String(data.sliceArray(i until i + length), Charsets.UTF_8)
                i += length

                when (fullTag) {
                    0x5F19 -> result["issuingAuthority"] = value
                    0x5F26 -> result["dateOfIssue"] = value
                    0x5F1A -> result["otherPersons"] = value
                    0x5F1B -> result["endorsements"] = value
                    0x5F1C -> result["taxExitRequirements"] = value
                    0x5F55 -> result["dateOfPersonalization"] = value
                    0x5F56 -> result["personalizationSystemSerialNumber"] = value
                    0x6C -> continue // DG12 tag itself
                    else -> {
                        // Unknown tag, skip
                    }
                }
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse DG12", e)
            return null
        }

        return if (result.isNotEmpty()) result else null
    }

    /**
     * Build raw data map for debugging (without sensitive data).
     */
    private fun buildRawDataMap(
        personalData: Dg1Parser.PersonalData?,
        availableDataGroups: List<Int>
    ): Map<String, Any> {
        return buildMap {
            personalData?.let {
                put("parsed", true)
                put("mrzFormat", "TD1")
            } ?: put("parsed", false)
            put("availableDataGroups", availableDataGroups)
        }
    }

    /**
     * Build raw data map from MrzParser result.
     */
    private fun buildRawDataMapFromMrzParser(
        mrzData: MrzParser.MrzData,
        availableDataGroups: List<Int>
    ): Map<String, Any> {
        return buildMap {
            put("parsed", true)
            put("mrzFormat", mrzData.documentType.name)
            put("checksumValid", mrzData.checksumValid)
            put("availableDataGroups", availableDataGroups)
        }
    }

    /**
     * Extracts raw MRZ string from DG1 TLV data.
     *
     * DG1 structure: 61 LL [5F1F LL MRZ-data]
     */
    private fun extractMrzFromDg1(dg1Data: ByteArray): String? {
        try {
            var i = 0

            // Skip outer tag (0x61)
            if (i >= dg1Data.size) return null
            val outerTag = dg1Data[i].toInt() and 0xFF
            i++
            if (outerTag != 0x61) {
                SecureLogger.w(TAG, "Unexpected DG1 outer tag: 0x${outerTag.toString(16)}")
            }

            // Skip outer length
            if (i >= dg1Data.size) return null
            val outerLen = dg1Data[i].toInt() and 0xFF
            i++
            if (outerLen > 127) {
                // Long form length
                val numLenBytes = outerLen and 0x7F
                i += numLenBytes
            }

            // Look for MRZ tag (0x5F1F)
            while (i < dg1Data.size - 2) {
                val tag1 = dg1Data[i].toInt() and 0xFF
                i++

                if (tag1 == 0x5F && i < dg1Data.size) {
                    val tag2 = dg1Data[i].toInt() and 0xFF
                    i++
                    if (tag2 == 0x1F) {
                        // Found MRZ tag, read length
                        if (i >= dg1Data.size) return null
                        var mrzLen = dg1Data[i].toInt() and 0xFF
                        i++

                        if (mrzLen > 127) {
                            // Long form length
                            val numLenBytes = mrzLen and 0x7F
                            mrzLen = 0
                            repeat(numLenBytes) {
                                if (i < dg1Data.size) {
                                    mrzLen = (mrzLen shl 8) or (dg1Data[i].toInt() and 0xFF)
                                    i++
                                }
                            }
                        }

                        // Extract MRZ data
                        if (i + mrzLen <= dg1Data.size) {
                            val mrzBytes = dg1Data.sliceArray(i until i + mrzLen)
                            return String(mrzBytes, Charsets.UTF_8)
                        }
                    }
                }
            }

            return null
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to extract MRZ from DG1", e)
            return null
        }
    }
}
