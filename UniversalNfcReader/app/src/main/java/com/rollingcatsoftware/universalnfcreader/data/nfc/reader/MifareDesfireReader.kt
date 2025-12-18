package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.nfc.Tag
import android.nfc.TagLostException
import android.nfc.tech.IsoDep
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.DesfireVersion
import com.rollingcatsoftware.universalnfcreader.domain.model.IstanbulkartData
import com.rollingcatsoftware.universalnfcreader.domain.model.MifareDesfireData
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import com.rollingcatsoftware.universalnfcreader.util.Constants
import com.rollingcatsoftware.universalnfcreader.util.getStatusWord
import com.rollingcatsoftware.universalnfcreader.util.toHexString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Reader for MIFARE DESFire cards (including Istanbulkart).
 *
 * Reads publicly available information without authentication:
 * - Card UID
 * - DESFire version info
 * - Application IDs
 * - Free memory
 *
 * Protected data (balance, transactions) requires proprietary keys.
 */
class MifareDesfireReader : BaseCardReader() {

    companion object {
        private const val TAG = "MifareDesfireReader"
    }

    override val supportedCardTypes = listOf(
        CardType.MIFARE_DESFIRE,
        CardType.ISTANBULKART
    )

    override fun requiresAuthentication(): Boolean = false

    override suspend fun readCard(tag: Tag): Result<CardData> = withContext(Dispatchers.IO) {
        val basicInfo = readBasicInfo(tag)
        val isoDep = IsoDep.get(tag)

        if (isoDep == null) {
            SecureLogger.e(TAG, "Failed to get IsoDep from tag")
            return@withContext Result.error(
                CardError.UnsupportedCard(
                    detectedTechnologies = basicInfo.technologies
                )
            )
        }

        try {
            isoDep.connect()
            isoDep.timeout = Constants.Timeout.DEFAULT_MS

            // Read DESFire version
            val version = readDesfireVersion(isoDep)

            // Read application IDs
            val appIds = readApplicationIds(isoDep)

            // Read free memory (may fail on some cards)
            val freeMemory = readFreeMemory(isoDep)

            // Determine if this is Istanbulkart
            val cardType = determineCardType(tag, version)

            val cardData: CardData = if (cardType == CardType.ISTANBULKART) {
                IstanbulkartData(
                    uid = basicInfo.uid.toHexString(),
                    cardType = CardType.ISTANBULKART,
                    readTimestamp = System.currentTimeMillis(),
                    technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                    rawData = buildRawDataMap(version, appIds, freeMemory),
                    desfireVersion = version,
                    applicationIds = appIds,
                    freeMemory = freeMemory
                )
            } else {
                MifareDesfireData(
                    uid = basicInfo.uid.toHexString(),
                    cardType = CardType.MIFARE_DESFIRE,
                    readTimestamp = System.currentTimeMillis(),
                    technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                    rawData = buildRawDataMap(version, appIds, freeMemory),
                    version = version,
                    applicationIds = appIds,
                    freeMemory = freeMemory,
                    cardSize = version?.storageSizeBytes
                )
            }

            SecureLogger.d(TAG, "Successfully read DESFire card: ${cardData.uid}")
            Result.success(cardData)

        } catch (e: TagLostException) {
            SecureLogger.e(TAG, "Tag lost: ${e.message}")
            Result.error(CardError.ConnectionLost())
        } catch (e: IOException) {
            SecureLogger.e(TAG, "IO error: ${e.message}")
            Result.error(CardError.IoError())
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Unexpected error: ${e.message}", e)
            Result.error(CardError.Unknown())
        } finally {
            try {
                if (isoDep.isConnected) isoDep.close()
            } catch (e: IOException) {
                SecureLogger.w(TAG, "Error closing IsoDep: ${e.message}")
            }
        }
    }

    /**
     * Read DESFire version information.
     * Requires 3 GET_VERSION commands to get full info.
     */
    private fun readDesfireVersion(isoDep: IsoDep): DesfireVersion? {
        try {
            // First frame
            val cmd1 = wrapNativeCommand(Constants.DESFire.CMD_GET_VERSION)
            val response1 = isoDep.transceive(cmd1)
            if (!isDesfireSuccess(response1)) {
                SecureLogger.d(TAG, "GET_VERSION failed: ${response1.toHexString()}")
                return null
            }

            // Second frame
            val cmd2 = wrapNativeCommand(Constants.DESFire.CMD_ADDITIONAL_FRAME)
            val response2 = isoDep.transceive(cmd2)
            if (!isDesfireSuccess(response2)) return null

            // Third frame
            val cmd3 = wrapNativeCommand(Constants.DESFire.CMD_ADDITIONAL_FRAME)
            val response3 = isoDep.transceive(cmd3)
            if (!isDesfireFinalSuccess(response3)) return null

            // Parse version data
            val data1 = response1.dropLast(2).toByteArray()
            val data2 = response2.dropLast(2).toByteArray()
            val data3 = response3.dropLast(2).toByteArray()

            if (data1.size < 7 || data2.size < 7 || data3.size < 14) {
                SecureLogger.w(TAG, "Incomplete version data")
                return null
            }

            val version = DesfireVersion(
                hardwareVendorId = data1[0].toInt() and 0xFF,
                hardwareType = data1[1].toInt() and 0xFF,
                hardwareSubType = data1[2].toInt() and 0xFF,
                hardwareMajorVersion = data1[3].toInt() and 0xFF,
                hardwareMinorVersion = data1[4].toInt() and 0xFF,
                hardwareStorageSize = data1[5].toInt() and 0xFF,
                hardwareProtocol = data1[6].toInt() and 0xFF,
                softwareVendorId = data2[0].toInt() and 0xFF,
                softwareType = data2[1].toInt() and 0xFF,
                softwareSubType = data2[2].toInt() and 0xFF,
                softwareMajorVersion = data2[3].toInt() and 0xFF,
                softwareMinorVersion = data2[4].toInt() and 0xFF,
                softwareStorageSize = data2[5].toInt() and 0xFF,
                softwareProtocol = data2[6].toInt() and 0xFF,
                uid = data3.copyOfRange(0, 7),
                batchNumber = data3.copyOfRange(7, 12),
                productionWeek = data3[12].toInt() and 0xFF,
                productionYear = data3[13].toInt() and 0xFF
            )

            // Log detailed version info
            SecureLogger.d(TAG, "═══ DESFIRE VERSION DETAILS ═══")
            SecureLogger.d(
                TAG,
                "Hardware: Vendor=0x${
                    String.format(
                        "%02X",
                        version.hardwareVendorId
                    )
                } (${if (version.hardwareVendorId == 0x04) "NXP" else "Unknown"})"
            )
            SecureLogger.d(
                TAG,
                "Hardware: Type=0x${
                    String.format(
                        "%02X",
                        version.hardwareType
                    )
                } SubType=0x${String.format("%02X", version.hardwareSubType)}"
            )
            SecureLogger.d(
                TAG,
                "Hardware: Version=${version.hardwareMajorVersion}.${version.hardwareMinorVersion}"
            )
            SecureLogger.d(
                TAG,
                "Hardware: Storage=0x${
                    String.format(
                        "%02X",
                        version.hardwareStorageSize
                    )
                } (${version.storageSizeBytes} bytes)"
            )
            SecureLogger.d(TAG, "Hardware: Protocol=0x${String.format("%02X", version.hardwareProtocol)}")
            SecureLogger.d(TAG, "Software: Vendor=0x${String.format("%02X", version.softwareVendorId)}")
            SecureLogger.d(
                TAG,
                "Software: Type=0x${
                    String.format(
                        "%02X",
                        version.softwareType
                    )
                } SubType=0x${String.format("%02X", version.softwareSubType)}"
            )
            SecureLogger.d(
                TAG,
                "Software: Version=${version.softwareMajorVersion}.${version.softwareMinorVersion}"
            )
            SecureLogger.d(TAG, "Software: Protocol=0x${String.format("%02X", version.softwareProtocol)}")
            SecureLogger.d(TAG, "Card UID: ${version.uid.joinToString("") { "%02X".format(it) }}")
            SecureLogger.d(
                TAG,
                "Batch Number: ${version.batchNumber.joinToString("") { "%02X".format(it) }}"
            )
            SecureLogger.d(
                TAG,
                "Production: Week ${version.productionWeek}, Year 20${
                    String.format(
                        "%02d",
                        version.productionYear
                    )
                }"
            )
            SecureLogger.d(TAG, "═══════════════════════════════")

            return version

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading version: ${e.message}")
            return null
        }
    }

    /**
     * Read application IDs from the card.
     */
    private fun readApplicationIds(isoDep: IsoDep): List<String> {
        try {
            // First select PICC application (master)
            val selectPicc = wrapNativeCommand(
                Constants.DESFire.CMD_SELECT_APPLICATION,
                Constants.DESFire.PICC_APPLICATION
            )
            val selectResponse = isoDep.transceive(selectPicc)
            if (!isDesfireFinalSuccess(selectResponse)) {
                SecureLogger.d(TAG, "Failed to select PICC application")
            }

            // Get application IDs
            val cmd = wrapNativeCommand(Constants.DESFire.CMD_GET_APPLICATION_IDS)
            val response = isoDep.transceive(cmd)

            if (!isDesfireFinalSuccess(response)) {
                SecureLogger.d(TAG, "GET_APPLICATION_IDS failed")
                return emptyList()
            }

            val data = response.dropLast(2).toByteArray()
            val appIds = mutableListOf<String>()

            // Each AID is 3 bytes
            for (i in data.indices step 3) {
                if (i + 3 <= data.size) {
                    val aid = data.copyOfRange(i, i + 3)
                    appIds.add(aid.toHexString())
                }
            }

            SecureLogger.d(TAG, "Found ${appIds.size} applications: $appIds")
            return appIds

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading application IDs: ${e.message}")
            return emptyList()
        }
    }

    /**
     * Read free memory from the card.
     */
    private fun readFreeMemory(isoDep: IsoDep): Int? {
        try {
            val cmd = wrapNativeCommand(Constants.DESFire.CMD_GET_FREE_MEMORY)
            val response = isoDep.transceive(cmd)

            if (!isDesfireFinalSuccess(response)) {
                SecureLogger.d(TAG, "GET_FREE_MEMORY not supported or failed")
                return null
            }

            val data = response.dropLast(2).toByteArray()
            if (data.size >= 3) {
                // Little endian 3-byte value
                return (data[0].toInt() and 0xFF) or
                        ((data[1].toInt() and 0xFF) shl 8) or
                        ((data[2].toInt() and 0xFF) shl 16)
            }
            return null

        } catch (e: Exception) {
            SecureLogger.d(TAG, "Error reading free memory: ${e.message}")
            return null
        }
    }

    /**
     * Wrap a native DESFire command in ISO 7816 APDU.
     */
    private fun wrapNativeCommand(ins: Byte, data: ByteArray? = null): ByteArray {
        val dataLen = data?.size ?: 0
        return if (data != null) {
            byteArrayOf(
                0x90.toByte(), // CLA: DESFire native wrapped
                ins,           // INS: Command
                0x00,          // P1
                0x00,          // P2
                dataLen.toByte(), // Lc
                *data,         // Data
                0x00           // Le
            )
        } else {
            byteArrayOf(
                0x90.toByte(), // CLA: DESFire native wrapped
                ins,           // INS: Command
                0x00,          // P1
                0x00,          // P2
                0x00           // Le
            )
        }
    }

    /**
     * Check if response indicates success with more data.
     */
    private fun isDesfireSuccess(response: ByteArray): Boolean {
        val sw = response.getStatusWord()
        return sw == 0x91AF || sw == 0x9100
    }

    /**
     * Check if response indicates final success.
     */
    private fun isDesfireFinalSuccess(response: ByteArray): Boolean {
        return response.getStatusWord() == 0x9100
    }

    /**
     * Determine specific card type from version and UID.
     */
    private fun determineCardType(tag: Tag, version: DesfireVersion?): CardType {
        val uid = tag.id

        // Istanbulkart: NXP chip, 7-byte UID
        if (uid.size == 7 && uid[0] == Constants.NXP_MANUFACTURER_CODE) {
            // Additional checks could be made based on version
            if (version != null && version.hardwareVendorId == 0x04) {
                return CardType.ISTANBULKART
            }
        }

        return CardType.MIFARE_DESFIRE
    }

    /**
     * Build raw data map for debugging.
     */
    private fun buildRawDataMap(
        version: DesfireVersion?,
        appIds: List<String>,
        freeMemory: Int?
    ): Map<String, Any> {
        return buildMap {
            version?.let {
                put("hardwareVersion", "${it.hardwareMajorVersion}.${it.hardwareMinorVersion}")
                put("softwareVersion", "${it.softwareMajorVersion}.${it.softwareMinorVersion}")
                put("storageSize", it.storageSizeBytes)
            }
            put("applicationIds", appIds)
            freeMemory?.let { put("freeMemory", it) }
        }
    }

    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> {
        // DESFire authentication requires proprietary keys
        // For now, just read public data
        return readCard(tag)
    }
}
