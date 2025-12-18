package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.nfc.Tag
import android.nfc.tech.IsoDep
import android.nfc.tech.NfcA
import android.nfc.tech.NfcB
import android.nfc.tech.NfcF
import android.nfc.tech.NfcV
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.GenericCardData
import com.rollingcatsoftware.universalnfcreader.domain.model.Iso15693Data
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import com.rollingcatsoftware.universalnfcreader.util.Constants
import com.rollingcatsoftware.universalnfcreader.util.toHexString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Generic reader for unidentified or basic NFC cards.
 *
 * Reads basic information available on any NFC tag:
 * - UID
 * - Technologies
 * - ATQA (for NFC-A)
 * - SAK (for NFC-A)
 * - ATS (for IsoDep)
 */
class GenericCardReader : BaseCardReader() {

    companion object {
        private const val TAG = "GenericCardReader"
    }

    override val supportedCardTypes = listOf(
        CardType.ISO_14443_A,
        CardType.ISO_14443_B,
        CardType.ISO_15693,
        CardType.FELICA,
        CardType.UNKNOWN
    )

    override fun requiresAuthentication(): Boolean = false

    override suspend fun readCard(tag: Tag): Result<CardData> = withContext(Dispatchers.IO) {
        val basicInfo = readBasicInfo(tag)
        val techList = tag.techList.toList()

        SecureLogger.d(TAG, "Reading generic card. Technologies: ${techList.joinToString()}")

        try {
            // Try to get more info based on available technologies
            val cardData = when {
                techList.any { it.contains("NfcV") } -> readNfcV(tag, basicInfo)
                techList.any { it.contains("IsoDep") } -> readIsoDep(tag, basicInfo)
                techList.any { it.contains("NfcA") } -> readNfcA(tag, basicInfo)
                techList.any { it.contains("NfcB") } -> readNfcB(tag, basicInfo)
                techList.any { it.contains("NfcF") } -> readNfcF(tag, basicInfo)
                else -> readBasicOnly(basicInfo)
            }

            Result.success(cardData)

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading generic card: ${e.message}", e)
            // Return at least basic info on error
            Result.success(readBasicOnly(basicInfo))
        }
    }

    /**
     * Read ISO 15693 (NfcV) card.
     */
    private fun readNfcV(tag: Tag, basicInfo: BasicCardInfo): CardData {
        val nfcV = NfcV.get(tag)

        return try {
            nfcV?.connect()

            val dsfId = nfcV?.dsfId?.toInt()?.and(0xFF) ?: 0
            val responseFlags = nfcV?.responseFlags?.toInt()?.and(0xFF) ?: 0

            // Try to read system info
            val (blockSize, blockCount, manufacturer) = readNfcVSystemInfo(nfcV)

            // Try to read first few blocks
            val blocks = readNfcVBlocks(nfcV, minOf(blockCount, 16))

            Iso15693Data(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.ISO_15693,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildMap {
                    put("dsfId", dsfId)
                    put("responseFlags", responseFlags)
                    put("blockSize", blockSize)
                    put("blockCount", blockCount)
                },
                dsfId = dsfId,
                responseFlags = responseFlags,
                blockSize = blockSize,
                blockCount = blockCount,
                manufacturer = manufacturer,
                blocks = blocks
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading NfcV: ${e.message}")
            Iso15693Data(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.ISO_15693,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = emptyMap()
            )
        } finally {
            try {
                nfcV?.close()
            } catch (_: Exception) {
            }
        }
    }

    /**
     * Read NfcV system info.
     */
    private fun readNfcVSystemInfo(nfcV: NfcV?): Triple<Int, Int, String> {
        if (nfcV == null) return Triple(4, 0, "Unknown")

        return try {
            // GET SYSTEM INFO command
            val cmd = byteArrayOf(
                0x02, // Flags: High data rate
                0x2B  // GET SYSTEM INFO
            )
            val response = nfcV.transceive(cmd)

            if (response.isNotEmpty() && response[0] == 0x00.toByte()) {
                // Parse response
                val infoFlags = response[1].toInt() and 0xFF
                val hasBlockInfo = (infoFlags and 0x10) != 0

                var blockSize = 4
                var blockCount = 0

                if (hasBlockInfo && response.size >= 12) {
                    blockSize = (response[10].toInt() and 0x1F) + 1
                    blockCount = (response[11].toInt() and 0xFF) + 1
                }

                // Get manufacturer from UID
                val manufacturer = getManufacturerName(nfcV.tag.id[1])

                Triple(blockSize, blockCount, manufacturer)
            } else {
                Triple(4, 0, "Unknown")
            }
        } catch (e: Exception) {
            SecureLogger.d(TAG, "GET SYSTEM INFO failed: ${e.message}")
            Triple(4, 0, "Unknown")
        }
    }

    /**
     * Read NfcV blocks.
     */
    private fun readNfcVBlocks(nfcV: NfcV?, count: Int): List<ByteArray> {
        if (nfcV == null || count == 0) return emptyList()

        val blocks = mutableListOf<ByteArray>()
        for (blockNum in 0 until count) {
            try {
                // READ SINGLE BLOCK command
                val cmd = byteArrayOf(
                    0x02, // Flags
                    0x20, // READ SINGLE BLOCK
                    blockNum.toByte()
                )
                val response = nfcV.transceive(cmd)
                if (response.isNotEmpty() && response[0] == 0x00.toByte()) {
                    blocks.add(response.copyOfRange(1, response.size))
                }
            } catch (e: Exception) {
                break
            }
        }
        return blocks
    }

    /**
     * Get manufacturer name from IC manufacturer code.
     */
    private fun getManufacturerName(code: Byte): String {
        return when (code.toInt() and 0xFF) {
            0x01 -> "Motorola"
            0x02 -> "STMicroelectronics"
            0x03 -> "Hitachi"
            0x04 -> "NXP Semiconductors"
            0x05 -> "Infineon Technologies"
            0x06 -> "Cylink"
            0x07 -> "Texas Instruments"
            0x08 -> "Fujitsu"
            0x09 -> "Matsushita"
            0x0A -> "NEC"
            0x0B -> "Oki Electric"
            0x0C -> "Toshiba"
            0x0D -> "Mitsubishi"
            0x0E -> "Samsung"
            0x0F -> "Hynix"
            0x16 -> "EM Microelectronic"
            else -> "Unknown (0x${"%02X".format(code)})"
        }
    }

    /**
     * Read IsoDep card.
     */
    private fun readIsoDep(tag: Tag, basicInfo: BasicCardInfo): CardData {
        val isoDep = IsoDep.get(tag)

        return try {
            isoDep?.connect()
            isoDep?.timeout = Constants.Timeout.SHORT_MS

            val historicalBytes = isoDep?.historicalBytes
            val hiLayerResponse = isoDep?.hiLayerResponse

            GenericCardData(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.ISO_14443_A,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildMap {
                    historicalBytes?.let { put("historicalBytes", it.toHexString()) }
                    hiLayerResponse?.let { put("hiLayerResponse", it.toHexString()) }
                    put("maxTransceiveLength", isoDep?.maxTransceiveLength ?: 0)
                    put(
                        "isExtendedLengthApduSupported",
                        isoDep?.isExtendedLengthApduSupported ?: false
                    )
                },
                historicalBytes = historicalBytes
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading IsoDep: ${e.message}")
            readBasicOnly(basicInfo)
        } finally {
            try {
                isoDep?.close()
            } catch (_: Exception) {
            }
        }
    }

    /**
     * Read NfcA card.
     */
    private fun readNfcA(tag: Tag, basicInfo: BasicCardInfo): CardData {
        val nfcA = NfcA.get(tag)

        return try {
            GenericCardData(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.ISO_14443_A,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildMap {
                    put("atqa", nfcA?.atqa?.toHexString() ?: "")
                    put("sak", nfcA?.sak?.toInt()?.and(0xFF) ?: 0)
                    put("maxTransceiveLength", nfcA?.maxTransceiveLength ?: 0)
                },
                atqa = nfcA?.atqa,
                sak = nfcA?.sak
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading NfcA: ${e.message}")
            readBasicOnly(basicInfo)
        }
    }

    /**
     * Read NfcB card.
     */
    private fun readNfcB(tag: Tag, basicInfo: BasicCardInfo): CardData {
        val nfcB = NfcB.get(tag)

        return try {
            GenericCardData(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.ISO_14443_B,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildMap {
                    nfcB?.applicationData?.let { put("applicationData", it.toHexString()) }
                    nfcB?.protocolInfo?.let { put("protocolInfo", it.toHexString()) }
                    put("maxTransceiveLength", nfcB?.maxTransceiveLength ?: 0)
                }
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading NfcB: ${e.message}")
            readBasicOnly(basicInfo)
        }
    }

    /**
     * Read NfcF (FeliCa) card.
     */
    private fun readNfcF(tag: Tag, basicInfo: BasicCardInfo): CardData {
        val nfcF = NfcF.get(tag)

        return try {
            GenericCardData(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.FELICA,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildMap {
                    nfcF?.systemCode?.let { put("systemCode", it.toHexString()) }
                    nfcF?.manufacturer?.let { put("manufacturer", it.toHexString()) }
                    put("maxTransceiveLength", nfcF?.maxTransceiveLength ?: 0)
                }
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error reading NfcF: ${e.message}")
            readBasicOnly(basicInfo)
        }
    }

    /**
     * Return basic card info only.
     */
    private fun readBasicOnly(basicInfo: BasicCardInfo): CardData {
        return GenericCardData(
            uid = basicInfo.uid.toHexString(),
            cardType = CardType.UNKNOWN,
            readTimestamp = System.currentTimeMillis(),
            technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
            rawData = emptyMap()
        )
    }

    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> {
        // Generic reader doesn't support authentication
        return readCard(tag)
    }
}
