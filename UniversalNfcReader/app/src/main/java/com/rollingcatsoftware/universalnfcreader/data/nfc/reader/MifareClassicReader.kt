package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.nfc.Tag
import android.nfc.TagLostException
import android.nfc.tech.MifareClassic
import android.nfc.tech.NfcA
import android.util.Log
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.MifareClassicData
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import com.rollingcatsoftware.universalnfcreader.domain.model.SectorData
import com.rollingcatsoftware.universalnfcreader.domain.model.StudentCardData
import com.rollingcatsoftware.universalnfcreader.util.Constants
import com.rollingcatsoftware.universalnfcreader.util.toHexString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Reader for MIFARE Classic cards (1K and 4K).
 *
 * Tries to authenticate using common default keys.
 * Reads all accessible sectors and returns partial data if some sectors are protected.
 *
 * Security Note: MIFARE Classic uses the Crypto-1 cipher which is broken,
 * but this reader only uses default keys - it does not attempt to crack keys.
 */
class MifareClassicReader : BaseCardReader() {

    companion object {
        private const val TAG = "MifareClassicReader"

        // Common default keys to try
        private val DEFAULT_KEYS = listOf(
            byteArrayOf(0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte()), // Factory default
            byteArrayOf(0x00, 0x00, 0x00, 0x00, 0x00, 0x00), // All zeros
            byteArrayOf(0xA0.toByte(), 0xA1.toByte(), 0xA2.toByte(), 0xA3.toByte(), 0xA4.toByte(), 0xA5.toByte()), // MAD key
            byteArrayOf(0xB0.toByte(), 0xB1.toByte(), 0xB2.toByte(), 0xB3.toByte(), 0xB4.toByte(), 0xB5.toByte()), // Alternative
            byteArrayOf(0xD3.toByte(), 0xF7.toByte(), 0xD3.toByte(), 0xF7.toByte(), 0xD3.toByte(), 0xF7.toByte()), // NFC Forum
            byteArrayOf(0x4D.toByte(), 0x3A.toByte(), 0x99.toByte(), 0xC3.toByte(), 0x51.toByte(), 0xDD.toByte())  // Common
        )
    }

    override val supportedCardTypes = listOf(
        CardType.MIFARE_CLASSIC_1K,
        CardType.MIFARE_CLASSIC_4K,
        CardType.STUDENT_CARD_CLASSIC
    )

    override fun requiresAuthentication(): Boolean = false // Tries default keys

    override suspend fun readCard(tag: Tag): Result<CardData> = withContext(Dispatchers.IO) {
        val basicInfo = readBasicInfo(tag)
        val mifare = MifareClassic.get(tag)

        if (mifare == null) {
            Log.e(TAG, "Failed to get MifareClassic from tag")
            return@withContext Result.error(CardError.UnsupportedCard(
                detectedTechnologies = basicInfo.technologies
            ))
        }

        // Get NfcA for SAK and ATQA values
        val nfcA = NfcA.get(tag)
        val sak = nfcA?.sak?.toInt()?.and(0xFF) ?: 0
        val atqa = nfcA?.atqa ?: ByteArray(0)

        try {
            mifare.connect()

            val sectorCount = mifare.sectorCount
            val blockCount = mifare.blockCount
            val size = mifare.size
            val type = when (mifare.type) {
                MifareClassic.TYPE_CLASSIC -> if (size == 1024) CardType.MIFARE_CLASSIC_1K else CardType.MIFARE_CLASSIC_4K
                MifareClassic.TYPE_PLUS -> CardType.MIFARE_CLASSIC_1K
                MifareClassic.TYPE_PRO -> CardType.MIFARE_CLASSIC_4K
                else -> CardType.MIFARE_CLASSIC_1K
            }

            Log.d(TAG, "Reading MIFARE Classic: $sectorCount sectors, $blockCount blocks, ${size}B")

            // Read all accessible sectors
            val sectorsRead = mutableListOf<SectorData>()
            var accessibleSectors = 0

            for (sectorIndex in 0 until sectorCount) {
                val sectorData = readSector(mifare, sectorIndex)
                if (sectorData != null) {
                    sectorsRead.add(sectorData)
                    accessibleSectors++
                }
            }

            Log.d(TAG, "Read $accessibleSectors of $sectorCount sectors")

            // Check for student card patterns
            val isStudentCard = detectStudentCard(sectorsRead)

            val cardData: CardData = if (isStudentCard) {
                parseStudentCardData(basicInfo, sectorsRead, sectorCount)
            } else {
                MifareClassicData(
                    uid = basicInfo.uid.toHexString(),
                    cardType = type,
                    readTimestamp = System.currentTimeMillis(),
                    technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                    rawData = buildRawDataMap(sectorsRead),
                    sak = sak,
                    atqa = atqa,
                    sectorCount = sectorCount,
                    blockCount = blockCount,
                    size = size,
                    sectorsRead = sectorsRead,
                    accessibleSectors = accessibleSectors
                )
            }

            Result.success(cardData)

        } catch (e: TagLostException) {
            Log.e(TAG, "Tag lost: ${e.message}")
            Result.error(CardError.ConnectionLost())
        } catch (e: IOException) {
            Log.e(TAG, "IO error: ${e.message}")
            Result.error(CardError.IoError())
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error: ${e.message}", e)
            Result.error(CardError.Unknown())
        } finally {
            try {
                if (mifare.isConnected) mifare.close()
            } catch (e: IOException) {
                Log.w(TAG, "Error closing MifareClassic: ${e.message}")
            }
        }
    }

    /**
     * Read a single sector using default keys.
     */
    private fun readSector(mifare: MifareClassic, sectorIndex: Int): SectorData? {
        var successKeyType: SectorData.KeyType = SectorData.KeyType.NONE

        // Try Key A with each default key
        for (key in DEFAULT_KEYS) {
            try {
                if (mifare.authenticateSectorWithKeyA(sectorIndex, key)) {
                    successKeyType = SectorData.KeyType.KEY_A
                    break
                }
            } catch (e: IOException) {
                // Key didn't work, try next
            }
        }

        // If Key A failed, try Key B
        if (successKeyType == SectorData.KeyType.NONE) {
            for (key in DEFAULT_KEYS) {
                try {
                    if (mifare.authenticateSectorWithKeyB(sectorIndex, key)) {
                        successKeyType = SectorData.KeyType.KEY_B
                        break
                    }
                } catch (e: IOException) {
                    // Key didn't work, try next
                }
            }
        }

        if (successKeyType == SectorData.KeyType.NONE) {
            Log.d(TAG, "Sector $sectorIndex: No default key worked")
            return null
        }

        // Read all blocks in the sector
        try {
            val firstBlock = mifare.sectorToBlock(sectorIndex)
            val blocksInSector = mifare.getBlockCountInSector(sectorIndex)
            val blocks = mutableListOf<ByteArray>()
            var accessBits: ByteArray? = null

            for (blockOffset in 0 until blocksInSector) {
                val blockIndex = firstBlock + blockOffset
                val blockData = mifare.readBlock(blockIndex)
                blocks.add(blockData)

                // Log the block data
                val hexString = blockData.joinToString(" ") { "%02X".format(it) }
                val asciiString = blockData.map { b ->
                    val c = b.toInt() and 0xFF
                    if (c in 0x20..0x7E) c.toChar() else '.'
                }.joinToString("")
                Log.d(TAG, "Sector $sectorIndex Block $blockIndex: $hexString | ASCII: $asciiString")

                // Last block in sector contains access bits (bytes 6-9)
                if (blockOffset == blocksInSector - 1) {
                    accessBits = blockData.copyOfRange(6, 10)
                }
            }

            return SectorData(
                sectorNumber = sectorIndex,
                blocks = blocks,
                keyType = successKeyType,
                accessBits = accessBits
            )

        } catch (e: IOException) {
            Log.e(TAG, "Error reading sector $sectorIndex: ${e.message}")
            return null
        }
    }

    /**
     * Detect if this is a student card based on sector contents.
     */
    private fun detectStudentCard(sectors: List<SectorData>): Boolean {
        // Look for common student card patterns:
        // - Text containing "UNIV", "STU", "ID" in sectors 1-3
        // - Specific sector access patterns

        for (sector in sectors) {
            for (block in sector.blocks) {
                val text = block.decodeToStringOrNull()
                if (text != null) {
                    val upperText = text.uppercase()
                    if (upperText.contains("UNIV") ||
                        upperText.contains("STUDENT") ||
                        upperText.contains("OGRENCI") ||
                        upperText.contains("KIMLIK")) {
                        return true
                    }
                }
            }
        }
        return false
    }

    /**
     * Parse student card data from sectors.
     */
    private fun parseStudentCardData(
        basicInfo: BasicCardInfo,
        sectors: List<SectorData>,
        totalSectors: Int
    ): StudentCardData {
        var studentId: String? = null
        var studentName: String? = null
        var universityName: String? = null

        // Try to extract student info from readable sectors
        for (sector in sectors) {
            for (block in sector.blocks) {
                val text = block.decodeToStringOrNull()?.trim()
                if (text != null && text.isNotEmpty()) {
                    // Simple heuristics - this would be customized per university
                    when {
                        text.matches(Regex("\\d{7,12}")) -> studentId = text
                        text.contains("Üniversite", ignoreCase = true) ||
                        text.contains("University", ignoreCase = true) -> universityName = text
                    }
                }
            }
        }

        return StudentCardData(
            uid = basicInfo.uid.toHexString(),
            cardType = CardType.STUDENT_CARD_CLASSIC,
            readTimestamp = System.currentTimeMillis(),
            technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
            rawData = buildRawDataMap(sectors),
            studentId = studentId,
            studentName = studentName,
            universityName = universityName,
            sectorsRead = sectors.size,
            totalSectors = totalSectors
        )
    }

    /**
     * Try to decode ByteArray as ASCII string.
     */
    private fun ByteArray.decodeToStringOrNull(): String? {
        return try {
            // Filter out non-printable characters
            val filtered = filter { it in 0x20..0x7E || it.toInt() == 0 }
            if (filtered.isEmpty()) null
            else String(filtered.toByteArray(), Charsets.US_ASCII).takeIf { it.isNotBlank() }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Build raw data map for debugging.
     */
    private fun buildRawDataMap(sectors: List<SectorData>): Map<String, Any> {
        return buildMap {
            put("sectorsRead", sectors.size)
            put("sectorNumbers", sectors.map { it.sectorNumber })
            // Don't include raw block data in map to avoid memory issues
        }
    }

    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> {
        // If custom key provided, use it
        if (authData is AuthenticationData.MifareKeyData) {
            return readCardWithKey(tag, authData)
        }
        return readCard(tag)
    }

    /**
     * Read card with specific MIFARE key.
     */
    private suspend fun readCardWithKey(
        tag: Tag,
        keyData: AuthenticationData.MifareKeyData
    ): Result<CardData> = withContext(Dispatchers.IO) {
        val basicInfo = readBasicInfo(tag)
        val mifare = MifareClassic.get(tag)
            ?: return@withContext Result.error(CardError.UnsupportedCard())

        try {
            mifare.connect()
            val sectorsRead = mutableListOf<SectorData>()
            val key = keyData.getKey()

            for (sectorIndex in 0 until mifare.sectorCount) {
                val authenticated = when (keyData.keyType) {
                    AuthenticationData.MifareKeyData.MifareKeyType.KEY_A ->
                        mifare.authenticateSectorWithKeyA(sectorIndex, key)
                    AuthenticationData.MifareKeyData.MifareKeyType.KEY_B ->
                        mifare.authenticateSectorWithKeyB(sectorIndex, key)
                }

                if (authenticated) {
                    val firstBlock = mifare.sectorToBlock(sectorIndex)
                    val blocksInSector = mifare.getBlockCountInSector(sectorIndex)
                    val blocks = (0 until blocksInSector).map { offset ->
                        mifare.readBlock(firstBlock + offset)
                    }
                    sectorsRead.add(SectorData(
                        sectorNumber = sectorIndex,
                        blocks = blocks,
                        keyType = if (keyData.keyType == AuthenticationData.MifareKeyData.MifareKeyType.KEY_A)
                            SectorData.KeyType.KEY_A else SectorData.KeyType.KEY_B,
                        accessBits = blocks.lastOrNull()?.copyOfRange(6, 10)
                    ))
                }
            }

            Result.success(MifareClassicData(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.MIFARE_CLASSIC_1K,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildRawDataMap(sectorsRead),
                sectorCount = mifare.sectorCount,
                blockCount = mifare.blockCount,
                size = mifare.size,
                sectorsRead = sectorsRead,
                accessibleSectors = sectorsRead.size
            ))

        } catch (e: Exception) {
            Log.e(TAG, "Error with custom key: ${e.message}")
            Result.error(CardError.AuthenticationFailed())
        } finally {
            keyData.clear()
            try { mifare.close() } catch (_: Exception) {}
        }
    }
}
