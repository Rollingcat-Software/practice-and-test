package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.nfc.Tag
import android.nfc.TagLostException
import android.nfc.tech.MifareUltralight
import android.util.Log
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.MifareUltralightData
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import com.rollingcatsoftware.universalnfcreader.domain.model.UltralightType
import com.rollingcatsoftware.universalnfcreader.util.toHexString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Reader for MIFARE Ultralight cards.
 *
 * Supports:
 * - MIFARE Ultralight (16 pages, 64 bytes)
 * - MIFARE Ultralight C (48 pages, 192 bytes, optional 3DES auth)
 * - NTAG 213/215/216
 *
 * No authentication required for basic read operations.
 */
class MifareUltralightReader : BaseCardReader() {

    companion object {
        private const val TAG = "MifareUltralightReader"
        private const val PAGE_SIZE = 4
    }

    override val supportedCardTypes = listOf(
        CardType.MIFARE_ULTRALIGHT,
        CardType.MIFARE_ULTRALIGHT_C
    )

    override fun requiresAuthentication(): Boolean = false

    override suspend fun readCard(tag: Tag): Result<CardData> = withContext(Dispatchers.IO) {
        val basicInfo = readBasicInfo(tag)
        val ultralight = MifareUltralight.get(tag)

        if (ultralight == null) {
            Log.e(TAG, "Failed to get MifareUltralight from tag")
            return@withContext Result.error(
                CardError.UnsupportedCard(
                    detectedTechnologies = basicInfo.technologies
                )
            )
        }

        try {
            ultralight.connect()

            // Determine card type
            val ultralightType = determineUltralightType(ultralight)
            val pageCount = getPageCount(ultralightType)

            Log.d(TAG, "Reading MIFARE Ultralight type: $ultralightType, $pageCount pages")

            // Read all pages
            val pages = mutableListOf<ByteArray>()
            var readError = false

            // Read pages in groups of 4 (ultralight.readPages returns 4 pages at once)
            var pageIndex = 0
            while (pageIndex < pageCount && !readError) {
                try {
                    val pagesData = ultralight.readPages(pageIndex)
                    // Split into individual pages
                    for (i in 0 until 4) {
                        if (pageIndex + i < pageCount) {
                            val pageData = pagesData.copyOfRange(i * PAGE_SIZE, (i + 1) * PAGE_SIZE)
                            pages.add(pageData)
                        }
                    }
                    pageIndex += 4
                } catch (e: IOException) {
                    Log.w(TAG, "Error reading page $pageIndex: ${e.message}")
                    readError = true
                }
            }

            Log.d(TAG, "Read ${pages.size} of $pageCount pages")

            // Try to extract NDEF message
            val ndefMessage = extractNdefMessage(pages)

            val cardType = when (ultralightType) {
                UltralightType.ULTRALIGHT_C -> CardType.MIFARE_ULTRALIGHT_C
                else -> CardType.MIFARE_ULTRALIGHT
            }

            val cardData = MifareUltralightData(
                uid = basicInfo.uid.toHexString(),
                cardType = cardType,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildRawDataMap(pages, ultralightType),
                pageCount = pages.size,
                ultralightType = ultralightType,
                pages = pages,
                ndefMessage = ndefMessage
            )

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
                if (ultralight.isConnected) ultralight.close()
            } catch (e: IOException) {
                Log.w(TAG, "Error closing MifareUltralight: ${e.message}")
            }
        }
    }

    /**
     * Determine the specific Ultralight card type.
     */
    private fun determineUltralightType(ultralight: MifareUltralight): UltralightType {
        return when (ultralight.type) {
            MifareUltralight.TYPE_ULTRALIGHT -> {
                // Try to detect NTAG by reading signature or version
                tryDetectNtag(ultralight) ?: UltralightType.ULTRALIGHT
            }

            MifareUltralight.TYPE_ULTRALIGHT_C -> UltralightType.ULTRALIGHT_C
            else -> UltralightType.UNKNOWN
        }
    }

    /**
     * Try to detect NTAG type by reading version command.
     */
    private fun tryDetectNtag(ultralight: MifareUltralight): UltralightType? {
        try {
            // GET_VERSION command
            val versionCmd = byteArrayOf(0x60)
            val response = ultralight.transceive(versionCmd)

            if (response.size >= 8) {
                val productType = response[2].toInt() and 0xFF
                val productSubtype = response[3].toInt() and 0xFF
                val storageSize = response[6].toInt() and 0xFF

                // NXP product types
                if (response[1].toInt() and 0xFF == 0x04) { // NXP vendor
                    return when {
                        productType == 0x04 && storageSize == 0x0F -> UltralightType.NTAG_213
                        productType == 0x04 && storageSize == 0x11 -> UltralightType.NTAG_215
                        productType == 0x04 && storageSize == 0x13 -> UltralightType.NTAG_216
                        productType == 0x03 && storageSize == 0x0B -> UltralightType.ULTRALIGHT_EV1_MF0UL11
                        productType == 0x03 && storageSize == 0x0E -> UltralightType.ULTRALIGHT_EV1_MF0UL21
                        else -> null
                    }
                }
            }
        } catch (e: IOException) {
            Log.d(TAG, "GET_VERSION not supported: ${e.message}")
        }
        return null
    }

    /**
     * Get expected page count for card type.
     */
    private fun getPageCount(type: UltralightType): Int {
        return when (type) {
            UltralightType.ULTRALIGHT -> 16
            UltralightType.ULTRALIGHT_C -> 48
            UltralightType.ULTRALIGHT_EV1_MF0UL11 -> 20
            UltralightType.ULTRALIGHT_EV1_MF0UL21 -> 41
            UltralightType.NTAG_213 -> 45
            UltralightType.NTAG_215 -> 135
            UltralightType.NTAG_216 -> 231
            UltralightType.UNKNOWN -> 16 // Default to standard Ultralight
        }
    }

    /**
     * Extract NDEF message from pages if present.
     */
    private fun extractNdefMessage(pages: List<ByteArray>): String? {
        if (pages.size < 5) return null

        try {
            // NDEF data starts at page 4 for Ultralight
            // Check for NDEF TLV (0x03)
            val page4 = pages[4]
            if (page4[0] != 0x03.toByte()) return null

            val length = page4[1].toInt() and 0xFF
            if (length == 0) return null

            // Combine pages to get NDEF data
            val ndefData = ByteArray(length)
            var offset = 2 // Start after TLV header
            var dataIndex = 0
            var pageIndex = 4

            while (dataIndex < length && pageIndex < pages.size) {
                val page = pages[pageIndex]
                while (offset < PAGE_SIZE && dataIndex < length) {
                    ndefData[dataIndex++] = page[offset++]
                }
                offset = 0
                pageIndex++
            }

            // Try to parse as text record
            return parseNdefTextRecord(ndefData)

        } catch (e: Exception) {
            Log.d(TAG, "Error extracting NDEF: ${e.message}")
            return null
        }
    }

    /**
     * Parse NDEF Text Record.
     */
    private fun parseNdefTextRecord(data: ByteArray): String? {
        if (data.size < 3) return null

        try {
            // Check for text record type (TNF=1, Type="T")
            val tnf = data[0].toInt() and 0x07
            if (tnf != 1) return null

            val typeLength = data[1].toInt() and 0xFF
            val payloadLength = data[2].toInt() and 0xFF

            if (data.size < 3 + typeLength + payloadLength) return null
            if (typeLength != 1 || data[3] != 'T'.code.toByte()) return null

            // Payload starts after header
            val payloadStart = 3 + typeLength
            if (payloadStart >= data.size) return null

            val statusByte = data[payloadStart].toInt() and 0xFF
            val langCodeLength = statusByte and 0x3F
            val textStart = payloadStart + 1 + langCodeLength

            if (textStart >= data.size) return null

            return String(data, textStart, data.size - textStart, Charsets.UTF_8)

        } catch (e: Exception) {
            Log.d(TAG, "Error parsing NDEF text: ${e.message}")
            return null
        }
    }

    /**
     * Build raw data map for debugging.
     */
    private fun buildRawDataMap(pages: List<ByteArray>, type: UltralightType): Map<String, Any> {
        return buildMap {
            put("cardType", type.name)
            put("pagesRead", pages.size)
            // Include first few pages as hex for debugging
            pages.take(8).forEachIndexed { index, page ->
                put("page$index", page.toHexString())
            }
        }
    }

    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> {
        // Ultralight C 3DES authentication not implemented yet
        // Just read without auth for now
        return readCard(tag)
    }
}
