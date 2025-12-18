package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.nfc.NdefMessage
import android.nfc.Tag
import android.nfc.TagLostException
import android.nfc.tech.Ndef
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardError
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.NdefData
import com.rollingcatsoftware.universalnfcreader.domain.model.NdefRecord
import com.rollingcatsoftware.universalnfcreader.domain.model.Result
import com.rollingcatsoftware.universalnfcreader.util.toHexString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Reader for NDEF formatted NFC tags.
 *
 * Reads NFC Forum standardized NDEF messages containing:
 * - Text records
 * - URI records
 * - Smart poster records
 * - MIME type records
 */
class NdefReader : BaseCardReader() {

    companion object {
        private const val TAG = "NdefReader"

        // NDEF Type Name Format (TNF) values
        const val TNF_EMPTY = 0x00.toShort()
        const val TNF_WELL_KNOWN = 0x01.toShort()
        const val TNF_MIME_MEDIA = 0x02.toShort()
        const val TNF_ABSOLUTE_URI = 0x03.toShort()
        const val TNF_EXTERNAL_TYPE = 0x04.toShort()
        const val TNF_UNKNOWN = 0x05.toShort()
        const val TNF_UNCHANGED = 0x06.toShort()

        // URI Prefix codes as per NFC Forum URI Record Type Definition
        val URI_PREFIXES = mapOf(
            0x00 to "",
            0x01 to "http://www.",
            0x02 to "https://www.",
            0x03 to "http://",
            0x04 to "https://",
            0x05 to "tel:",
            0x06 to "mailto:",
            0x07 to "ftp://anonymous:anonymous@",
            0x08 to "ftp://ftp.",
            0x09 to "ftps://",
            0x0A to "sftp://",
            0x0B to "smb://",
            0x0C to "nfs://",
            0x0D to "ftp://",
            0x0E to "dav://",
            0x0F to "news:",
            0x10 to "telnet://",
            0x11 to "imap:",
            0x12 to "rtsp://",
            0x13 to "urn:",
            0x14 to "pop:",
            0x15 to "sip:",
            0x16 to "sips:",
            0x17 to "tftp:",
            0x18 to "btspp://",
            0x19 to "btl2cap://",
            0x1A to "btgoep://",
            0x1B to "tcpobex://",
            0x1C to "irdaobex://",
            0x1D to "file://",
            0x1E to "urn:epc:id:",
            0x1F to "urn:epc:tag:",
            0x20 to "urn:epc:pat:",
            0x21 to "urn:epc:raw:",
            0x22 to "urn:epc:",
            0x23 to "urn:nfc:"
        )
    }

    override val supportedCardTypes = listOf(CardType.NDEF)

    override fun requiresAuthentication(): Boolean = false

    override suspend fun readCard(tag: Tag): Result<CardData> = withContext(Dispatchers.IO) {
        val basicInfo = readBasicInfo(tag)
        val ndef = Ndef.get(tag)

        if (ndef == null) {
            SecureLogger.e(TAG, "Failed to get Ndef from tag")
            return@withContext Result.error(
                CardError.UnsupportedCard(
                    detectedTechnologies = basicInfo.technologies
                )
            )
        }

        try {
            ndef.connect()

            val ndefMessage = ndef.ndefMessage
            val isWritable = ndef.isWritable
            val maxSize = ndef.maxSize
            val ndefType = ndef.type

            SecureLogger.d(TAG, "Reading NDEF tag: type=$ndefType, maxSize=$maxSize, writable=$isWritable")

            val records = if (ndefMessage != null) {
                parseNdefMessage(ndefMessage)
            } else {
                SecureLogger.d(TAG, "No NDEF message on tag")
                emptyList()
            }

            val usedSize = ndefMessage?.byteArrayLength ?: 0

            val cardData = NdefData(
                uid = basicInfo.uid.toHexString(),
                cardType = CardType.NDEF,
                readTimestamp = System.currentTimeMillis(),
                technologies = basicInfo.technologies.map { it.substringAfterLast('.') },
                rawData = buildRawDataMap(ndefType, records),
                records = records,
                isWritable = isWritable,
                maxSize = maxSize,
                usedSize = usedSize
            )

            SecureLogger.d(TAG, "Successfully read ${records.size} NDEF records")
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
                if (ndef.isConnected) ndef.close()
            } catch (e: IOException) {
                SecureLogger.w(TAG, "Error closing Ndef: ${e.message}")
            }
        }
    }

    /**
     * Parse NDEF message into records.
     */
    private fun parseNdefMessage(message: NdefMessage): List<NdefRecord> {
        return message.records.map { record ->
            NdefRecord(
                tnf = record.tnf,
                type = record.type,
                id = record.id,
                payload = record.payload,
                payloadAsString = parsePayload(record)
            )
        }
    }

    /**
     * Parse record payload to human-readable string.
     */
    private fun parsePayload(record: android.nfc.NdefRecord): String? {
        return when (record.tnf) {
            TNF_WELL_KNOWN -> parseWellKnownRecord(record)
            TNF_MIME_MEDIA -> parseMimeRecord(record)
            TNF_ABSOLUTE_URI -> parseAbsoluteUri(record)
            TNF_EXTERNAL_TYPE -> parseExternalType(record)
            else -> null
        }
    }

    /**
     * Parse well-known record types (Text, URI, Smart Poster).
     */
    private fun parseWellKnownRecord(record: android.nfc.NdefRecord): String? {
        val type = record.type
        val payload = record.payload

        return when {
            type.contentEquals(android.nfc.NdefRecord.RTD_TEXT) -> parseTextRecord(payload)
            type.contentEquals(android.nfc.NdefRecord.RTD_URI) -> parseUriRecord(payload)
            type.contentEquals(android.nfc.NdefRecord.RTD_SMART_POSTER) -> parseSmartPoster(payload)
            else -> "Unknown well-known type: ${type.toHexString()}"
        }
    }

    /**
     * Parse NDEF Text Record.
     */
    private fun parseTextRecord(payload: ByteArray): String? {
        if (payload.isEmpty()) return null

        try {
            val statusByte = payload[0].toInt() and 0xFF
            val isUtf16 = (statusByte and 0x80) != 0
            val langCodeLength = statusByte and 0x3F

            val charset = if (isUtf16) Charsets.UTF_16 else Charsets.UTF_8
            val textStart = 1 + langCodeLength

            if (textStart >= payload.size) return null

            return String(payload, textStart, payload.size - textStart, charset)

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error parsing text record: ${e.message}")
            return null
        }
    }

    /**
     * Parse NDEF URI Record.
     */
    private fun parseUriRecord(payload: ByteArray): String? {
        if (payload.isEmpty()) return null

        try {
            val prefixCode = payload[0].toInt() and 0xFF
            val prefix = URI_PREFIXES.getOrElse(prefixCode) { "" }
            val suffix = String(payload, 1, payload.size - 1, Charsets.UTF_8)

            return prefix + suffix

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error parsing URI record: ${e.message}")
            return null
        }
    }

    /**
     * Parse Smart Poster record (contains nested NDEF message).
     */
    private fun parseSmartPoster(payload: ByteArray): String? {
        return try {
            val nestedMessage = NdefMessage(payload)
            val texts = mutableListOf<String>()

            for (record in nestedMessage.records) {
                val parsed = parsePayload(record)
                if (parsed != null) texts.add(parsed)
            }

            texts.joinToString(" | ")

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error parsing smart poster: ${e.message}")
            null
        }
    }

    /**
     * Parse MIME type record.
     */
    private fun parseMimeRecord(record: android.nfc.NdefRecord): String? {
        val mimeType = String(record.type, Charsets.US_ASCII)
        val payload = record.payload

        return when {
            mimeType.startsWith("text/") -> String(payload, Charsets.UTF_8)
            mimeType == "application/json" -> String(payload, Charsets.UTF_8)
            else -> "[$mimeType] ${payload.toHexString()}"
        }
    }

    /**
     * Parse absolute URI record.
     */
    private fun parseAbsoluteUri(record: android.nfc.NdefRecord): String? {
        return try {
            String(record.type, Charsets.UTF_8)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse external type record.
     */
    private fun parseExternalType(record: android.nfc.NdefRecord): String? {
        val domain = String(record.type, Charsets.UTF_8)
        return "External: $domain"
    }

    /**
     * Build raw data map for debugging.
     */
    private fun buildRawDataMap(ndefType: String?, records: List<NdefRecord>): Map<String, Any> {
        return buildMap {
            ndefType?.let { put("ndefType", it) }
            put("recordCount", records.size)
            records.forEachIndexed { index, record ->
                put("record${index}_tnf", record.tnf)
                put("record${index}_type", record.type.toHexString())
                record.payloadAsString?.let { put("record${index}_text", it) }
            }
        }
    }

    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> {
        // NDEF doesn't require authentication
        return readCard(tag)
    }
}
