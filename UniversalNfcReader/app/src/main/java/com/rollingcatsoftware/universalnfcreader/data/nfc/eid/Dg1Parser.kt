package com.rollingcatsoftware.universalnfcreader.data.nfc.eid

import android.util.Log
import java.io.ByteArrayInputStream
import java.nio.charset.Charset

/**
 * Parser for DG1 (Data Group 1) from Turkish eID card.
 *
 * DG1 contains MRZ (Machine Readable Zone) information following ICAO 9303 standard.
 * The data is encoded in ASN.1 TLV (Tag-Length-Value) format.
 *
 * Structure (TD1 format - 3 lines, 30 characters each):
 * - Line 1: DocType(2) | Country(3) | DocNo(9) | Check(1) | Optional(15)
 * - Line 2: DOB(6) | Check(1) | Sex(1) | DOE(6) | Check(1) | Nationality(3) | Optional(11) | Check(1)
 * - Line 3: Names(30) - Format: LASTNAME<<FIRSTNAME
 */
object Dg1Parser {

    private const val TAG = "Dg1Parser"

    // ASN.1 Tags
    private const val TAG_DG1 = 0x61          // DG1 wrapper tag
    private const val TAG_MRZ = 0x5F1F        // MRZ data tag

    /**
     * Parsed personal data from MRZ.
     */
    data class PersonalData(
        val tckn: String,              // Turkish Citizenship Number (11 digits)
        val firstName: String,          // Given name(s)
        val lastName: String,           // Surname
        val birthDate: String,          // Date of birth (DD/MM/YYYY)
        val gender: String,             // Gender (M/F)
        val nationality: String,        // Nationality (typically "TUR")
        val documentNumber: String,     // Document number
        val expiryDate: String,         // Card expiry date (DD/MM/YYYY)
        val serialNumber: String        // Serial number of the card
    )

    /**
     * Parses DG1 data to extract personal information.
     *
     * @param dg1Data Raw DG1 data bytes
     * @return PersonalData object or null if parsing fails
     */
    fun parse(dg1Data: ByteArray): PersonalData? {
        return try {
            Log.d(TAG, "Parsing DG1 data (${dg1Data.size} bytes)")
            Log.d(TAG, "DG1 hex: ${toHexString(dg1Data.take(50).toByteArray())}...")

            val stream = ByteArrayInputStream(dg1Data)

            // Read outer tag (should be 0x61 for DG1)
            val tag = readTag(stream)
            if (tag != TAG_DG1) {
                Log.w(TAG, "Unexpected DG1 tag: 0x${tag.toString(16)}, expected 0x61")
            }

            // Read length
            val length = readLength(stream)
            Log.d(TAG, "DG1 content length: $length bytes")

            // Read the remaining data
            val contentData = ByteArray(stream.available())
            stream.read(contentData)

            // Try to parse MRZ format
            val mrzData = extractMrzData(contentData)
            if (mrzData != null) {
                return parseMrz(mrzData)
            }

            Log.e(TAG, "Failed to extract MRZ data")
            null

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse DG1 data", e)
            null
        }
    }

    /**
     * Extracts MRZ data from DG1 content.
     */
    private fun extractMrzData(data: ByteArray): String? {
        try {
            val stream = ByteArrayInputStream(data)

            // Look for MRZ tag (0x5F1F)
            while (stream.available() > 0) {
                val tag = readTag(stream)
                val length = readLength(stream)

                if (tag == TAG_MRZ) {
                    val mrzBytes = ByteArray(length)
                    stream.read(mrzBytes)
                    val mrz = String(mrzBytes, Charset.forName("UTF-8"))
                    Log.d(TAG, "Found MRZ data: $mrz")
                    return mrz
                } else {
                    // Skip this field
                    stream.skip(length.toLong())
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract MRZ data", e)
        }

        return null
    }

    /**
     * Parses MRZ (Machine Readable Zone) data.
     */
    private fun parseMrz(mrz: String): PersonalData? {
        try {
            // Remove any whitespace and split into lines
            val cleanMrz = mrz.replace(" ", "").replace("\r", "")
            var lines = cleanMrz.split("\n").filter { it.isNotEmpty() }

            Log.d(TAG, "MRZ raw length: ${cleanMrz.length}")
            Log.d(TAG, "MRZ lines from split: ${lines.size}")

            // If we got a single line of 90 chars, it's TD1 format without line breaks
            if (lines.size == 1 && lines[0].length >= 90) {
                val singleLine = lines[0]
                Log.d(
                    TAG,
                    "Single line MRZ detected (${singleLine.length} chars), splitting into 3 lines of 30"
                )
                lines = listOf(
                    singleLine.substring(0, 30),
                    singleLine.substring(30, 60),
                    singleLine.substring(60, minOf(90, singleLine.length))
                )
            }

            lines.forEachIndexed { index, line ->
                Log.d(TAG, "Line $index: $line (${line.length} chars)")
            }

            if (lines.size < 3) {
                Log.e(TAG, "Invalid MRZ: expected 3 lines, got ${lines.size}")
                return null
            }

            val line1 = lines[0].padEnd(30, '<')
            val line2 = lines[1].padEnd(30, '<')
            val line3 = lines[2].padEnd(30, '<')

            // Parse Line 1
            val documentNumber = line1.substring(5, 14).replace("<", "").trim()

            // Parse Line 2
            val birthDate = line2.substring(0, 6)
            val gender = line2.substring(7, 8).replace("<", "M")
            val expiryDate = line2.substring(8, 14)
            val nationality = line2.substring(15, 18).replace("<", "").trim()

            // Parse Line 3 (names)
            val names = line3.replace("<", " ").trim().split("  ").filter { it.isNotEmpty() }
            val lastName = if (names.isNotEmpty()) names[0].trim() else ""
            val firstName = if (names.size > 1) names.drop(1).joinToString(" ").trim() else ""

            // Try to extract TCKN from optional data in line 1 or line 2
            val optional1 = line1.substring(15, 30).replace("<", "").trim()
            val optional2 = line2.substring(18, 29).replace("<", "").trim()

            // TCKN is 11 digits, look for it in optional fields
            val tckn = extractTckn(optional1) ?: extractTckn(optional2) ?: "Unknown"

            // Format dates from YYMMDD to readable format
            val formattedBirthDate = formatDate(birthDate)
            val formattedExpiryDate = formatDate(expiryDate)

            return PersonalData(
                tckn = tckn,
                firstName = firstName,
                lastName = lastName,
                birthDate = formattedBirthDate,
                gender = gender,
                nationality = nationality,
                documentNumber = documentNumber,
                expiryDate = formattedExpiryDate,
                serialNumber = documentNumber
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse MRZ", e)
            return null
        }
    }

    /**
     * Extracts TCKN (11-digit Turkish ID number) from a string.
     */
    private fun extractTckn(data: String): String? {
        val digits = data.filter { it.isDigit() }
        return if (digits.length == 11) digits else null
    }

    /**
     * Formats date from YYMMDD to DD/MM/YYYY.
     */
    private fun formatDate(yymmdd: String): String {
        if (yymmdd.length != 6 || !yymmdd.all { it.isDigit() }) {
            return yymmdd
        }

        val yy = yymmdd.substring(0, 2).toIntOrNull() ?: return yymmdd
        val mm = yymmdd.substring(2, 4)
        val dd = yymmdd.substring(4, 6)

        // Determine century (assume < 50 is 2000s, >= 50 is 1900s)
        val yyyy = if (yy < 50) 2000 + yy else 1900 + yy

        return "$dd/$mm/$yyyy"
    }

    /**
     * Reads an ASN.1 tag from the stream.
     */
    private fun readTag(stream: ByteArrayInputStream): Int {
        val firstByte = stream.read()
        if (firstByte == -1) throw IllegalStateException("Unexpected end of stream")

        // Check if this is a multi-byte tag
        return if ((firstByte and 0x1F) == 0x1F) {
            // Multi-byte tag
            var tag = firstByte shl 8
            tag = tag or stream.read()
            tag
        } else {
            // Single byte tag
            firstByte
        }
    }

    /**
     * Reads an ASN.1 length from the stream.
     */
    private fun readLength(stream: ByteArrayInputStream): Int {
        val firstByte = stream.read()
        if (firstByte == -1) throw IllegalStateException("Unexpected end of stream")

        return if ((firstByte and 0x80) == 0) {
            // Short form: length is in the first byte
            firstByte
        } else {
            // Long form: first byte indicates how many bytes encode the length
            val numBytes = firstByte and 0x7F
            var length = 0
            repeat(numBytes) {
                length = (length shl 8) or stream.read()
            }
            length
        }
    }

    /**
     * Converts byte array to hex string for debugging.
     */
    private fun toHexString(bytes: ByteArray): String {
        return bytes.joinToString(" ") { String.format("%02X", it) }
    }
}
