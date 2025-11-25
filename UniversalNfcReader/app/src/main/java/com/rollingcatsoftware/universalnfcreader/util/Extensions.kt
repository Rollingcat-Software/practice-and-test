package com.rollingcatsoftware.universalnfcreader.util

/**
 * Extension functions and utilities for NFC data manipulation.
 */

/**
 * Convert ByteArray to hexadecimal string.
 */
fun ByteArray.toHexString(): String = joinToString("") { "%02X".format(it) }

/**
 * Convert ByteArray to hexadecimal string with spaces between bytes.
 */
fun ByteArray.toHexStringWithSpaces(): String = joinToString(" ") { "%02X".format(it) }

/**
 * Convert hexadecimal string to ByteArray.
 */
fun String.hexToByteArray(): ByteArray {
    val cleanHex = this.replace(" ", "").replace(":", "")
    require(cleanHex.length % 2 == 0) { "Hex string must have even length" }
    return cleanHex.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
}

/**
 * Get status word (SW1-SW2) from APDU response.
 * Returns the last two bytes as an integer.
 */
fun ByteArray.getStatusWord(): Int {
    require(size >= 2) { "Response must be at least 2 bytes" }
    return ((this[size - 2].toInt() and 0xFF) shl 8) or (this[size - 1].toInt() and 0xFF)
}

/**
 * Check if APDU response indicates success (SW = 0x9000).
 */
fun ByteArray.isSuccess(): Boolean = getStatusWord() == 0x9000

/**
 * Get response data without status word.
 */
fun ByteArray.getResponseData(): ByteArray {
    require(size >= 2) { "Response must be at least 2 bytes" }
    return copyOfRange(0, size - 2)
}

/**
 * Extract remaining bytes count from status word 0x61XX.
 * Returns null if status word is not 0x61XX.
 */
fun ByteArray.getRemainingBytes(): Int? {
    val sw = getStatusWord()
    return if ((sw and 0xFF00) == 0x6100) sw and 0x00FF else null
}

/**
 * Reverse a byte array (useful for UID display).
 */
fun ByteArray.reversed(): ByteArray = this.reversedArray()

/**
 * XOR two byte arrays of equal length.
 */
infix fun ByteArray.xor(other: ByteArray): ByteArray {
    require(size == other.size) { "Arrays must have equal length" }
    return ByteArray(size) { i -> (this[i].toInt() xor other[i].toInt()).toByte() }
}

/**
 * Pad byte array to specified length with given byte.
 */
fun ByteArray.padTo(length: Int, padByte: Byte = 0x00): ByteArray {
    if (size >= length) return this
    return this + ByteArray(length - size) { padByte }
}

/**
 * Pad byte array on the left to specified length.
 */
fun ByteArray.padStart(length: Int, padByte: Byte = 0x00): ByteArray {
    if (size >= length) return this
    return ByteArray(length - size) { padByte } + this
}

/**
 * Convert single byte to unsigned int.
 */
fun Byte.toUnsignedInt(): Int = this.toInt() and 0xFF

/**
 * Convert two bytes to unsigned int (big endian).
 */
fun twoByteToInt(high: Byte, low: Byte): Int =
    ((high.toInt() and 0xFF) shl 8) or (low.toInt() and 0xFF)

/**
 * Convert four bytes to unsigned int (big endian).
 */
fun fourBytesToInt(bytes: ByteArray, offset: Int = 0): Int {
    require(bytes.size >= offset + 4) { "Not enough bytes" }
    return ((bytes[offset].toInt() and 0xFF) shl 24) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 8) or
            (bytes[offset + 3].toInt() and 0xFF)
}

/**
 * Convert int to 4-byte array (big endian).
 */
fun Int.toByteArray(): ByteArray = byteArrayOf(
    (this shr 24).toByte(),
    (this shr 16).toByte(),
    (this shr 8).toByte(),
    this.toByte()
)

/**
 * Safely get a range from ByteArray, returning null if out of bounds.
 */
fun ByteArray.safeSlice(from: Int, length: Int): ByteArray? {
    if (from < 0 || from + length > size) return null
    return copyOfRange(from, from + length)
}

/**
 * Format UID for display (with colons).
 */
fun ByteArray.formatAsUid(): String = joinToString(":") { "%02X".format(it) }

/**
 * Calculate check digit for MRZ data (MOD 10 checksum).
 */
fun String.calculateMrzCheckDigit(): Int {
    val weights = intArrayOf(7, 3, 1)
    var sum = 0
    forEachIndexed { index, char ->
        val value = when {
            char.isDigit() -> char.digitToInt()
            char.isLetter() -> char.uppercaseChar().code - 'A'.code + 10
            char == '<' -> 0
            else -> 0
        }
        sum += value * weights[index % 3]
    }
    return sum % 10
}

/**
 * Validate MRZ check digit.
 */
fun String.validateMrzCheckDigit(expectedDigit: Int): Boolean =
    calculateMrzCheckDigit() == expectedDigit

/**
 * Clean MRZ string (replace < with space, trim).
 */
fun String.cleanMrz(): String = replace('<', ' ').trim()

/**
 * Format date from YYMMDD to human-readable format.
 */
fun String.formatMrzDate(): String {
    if (length != 6) return this
    val year = substring(0, 2).toIntOrNull() ?: return this
    val month = substring(2, 4)
    val day = substring(4, 6)
    // Assume 20xx for years 00-30, 19xx for years 31-99
    val fullYear = if (year <= 30) 2000 + year else 1900 + year
    return "$day/$month/$fullYear"
}
