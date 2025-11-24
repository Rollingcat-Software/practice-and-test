package com.turkey.eidnfc.util

import android.content.Context
import android.nfc.NfcAdapter
import com.turkey.eidnfc.util.Constants.Nfc.PIN_LENGTH
import com.turkey.eidnfc.util.Constants.TurkishEid.TCKN_LENGTH

/**
 * Extension functions for various types used throughout the app.
 */

// ============================================================================
// ByteArray Extensions
// ============================================================================

/**
 * Converts byte array to hex string for logging/display.
 * Example: [0x01, 0x02, 0xAB] -> "01 02 AB"
 */
fun ByteArray.toHexString(): String {
    return joinToString(" ") { String.format("%02X", it) }
}

/**
 * Converts byte array to integer.
 * Assumes big-endian byte order.
 */
fun ByteArray.toInt(): Int {
    require(size <= 4) { "ByteArray too large to convert to Int (max 4 bytes)" }
    var result = 0
    for (byte in this) {
        result = (result shl 8) or (byte.toInt() and 0xFF)
    }
    return result
}

// ============================================================================
// String Extensions
// ============================================================================

/**
 * Checks if string is a valid MRZ data format.
 * Format: documentNumber|dateOfBirth|dateOfExpiry
 * Example: A12345678|900115|301231
 */
fun String.isValidPin(): Boolean {
    val parts = split("|")
    if (parts.size != 3) return false

    val (docNo, dob, doe) = parts

    return docNo.isNotEmpty() &&
            docNo.length <= 9 &&
            docNo.all { it.isLetterOrDigit() } &&
            dob.length == 6 &&
            dob.all { it.isDigit() } &&
            doe.length == 6 &&
            doe.all { it.isDigit() }
}

/**
 * Checks if string is a valid MRZ data format (alias for isValidPin).
 */
fun String.isValidMrzData(): Boolean = isValidPin()

/**
 * Checks if string is a valid Turkish Citizenship Number (TCKN).
 *
 * TCKN Validation Rules:
 * 1. Must be 11 digits
 * 2. First digit cannot be 0
 * 3. 10th digit = ((sum of odd positions * 7) - (sum of even positions)) mod 10
 * 4. 11th digit = (sum of first 10 digits) mod 10
 */
fun String.isValidTckn(): Boolean {
    // Check length and all digits
    if (length != TCKN_LENGTH || !all { it.isDigit() }) return false

    val digits = map { it.toString().toInt() }

    // First digit cannot be 0
    if (digits[0] == 0) return false

    // Calculate 10th digit check
    val oddSum = digits[0] + digits[2] + digits[4] + digits[6] + digits[8]
    val evenSum = digits[1] + digits[3] + digits[5] + digits[7]
    val check10 = ((oddSum * 7) - evenSum) % 10

    if (check10 != digits[9]) return false

    // Calculate 11th digit check
    val sumAll = digits.take(10).sum()
    val check11 = sumAll % 10

    return check11 == digits[10]
}

/**
 * Masks a string for display (replaces characters with bullets).
 * Example: "123456" -> "••••••"
 */
fun String.mask(maskChar: Char = '•'): String {
    return maskChar.toString().repeat(length)
}

/**
 * Capitalizes first letter of each word.
 * Example: "mehmet ali" -> "Mehmet Ali"
 */
fun String.capitalizeWords(): String {
    return split(" ").joinToString(" ") { word ->
        word.lowercase().replaceFirstChar { it.uppercase() }
    }
}

// ============================================================================
// Context Extensions
// ============================================================================

/**
 * Checks if device has NFC hardware.
 */
fun Context.hasNfc(): Boolean {
    return NfcAdapter.getDefaultAdapter(this) != null
}

/**
 * Checks if NFC is currently enabled.
 */
fun Context.isNfcEnabled(): Boolean {
    return NfcAdapter.getDefaultAdapter(this)?.isEnabled == true
}

// ============================================================================
// Number Extensions
// ============================================================================

/**
 * Converts milliseconds to seconds.
 */
fun Long.toSeconds(): Long {
    return this / 1000
}

/**
 * Converts seconds to milliseconds.
 */
fun Int.toMillis(): Long {
    return this * 1000L
}

// ============================================================================
// Collection Extensions
// ============================================================================

/**
 * Safely gets element at index or returns null.
 */
fun <T> List<T>.getOrNull(index: Int): T? {
    return if (index in indices) this[index] else null
}
