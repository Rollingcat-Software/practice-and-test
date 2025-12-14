package com.rollingcatsoftware.universalnfcreader.data.nfc.security

import android.util.Log
import com.rollingcatsoftware.universalnfcreader.BuildConfig

/**
 * Secure logging wrapper that automatically redacts sensitive data.
 *
 * COMPLIANCE: se-checklist.md
 * - Section 5.2: "Never expose sensitive data in error messages"
 * - Section 5.2: "Log errors securely (no PII in logs)"
 *
 * This class provides:
 * 1. Automatic redaction of known sensitive patterns (TCKN, passport numbers, dates)
 * 2. Different logging behavior for debug vs release builds
 * 3. Truncation of potentially sensitive long data
 * 4. Hex data masking for cryptographic material
 *
 * Usage:
 * ```kotlin
 * SecureLogger.d(TAG, "Processing document: ${documentNumber}") // Auto-redacted
 * SecureLogger.logHex(TAG, "Key material", keyBytes) // Masked in release
 * ```
 *
 * Security Notes:
 * - In release builds, sensitive data is always redacted
 * - In debug builds, more detail is shown but still with some redaction
 * - Never log full cryptographic keys even in debug mode
 */
object SecureLogger {

    /**
     * Whether verbose logging is enabled.
     * In release builds, this is always false.
     */
    private val isVerboseLoggingEnabled: Boolean
        get() = BuildConfig.DEBUG

    /**
     * Whether to show any potentially sensitive data.
     * Always false in release builds.
     */
    private val showSensitiveData: Boolean
        get() = BuildConfig.DEBUG

    // Patterns for sensitive data detection
    private val TCKN_PATTERN = Regex("""\b\d{11}\b""") // Turkish ID number (11 digits)
    private val PASSPORT_PATTERN = Regex("""\b[A-Z]{1,2}\d{6,9}\b""") // Passport numbers
    private val DATE_YYMMDD_PATTERN = Regex("""\b\d{6}\b""") // Dates in YYMMDD format
    private val MRZ_LINE_PATTERN = Regex("""[A-Z0-9<]{20,44}""") // MRZ lines
    private val HEX_PATTERN = Regex("""([0-9A-Fa-f]{2}\s*){8,}""") // Hex strings (8+ bytes)

    /**
     * Log a debug message with automatic PII redaction.
     */
    fun d(tag: String, message: String) {
        if (isVerboseLoggingEnabled) {
            Log.d(tag, redactSensitiveData(message))
        }
    }

    /**
     * Log an info message with automatic PII redaction.
     */
    fun i(tag: String, message: String) {
        Log.i(tag, redactSensitiveData(message))
    }

    /**
     * Log a warning message with automatic PII redaction.
     */
    fun w(tag: String, message: String) {
        Log.w(tag, redactSensitiveData(message))
    }

    /**
     * Log a warning message with exception.
     */
    fun w(tag: String, message: String, throwable: Throwable) {
        Log.w(tag, redactSensitiveData(message), throwable)
    }

    /**
     * Log an error message with automatic PII redaction.
     */
    fun e(tag: String, message: String) {
        Log.e(tag, redactSensitiveData(message))
    }

    /**
     * Log an error message with exception.
     */
    fun e(tag: String, message: String, throwable: Throwable) {
        Log.e(tag, redactSensitiveData(message), throwable)
    }

    /**
     * Log a verbose message (only in debug builds).
     */
    fun v(tag: String, message: String) {
        if (isVerboseLoggingEnabled) {
            Log.v(tag, redactSensitiveData(message))
        }
    }

    /**
     * Log hex data with appropriate masking.
     *
     * In debug builds: Shows first and last 4 bytes, masks middle
     * In release builds: Shows only length
     *
     * @param tag Log tag
     * @param label Description of the data
     * @param data The byte array to log
     */
    fun logHex(tag: String, label: String, data: ByteArray?) {
        if (data == null) {
            d(tag, "$label: null")
            return
        }

        if (!isVerboseLoggingEnabled) {
            // Release: only log length
            Log.d(tag, "$label: [${data.size} bytes]")
            return
        }

        // Debug: show partial data
        val hex = if (data.size <= 8) {
            // Short data: mask entirely
            "[${data.size} bytes]"
        } else {
            // Show first 4 and last 4 bytes
            val first = data.take(4).joinToString("") { "%02X".format(it) }
            val last = data.takeLast(4).joinToString("") { "%02X".format(it) }
            "$first...[${data.size - 8} masked]...$last"
        }

        Log.d(tag, "$label: $hex")
    }

    /**
     * Log hex data from a SecureByteArray with appropriate masking.
     */
    fun logHex(tag: String, label: String, data: SecureByteArray?) {
        if (data == null || data.closed) {
            d(tag, "$label: null/closed")
            return
        }

        data.withData { bytes ->
            logHex(tag, label, bytes)
        }
    }

    /**
     * Log an APDU command with masking of sensitive data.
     *
     * @param tag Log tag
     * @param command The APDU command bytes
     */
    fun logApduCommand(tag: String, command: ByteArray) {
        if (!isVerboseLoggingEnabled) {
            Log.d(tag, "APDU Command: [${command.size} bytes]")
            return
        }

        // Show command header (CLA INS P1 P2) and mask the rest
        if (command.size >= 4) {
            val header = command.take(4).joinToString(" ") { "%02X".format(it) }
            val dataInfo = if (command.size > 5) {
                val lc = command[4].toInt() and 0xFF
                " Lc=$lc [data masked]"
            } else {
                ""
            }
            Log.d(tag, "APDU Command: $header$dataInfo")
        } else {
            Log.d(tag, "APDU Command: [${command.size} bytes - invalid]")
        }
    }

    /**
     * Log an APDU response with masking of sensitive data.
     *
     * @param tag Log tag
     * @param response The APDU response bytes
     */
    fun logApduResponse(tag: String, response: ByteArray) {
        if (!isVerboseLoggingEnabled) {
            Log.d(tag, "APDU Response: [${response.size} bytes]")
            return
        }

        // Show only status word and data length
        if (response.size >= 2) {
            val sw1 = response[response.size - 2].toInt() and 0xFF
            val sw2 = response[response.size - 1].toInt() and 0xFF
            val dataLen = response.size - 2
            Log.d(tag, "APDU Response: SW=%02X%02X Data=[%d bytes]".format(sw1, sw2, dataLen))
        } else {
            Log.d(tag, "APDU Response: [${response.size} bytes - invalid]")
        }
    }

    /**
     * Log MRZ data with appropriate masking.
     *
     * @param tag Log tag
     * @param documentNumber The document number
     * @param dateOfBirth Date of birth (YYMMDD)
     * @param dateOfExpiry Date of expiry (YYMMDD)
     */
    fun logMrzData(
        tag: String,
        documentNumber: String,
        dateOfBirth: String,
        dateOfExpiry: String
    ) {
        if (!isVerboseLoggingEnabled) {
            Log.d(tag, "MRZ Data: [redacted]")
            return
        }

        // Show partial document number, mask dates
        val maskedDocNum = maskDocumentNumber(documentNumber)
        Log.d(tag, "MRZ Data: DocNum=$maskedDocNum DOB=****** DOE=******")
    }

    /**
     * Redacts sensitive data from a message.
     *
     * @param message The message to redact
     * @return The redacted message
     */
    fun redactSensitiveData(message: String): String {
        var result = message

        // Always redact TCKN (11-digit numbers)
        result = TCKN_PATTERN.replace(result) { match ->
            val value = match.value
            "${value.take(3)}*****${value.takeLast(2)}"
        }

        // Redact passport-like numbers
        result = PASSPORT_PATTERN.replace(result) { match ->
            val value = match.value
            if (value.length > 4) {
                "${value.take(2)}***${value.takeLast(2)}"
            } else {
                "****"
            }
        }

        // In release builds, also redact dates and hex strings
        if (!showSensitiveData) {
            result = DATE_YYMMDD_PATTERN.replace(result) { "******" }
            result = MRZ_LINE_PATTERN.replace(result) { "[MRZ REDACTED]" }
            result = HEX_PATTERN.replace(result) { "[HEX REDACTED]" }
        }

        return result
    }

    /**
     * Masks a document number for logging.
     * Shows first 2 and last 2 characters.
     */
    fun maskDocumentNumber(documentNumber: String): String {
        return when {
            documentNumber.length <= 4 -> "****"
            else -> "${documentNumber.take(2)}***${documentNumber.takeLast(2)}"
        }
    }

    /**
     * Masks a name for logging.
     * Shows first letter and length.
     */
    fun maskName(name: String): String {
        return when {
            name.isEmpty() -> "[empty]"
            name.length == 1 -> "*"
            else -> "${name.first()}***[${name.length}]"
        }
    }

    /**
     * Creates a safe exception message without sensitive data.
     *
     * @param operation The operation that failed
     * @param errorCode Optional error code
     * @param hint Optional hint for the user
     * @return A safe error message
     */
    fun safeErrorMessage(
        operation: String,
        errorCode: String? = null,
        hint: String? = null
    ): String {
        return buildString {
            append("$operation failed")
            errorCode?.let { append(" (code: $it)") }
            hint?.let { append(". $it") }
        }
    }
}

/**
 * Extension function for easy secure logging from any class.
 */
fun Any.secureLog(message: String) {
    SecureLogger.d(this::class.java.simpleName, message)
}

/**
 * Extension function to get a masked string representation of a ByteArray.
 */
fun ByteArray.toMaskedHexString(): String {
    return if (size <= 8) {
        "[${size} bytes]"
    } else {
        val first = take(4).joinToString("") { "%02X".format(it) }
        val last = takeLast(4).joinToString("") { "%02X".format(it) }
        "$first...$last"
    }
}
