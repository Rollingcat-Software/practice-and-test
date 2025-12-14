package com.rollingcatsoftware.universalnfcreader.data.nfc.security

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for SecureLogger PII redaction functionality.
 *
 * Note: These tests focus on the redaction logic, not actual logging output
 * which depends on Android Log class.
 */
class SecureLoggerTest {

    @Test
    fun `redactSensitiveData masks 11-digit TCKN`() {
        val message = "User TCKN: 12345678901"

        val redacted = SecureLogger.redactSensitiveData(message)

        assertEquals("User TCKN: 123*****01", redacted)
    }

    @Test
    fun `redactSensitiveData masks multiple TCKNs`() {
        val message = "TCKNs: 12345678901 and 98765432109"

        val redacted = SecureLogger.redactSensitiveData(message)

        assertTrue(redacted.contains("123*****01"))
        assertTrue(redacted.contains("987*****09"))
    }

    @Test
    fun `redactSensitiveData masks passport numbers`() {
        val message = "Passport: A12345678"

        val redacted = SecureLogger.redactSensitiveData(message)

        assertEquals("Passport: A1***78", redacted)
    }

    @Test
    fun `redactSensitiveData masks two-letter prefix passport numbers`() {
        val message = "Passport: AB1234567"

        val redacted = SecureLogger.redactSensitiveData(message)

        assertEquals("Passport: AB***67", redacted)
    }

    @Test
    fun `redactSensitiveData preserves non-sensitive text`() {
        val message = "Processing completed successfully"

        val redacted = SecureLogger.redactSensitiveData(message)

        assertEquals(message, redacted)
    }

    @Test
    fun `redactSensitiveData handles empty string`() {
        val redacted = SecureLogger.redactSensitiveData("")

        assertEquals("", redacted)
    }

    @Test
    fun `redactSensitiveData handles mixed content`() {
        val message = "User 12345678901 has passport A12345678"

        val redacted = SecureLogger.redactSensitiveData(message)

        assertTrue(redacted.contains("123*****01"))
        assertTrue(redacted.contains("A1***78"))
    }

    @Test
    fun `maskDocumentNumber masks long numbers`() {
        val masked = SecureLogger.maskDocumentNumber("A12345678")

        assertEquals("A1***78", masked)
    }

    @Test
    fun `maskDocumentNumber returns stars for short numbers`() {
        val masked = SecureLogger.maskDocumentNumber("AB12")

        assertEquals("****", masked)
    }

    @Test
    fun `maskDocumentNumber handles empty string`() {
        val masked = SecureLogger.maskDocumentNumber("")

        assertEquals("****", masked)
    }

    @Test
    fun `maskName shows first letter and length`() {
        val masked = SecureLogger.maskName("Mehmet")

        assertEquals("M***[6]", masked)
    }

    @Test
    fun `maskName handles single character`() {
        val masked = SecureLogger.maskName("A")

        assertEquals("*", masked)
    }

    @Test
    fun `maskName handles empty string`() {
        val masked = SecureLogger.maskName("")

        assertEquals("[empty]", masked)
    }

    @Test
    fun `safeErrorMessage creates basic message`() {
        val message = SecureLogger.safeErrorMessage("Authentication")

        assertEquals("Authentication failed", message)
    }

    @Test
    fun `safeErrorMessage includes error code`() {
        val message = SecureLogger.safeErrorMessage("Authentication", errorCode = "6982")

        assertEquals("Authentication failed (code: 6982)", message)
    }

    @Test
    fun `safeErrorMessage includes hint`() {
        val message = SecureLogger.safeErrorMessage(
            "Authentication",
            hint = "Check MRZ data"
        )

        assertEquals("Authentication failed. Check MRZ data", message)
    }

    @Test
    fun `safeErrorMessage includes both code and hint`() {
        val message = SecureLogger.safeErrorMessage(
            "Authentication",
            errorCode = "6982",
            hint = "Check MRZ data"
        )

        assertEquals("Authentication failed (code: 6982). Check MRZ data", message)
    }

    @Test
    fun `toMaskedHexString masks long arrays`() {
        val data = byteArrayOf(
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C
        )

        val masked = data.toMaskedHexString()

        assertTrue(masked.contains("01020304"))  // First 4 bytes shown
        assertTrue(masked.contains("090A0B0C"))  // Last 4 bytes shown
        assertTrue(masked.contains("..."))       // Middle masked
    }

    @Test
    fun `toMaskedHexString shows only length for short arrays`() {
        val data = byteArrayOf(0x01, 0x02, 0x03, 0x04)

        val masked = data.toMaskedHexString()

        assertEquals("[4 bytes]", masked)
    }

    @Test
    fun `toMaskedHexString handles exactly 8 bytes`() {
        val data = byteArrayOf(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)

        val masked = data.toMaskedHexString()

        assertEquals("[8 bytes]", masked)
    }

    @Test
    fun `toMaskedHexString handles empty array`() {
        val data = byteArrayOf()

        val masked = data.toMaskedHexString()

        assertEquals("[0 bytes]", masked)
    }
}
