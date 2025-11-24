package com.turkey.eidnfc.data.nfc

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for ApduHelper.
 *
 * Tests APDU command construction, response parsing, and utility functions.
 */
class ApduHelperTest {

    @Test
    fun `selectCommand with AID creates correct APDU`() {
        // Given
        val aid = byteArrayOf(0xA0.toByte(), 0x00, 0x00, 0x01, 0x67)

        // When
        val command = ApduHelper.selectCommand(aid, isAid = true)

        // Then
        assertEquals(0x00, command[0]) // CLA
        assertEquals(0xA4.toByte(), command[1]) // INS
        assertEquals(0x04, command[2]) // P1 for AID
        assertEquals(0x0C, command[3]) // P2
        assertEquals(aid.size.toByte(), command[4]) // Lc
        assertEquals(0x00, command[command.size - 1]) // Le

        // Check AID is embedded
        for (i in aid.indices) {
            assertEquals(aid[i], command[5 + i])
        }
    }

    @Test
    fun `selectCommand with file creates correct APDU`() {
        // Given
        val fileId = byteArrayOf(0x01, 0x01)

        // When
        val command = ApduHelper.selectCommand(fileId, isAid = false)

        // Then
        assertEquals(0x00, command[0]) // CLA
        assertEquals(0xA4.toByte(), command[1]) // INS
        assertEquals(0x02, command[2]) // P1 for file
        assertEquals(0x0C, command[3]) // P2
        assertEquals(fileId.size.toByte(), command[4]) // Lc

        // Check file ID is embedded
        assertEquals(fileId[0], command[5])
        assertEquals(fileId[1], command[6])
    }

    @Test
    fun `selectEidAid creates correct Turkish eID AID command`() {
        // When
        val command = ApduHelper.selectEidAid()

        // Then
        assertEquals(0x00, command[0]) // CLA
        assertEquals(0xA4.toByte(), command[1]) // INS
        assertEquals(0x04, command[2]) // P1 for AID
        assertEquals(0x0C, command[3]) // P2
        assertEquals(0x09, command[4]) // Lc (AID is 9 bytes)

        // Check Turkish eID AID is embedded
        val expectedAid = ApduHelper.TURKISH_EID_AID
        for (i in expectedAid.indices) {
            assertEquals(expectedAid[i], command[5 + i])
        }
    }

    @Test
    fun `readBinaryCommand with default parameters creates correct APDU`() {
        // When
        val command = ApduHelper.readBinaryCommand()

        // Then
        assertEquals(5, command.size)
        assertEquals(0x00, command[0]) // CLA
        assertEquals(0xB0.toByte(), command[1]) // INS
        assertEquals(0x00, command[2]) // P1 (offset high)
        assertEquals(0x00, command[3]) // P2 (offset low)
        assertEquals(0x00, command[4]) // Le (0 means max)
    }

    @Test
    fun `readBinaryCommand with offset creates correct APDU`() {
        // When
        val command = ApduHelper.readBinaryCommand(offset = 256, length = 128)

        // Then
        assertEquals(5, command.size)
        assertEquals(0x00, command[0]) // CLA
        assertEquals(0xB0.toByte(), command[1]) // INS
        assertEquals(0x01, command[2]) // P1 (offset high = 256 >> 8)
        assertEquals(0x00, command[3]) // P2 (offset low = 256 & 0xFF)
        assertEquals(128.toByte(), command[4]) // Le
    }

    @Test
    fun `readBinaryCommand with large offset creates correct APDU`() {
        // When
        val command = ApduHelper.readBinaryCommand(offset = 512, length = 64)

        // Then
        assertEquals(0x02, command[2]) // P1 (512 >> 8 = 2)
        assertEquals(0x00, command[3]) // P2 (512 & 0xFF = 0)
        assertEquals(64.toByte(), command[4]) // Le
    }

    @Test
    fun `verifyPinCommand with valid PIN creates correct APDU`() {
        // Given
        val pin = "123456"

        // When
        val command = ApduHelper.verifyPinCommand(pin)

        // Then
        assertNotNull(command)
        command!!
        assertEquals(11, command.size) // Header (5) + PIN (6)
        assertEquals(0x00, command[0]) // CLA
        assertEquals(0x20, command[1]) // INS
        assertEquals(0x00, command[2]) // P1
        assertEquals(0x81.toByte(), command[3]) // P2 (PIN1 reference)
        assertEquals(0x06, command[4]) // Lc (PIN length)

        // Check PIN is embedded correctly
        val expectedPinBytes = pin.toByteArray(Charsets.UTF_8)
        for (i in expectedPinBytes.indices) {
            assertEquals(expectedPinBytes[i], command[5 + i])
        }
    }

    @Test
    fun `verifyPinCommand with invalid PIN returns null`() {
        // Test various invalid PINs
        assertNull(ApduHelper.verifyPinCommand("12345"))    // Too short
        assertNull(ApduHelper.verifyPinCommand("1234567"))  // Too long
        assertNull(ApduHelper.verifyPinCommand("12345a"))   // Contains letter
        assertNull(ApduHelper.verifyPinCommand(""))         // Empty
        assertNull(ApduHelper.verifyPinCommand("abc123"))   // Contains letters
    }

    @Test
    fun `isValidPin returns true for valid 6-digit PIN`() {
        assertTrue(ApduHelper.isValidPin("123456"))
        assertTrue(ApduHelper.isValidPin("000000"))
        assertTrue(ApduHelper.isValidPin("999999"))
    }

    @Test
    fun `isValidPin returns false for invalid PIN`() {
        assertFalse(ApduHelper.isValidPin("12345"))     // Too short
        assertFalse(ApduHelper.isValidPin("1234567"))   // Too long
        assertFalse(ApduHelper.isValidPin("12345a"))    // Contains letter
        assertFalse(ApduHelper.isValidPin(""))          // Empty
        assertFalse(ApduHelper.isValidPin("abcdef"))    // All letters
        assertFalse(ApduHelper.isValidPin("12 34 56"))  // Contains spaces
    }

    @Test
    fun `parseResponse extracts data and status word correctly`() {
        // Given - Response with data
        val response = byteArrayOf(0x01, 0x02, 0x03, 0x90.toByte(), 0x00)

        // When
        val (data, statusWord) = ApduHelper.parseResponse(response)

        // Then
        assertArrayEquals(byteArrayOf(0x01, 0x02, 0x03), data)
        assertEquals(0x9000, statusWord)
    }

    @Test
    fun `parseResponse handles response with no data`() {
        // Given - Response with only status word
        val response = byteArrayOf(0x90.toByte(), 0x00)

        // When
        val (data, statusWord) = ApduHelper.parseResponse(response)

        // Then
        assertEquals(0, data.size)
        assertEquals(0x9000, statusWord)
    }

    @Test
    fun `parseResponse handles error status word`() {
        // Given - Response with error status
        val response = byteArrayOf(0x6A.toByte(), 0x82.toByte())

        // When
        val (data, statusWord) = ApduHelper.parseResponse(response)

        // Then
        assertEquals(0, data.size)
        assertEquals(0x6A82, statusWord)
    }

    @Test
    fun `parseResponse handles short response`() {
        // Given - Response too short (less than 2 bytes)
        val response = byteArrayOf(0x90.toByte())

        // When
        val (data, statusWord) = ApduHelper.parseResponse(response)

        // Then
        assertEquals(0, data.size)
        assertEquals(0x6F00, statusWord) // General error
    }

    @Test
    fun `parseResponse handles empty response`() {
        // Given - Empty response
        val response = byteArrayOf()

        // When
        val (data, statusWord) = ApduHelper.parseResponse(response)

        // Then
        assertEquals(0, data.size)
        assertEquals(0x6F00, statusWord) // General error
    }

    @Test
    fun `isSuccess returns true for success status`() {
        assertTrue(ApduHelper.isSuccess(0x9000))
    }

    @Test
    fun `isSuccess returns false for error status`() {
        assertFalse(ApduHelper.isSuccess(0x6A82))
        assertFalse(ApduHelper.isSuccess(0x6982))
        assertFalse(ApduHelper.isSuccess(0x63C0))
        assertFalse(ApduHelper.isSuccess(0x6983))
    }

    @Test
    fun `getRemainingPinAttempts returns correct attempts for wrong PIN`() {
        // Test various wrong PIN status words
        assertEquals(3, ApduHelper.getRemainingPinAttempts(0x63C3))
        assertEquals(2, ApduHelper.getRemainingPinAttempts(0x63C2))
        assertEquals(1, ApduHelper.getRemainingPinAttempts(0x63C1))
        assertEquals(0, ApduHelper.getRemainingPinAttempts(0x63C0))
    }

    @Test
    fun `getRemainingPinAttempts returns -1 for non-PIN-error status`() {
        assertEquals(-1, ApduHelper.getRemainingPinAttempts(0x9000))
        assertEquals(-1, ApduHelper.getRemainingPinAttempts(0x6A82))
        assertEquals(-1, ApduHelper.getRemainingPinAttempts(0x6982))
        assertEquals(-1, ApduHelper.getRemainingPinAttempts(0x6983))
    }

    @Test
    fun `getStatusDescription returns correct message for success`() {
        val description = ApduHelper.getStatusDescription(0x9000)
        assertEquals("Success", description)
    }

    @Test
    fun `getStatusDescription returns correct message for wrong PIN`() {
        val description = ApduHelper.getStatusDescription(0x63C3)
        assertEquals("Wrong PIN. 3 attempt(s) remaining", description)
    }

    @Test
    fun `getStatusDescription returns correct message for security not satisfied`() {
        val description = ApduHelper.getStatusDescription(0x6982)
        assertEquals("Security condition not satisfied. PIN required", description)
    }

    @Test
    fun `getStatusDescription returns correct message for blocked`() {
        val description = ApduHelper.getStatusDescription(0x6983)
        assertEquals("Authentication method blocked. Card is locked", description)
    }

    @Test
    fun `getStatusDescription returns correct message for file not found`() {
        val description = ApduHelper.getStatusDescription(0x6A82)
        assertEquals("File or application not found", description)
    }

    @Test
    fun `getStatusDescription returns correct message for wrong parameters`() {
        val description = ApduHelper.getStatusDescription(0x6A86)
        assertEquals("Wrong parameters P1 or P2", description)
    }

    @Test
    fun `getStatusDescription returns correct message for wrong length`() {
        val description = ApduHelper.getStatusDescription(0x6700)
        assertEquals("Wrong length Lc", description)
    }

    @Test
    fun `getStatusDescription returns correct message for conditions not satisfied`() {
        val description = ApduHelper.getStatusDescription(0x6985)
        assertEquals("Conditions of use not satisfied", description)
    }

    @Test
    fun `getStatusDescription returns unknown for unrecognized status`() {
        val description = ApduHelper.getStatusDescription(0x1234)
        assertEquals("Unknown status: 0x1234", description)
    }

    @Test
    fun `toHexString converts byte array correctly`() {
        // Given
        val bytes = byteArrayOf(0x01, 0x02, 0x0A, 0xFF.toByte())

        // When
        val hexString = ApduHelper.toHexString(bytes)

        // Then
        assertEquals("01 02 0A FF", hexString)
    }

    @Test
    fun `toHexString handles empty array`() {
        val hexString = ApduHelper.toHexString(byteArrayOf())
        assertEquals("", hexString)
    }

    @Test
    fun `toHexString handles single byte`() {
        val hexString = ApduHelper.toHexString(byteArrayOf(0xAB.toByte()))
        assertEquals("AB", hexString)
    }

    @Test
    fun `FileIds constants are correct`() {
        // Verify Turkish eID file IDs
        assertArrayEquals(byteArrayOf(0x01, 0x1C), ApduHelper.FileIds.EF_CARD_ACCESS)
        assertArrayEquals(byteArrayOf(0x01, 0x1D), ApduHelper.FileIds.EF_SOD)
        assertArrayEquals(byteArrayOf(0x01, 0x01), ApduHelper.FileIds.DG1)
        assertArrayEquals(byteArrayOf(0x01, 0x02), ApduHelper.FileIds.DG2)
        assertArrayEquals(byteArrayOf(0x01, 0x03), ApduHelper.FileIds.DG3)
    }

    @Test
    fun `StatusWord constants are correct`() {
        assertEquals(0x9000, ApduHelper.StatusWord.SUCCESS)
        assertEquals(0x63C0, ApduHelper.StatusWord.WRONG_PIN_MASK)
        assertEquals(0x6982, ApduHelper.StatusWord.SECURITY_NOT_SATISFIED)
        assertEquals(0x6983, ApduHelper.StatusWord.AUTH_METHOD_BLOCKED)
        assertEquals(0x6A82, ApduHelper.StatusWord.FILE_NOT_FOUND)
        assertEquals(0x6A86, ApduHelper.StatusWord.WRONG_PARAMETERS)
        assertEquals(0x6700, ApduHelper.StatusWord.WRONG_LENGTH)
        assertEquals(0x6985, ApduHelper.StatusWord.CONDITIONS_NOT_SATISFIED)
    }

    @Test
    fun `TURKISH_EID_AID is correct`() {
        val expectedAid = byteArrayOf(
            0xA0.toByte(), 0x00, 0x00, 0x01, 0x67,
            0x45, 0x53, 0x49, 0x44
        )
        assertArrayEquals(expectedAid, ApduHelper.TURKISH_EID_AID)
    }
}
