package com.turkey.eidnfc.util

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for extension functions.
 *
 * Tests all extension functions except Context extensions which require Android framework.
 */
class ExtensionsTest {

    // ============================================================================
    // ByteArray Extensions Tests
    // ============================================================================

    @Test
    fun `toHexString converts byte array correctly`() {
        val bytes = byteArrayOf(0x01, 0x02, 0x0A, 0xFF.toByte())
        assertEquals("01 02 0A FF", bytes.toHexString())
    }

    @Test
    fun `toHexString handles empty array`() {
        assertEquals("", byteArrayOf().toHexString())
    }

    @Test
    fun `toHexString handles single byte`() {
        assertEquals("AB", byteArrayOf(0xAB.toByte()).toHexString())
    }

    @Test
    fun `toHexString handles zero bytes`() {
        assertEquals("00 00 00", byteArrayOf(0x00, 0x00, 0x00).toHexString())
    }

    @Test
    fun `toInt converts single byte correctly`() {
        assertEquals(255, byteArrayOf(0xFF.toByte()).toInt())
        assertEquals(0, byteArrayOf(0x00).toInt())
        assertEquals(127, byteArrayOf(0x7F).toInt())
    }

    @Test
    fun `toInt converts two bytes correctly`() {
        assertEquals(0x0102, byteArrayOf(0x01, 0x02).toInt())
        assertEquals(0xFFFF, byteArrayOf(0xFF.toByte(), 0xFF.toByte()).toInt())
        assertEquals(256, byteArrayOf(0x01, 0x00).toInt())
    }

    @Test
    fun `toInt converts four bytes correctly`() {
        assertEquals(
            0x01020304,
            byteArrayOf(0x01, 0x02, 0x03, 0x04).toInt()
        )
        assertEquals(
            -1, // 0xFFFFFFFF in signed int
            byteArrayOf(0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte()).toInt()
        )
    }

    @Test
    fun `toInt handles empty array`() {
        assertEquals(0, byteArrayOf().toInt())
    }

    @Test(expected = IllegalArgumentException::class)
    fun `toInt throws exception for array larger than 4 bytes`() {
        byteArrayOf(0x01, 0x02, 0x03, 0x04, 0x05).toInt()
    }

    // ============================================================================
    // String Extensions Tests - PIN Validation
    // ============================================================================

    @Test
    fun `isValidPin returns true for valid 6-digit PIN`() {
        assertTrue("123456".isValidPin())
        assertTrue("000000".isValidPin())
        assertTrue("999999".isValidPin())
        assertTrue("000001".isValidPin())
    }

    @Test
    fun `isValidPin returns false for invalid length`() {
        assertFalse("12345".isValidPin())   // Too short
        assertFalse("1234567".isValidPin()) // Too long
        assertFalse("".isValidPin())        // Empty
        assertFalse("1".isValidPin())       // Too short
    }

    @Test
    fun `isValidPin returns false for non-digit characters`() {
        assertFalse("12345a".isValidPin())   // Contains letter
        assertFalse("abcdef".isValidPin())   // All letters
        assertFalse("12 456".isValidPin())   // Contains space
        assertFalse("123-56".isValidPin())   // Contains dash
        assertFalse("12.456".isValidPin())   // Contains dot
    }

    // ============================================================================
    // String Extensions Tests - TCKN Validation
    // ============================================================================

    @Test
    fun `isValidTckn returns true for valid TCKN examples`() {
        // These are mathematically valid TCKNs (not real person IDs)
        assertTrue("10000000146".isValidTckn()) // Example valid TCKN
        assertTrue("11111111110".isValidTckn()) // Edge case valid TCKN
    }

    @Test
    fun `isValidTckn returns false for invalid length`() {
        assertFalse("1234567890".isValidTckn())   // Too short (10 digits)
        assertFalse("123456789012".isValidTckn()) // Too long (12 digits)
        assertFalse("".isValidTckn())             // Empty
    }

    @Test
    fun `isValidTckn returns false for non-digits`() {
        assertFalse("1234567890a".isValidTckn())  // Contains letter
        assertFalse("12345 67890".isValidTckn())  // Contains space
    }

    @Test
    fun `isValidTckn returns false when first digit is zero`() {
        assertFalse("01234567890".isValidTckn())
    }

    @Test
    fun `isValidTckn validates 10th digit correctly`() {
        // Manual calculation test
        // TCKN: 10000000146
        // Odd positions (1,3,5,7,9): 1+0+0+0+0 = 1
        // Even positions (2,4,6,8): 0+0+0+0 = 0
        // 10th digit: ((1*7) - 0) % 10 = 7 % 10 = 7? No, let me recalculate
        // Position 0-indexed: digits[0]=1, digits[2]=0, etc.
        // I'll test with a known valid TCKN structure
        val validTckn = "10000000146"
        assertTrue(validTckn.isValidTckn())

        // Invalid 10th digit (modify the 10th position)
        val invalidTckn = validTckn.substring(0, 9) + "9" + validTckn.substring(10)
        assertFalse(invalidTckn.isValidTckn())
    }

    @Test
    fun `isValidTckn validates 11th digit correctly`() {
        val validTckn = "10000000146"
        assertTrue(validTckn.isValidTckn())

        // Invalid 11th digit (modify the last position)
        val invalidTckn = validTckn.substring(0, 10) + "9"
        assertFalse(invalidTckn.isValidTckn())
    }

    @Test
    fun `isValidTckn algorithm verification with manual calculation`() {
        // Let's manually verify "11111111110"
        // digits: [1,1,1,1,1,1,1,1,1,1,0]
        // Odd sum (0,2,4,6,8): 1+1+1+1+1 = 5
        // Even sum (1,3,5,7): 1+1+1+1 = 4
        // 10th digit check: ((5*7) - 4) % 10 = (35-4) % 10 = 31 % 10 = 1
        // Expected 10th digit (index 9) = 1 ✓
        // Sum of first 10: 1+1+1+1+1+1+1+1+1+1 = 10
        // 11th digit check: 10 % 10 = 0
        // Expected 11th digit (index 10) = 0 ✓
        assertTrue("11111111110".isValidTckn())
    }

    // ============================================================================
    // String Extensions Tests - Masking
    // ============================================================================

    @Test
    fun `mask replaces all characters with bullets`() {
        assertEquals("••••••", "123456".mask())
        assertEquals("•••", "abc".mask())
    }

    @Test
    fun `mask with custom character works`() {
        assertEquals("******", "123456".mask('*'))
        assertEquals("xxx", "abc".mask('x'))
    }

    @Test
    fun `mask handles empty string`() {
        assertEquals("", "".mask())
    }

    @Test
    fun `mask handles single character`() {
        assertEquals("•", "a".mask())
    }

    // ============================================================================
    // String Extensions Tests - Capitalize Words
    // ============================================================================

    @Test
    fun `capitalizeWords capitalizes first letter of each word`() {
        assertEquals("Mehmet Ali", "mehmet ali".capitalizeWords())
        assertEquals("John Doe", "john doe".capitalizeWords())
    }

    @Test
    fun `capitalizeWords handles single word`() {
        assertEquals("Hello", "hello".capitalizeWords())
        assertEquals("World", "WORLD".capitalizeWords())
    }

    @Test
    fun `capitalizeWords handles already capitalized text`() {
        assertEquals("Hello World", "Hello World".capitalizeWords())
    }

    @Test
    fun `capitalizeWords handles mixed case`() {
        assertEquals("Ahmet Mehmet", "AHMET MEHMET".capitalizeWords())
        assertEquals("Ayşe Fatma", "aYşE fAtMa".capitalizeWords())
    }

    @Test
    fun `capitalizeWords handles empty string`() {
        assertEquals("", "".capitalizeWords())
    }

    @Test
    fun `capitalizeWords handles multiple spaces`() {
        assertEquals("Hello  World", "hello  world".capitalizeWords())
    }

    @Test
    fun `capitalizeWords handles single character words`() {
        assertEquals("A B C", "a b c".capitalizeWords())
    }

    // ============================================================================
    // Number Extensions Tests
    // ============================================================================

    @Test
    fun `toSeconds converts milliseconds correctly`() {
        assertEquals(1L, 1000L.toSeconds())
        assertEquals(5L, 5000L.toSeconds())
        assertEquals(0L, 500L.toSeconds())
        assertEquals(60L, 60000L.toSeconds())
    }

    @Test
    fun `toSeconds handles zero`() {
        assertEquals(0L, 0L.toSeconds())
    }

    @Test
    fun `toSeconds handles large values`() {
        assertEquals(3600L, 3600000L.toSeconds()) // 1 hour
    }

    @Test
    fun `toMillis converts seconds correctly`() {
        assertEquals(1000L, 1.toMillis())
        assertEquals(5000L, 5.toMillis())
        assertEquals(0L, 0.toMillis())
        assertEquals(60000L, 60.toMillis())
    }

    @Test
    fun `toMillis handles large values`() {
        assertEquals(3600000L, 3600.toMillis()) // 1 hour
    }

    // ============================================================================
    // Collection Extensions Tests
    // ============================================================================

    @Test
    fun `getOrNull returns element when index is valid`() {
        val list = listOf(1, 2, 3, 4, 5)
        assertEquals(1, list.getOrNull(0))
        assertEquals(3, list.getOrNull(2))
        assertEquals(5, list.getOrNull(4))
    }

    @Test
    fun `getOrNull returns null when index is out of bounds`() {
        val list = listOf(1, 2, 3)
        assertNull(list.getOrNull(-1))
        assertNull(list.getOrNull(3))
        assertNull(list.getOrNull(10))
    }

    @Test
    fun `getOrNull handles empty list`() {
        val list = emptyList<Int>()
        assertNull(list.getOrNull(0))
        assertNull(list.getOrNull(-1))
    }

    @Test
    fun `getOrNull works with different types`() {
        val stringList = listOf("a", "b", "c")
        assertEquals("b", stringList.getOrNull(1))
        assertNull(stringList.getOrNull(5))
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    @Test
    fun `ByteArray toHexString and String operations work together`() {
        val bytes = byteArrayOf(0x01, 0x02, 0x03)
        val hex = bytes.toHexString()
        assertEquals("01 02 03", hex)
        assertEquals("•• •• ••", hex.mask())
    }

    @Test
    fun `time conversion functions are inverses of each other`() {
        val seconds = 60
        val millis = seconds.toMillis()
        assertEquals(60000L, millis)
        assertEquals(seconds.toLong(), millis.toSeconds())
    }

    @Test
    fun `mask length equals original string length`() {
        val original = "test123"
        val masked = original.mask()
        assertEquals(original.length, masked.length)
    }
}
