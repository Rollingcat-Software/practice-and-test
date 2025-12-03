package com.rollingcatsoftware.universalnfcreader.util

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Unit tests for extension functions.
 */
class ExtensionsTest {

    // ==================== ByteArray Extensions ====================

    @Test
    fun `toHexString converts bytes correctly`() {
        val bytes = byteArrayOf(0x00, 0x0A, 0xFF.toByte(), 0x10)

        assertThat(bytes.toHexString()).isEqualTo("000AFF10")
    }

    @Test
    fun `toHexString handles empty array`() {
        val bytes = byteArrayOf()

        assertThat(bytes.toHexString()).isEmpty()
    }

    @Test
    fun `toHexStringWithSpaces adds spaces`() {
        val bytes = byteArrayOf(0x01, 0x02, 0x03)

        assertThat(bytes.toHexStringWithSpaces()).isEqualTo("01 02 03")
    }

    @Test
    fun `hexToByteArray converts string correctly`() {
        val hex = "000AFF10"

        val bytes = hex.hexToByteArray()

        assertThat(bytes).isEqualTo(byteArrayOf(0x00, 0x0A, 0xFF.toByte(), 0x10))
    }

    @Test
    fun `hexToByteArray handles spaces and colons`() {
        val hex = "00:0A FF:10"

        val bytes = hex.hexToByteArray()

        assertThat(bytes).isEqualTo(byteArrayOf(0x00, 0x0A, 0xFF.toByte(), 0x10))
    }

    @Test
    fun `getStatusWord extracts last two bytes`() {
        val response = byteArrayOf(0x01, 0x02, 0x90.toByte(), 0x00)

        assertThat(response.getStatusWord()).isEqualTo(0x9000)
    }

    @Test
    fun `isSuccess returns true for 9000`() {
        val response = byteArrayOf(0x90.toByte(), 0x00)

        assertThat(response.isSuccess()).isTrue()
    }

    @Test
    fun `isSuccess returns false for error status`() {
        val response = byteArrayOf(0x69.toByte(), 0x82.toByte())

        assertThat(response.isSuccess()).isFalse()
    }

    @Test
    fun `getResponseData returns data without status word`() {
        val response = byteArrayOf(0x01, 0x02, 0x03, 0x90.toByte(), 0x00)

        assertThat(response.getResponseData()).isEqualTo(byteArrayOf(0x01, 0x02, 0x03))
    }

    @Test
    fun `getRemainingBytes extracts count from 61XX status`() {
        val response = byteArrayOf(0x61, 0x20)

        assertThat(response.getRemainingBytes()).isEqualTo(0x20)
    }

    @Test
    fun `getRemainingBytes returns null for non-61XX status`() {
        val response = byteArrayOf(0x90.toByte(), 0x00)

        assertThat(response.getRemainingBytes()).isNull()
    }

    @Test
    fun `xor combines arrays correctly`() {
        val a = byteArrayOf(0x0F, 0xF0.toByte())
        val b = byteArrayOf(0x55, 0xAA.toByte())

        val result = a xor b

        assertThat(result).isEqualTo(byteArrayOf(0x5A, 0x5A))
    }

    @Test
    fun `padTo extends array with pad bytes`() {
        val bytes = byteArrayOf(0x01, 0x02)

        val padded = bytes.padTo(4, 0xFF.toByte())

        assertThat(padded).isEqualTo(byteArrayOf(0x01, 0x02, 0xFF.toByte(), 0xFF.toByte()))
    }

    @Test
    fun `padTo returns original if already long enough`() {
        val bytes = byteArrayOf(0x01, 0x02, 0x03, 0x04)

        val padded = bytes.padTo(2)

        assertThat(padded).isEqualTo(bytes)
    }

    @Test
    fun `padStart adds bytes at beginning`() {
        val bytes = byteArrayOf(0x01, 0x02)

        val padded = bytes.padStart(4, 0x00)

        assertThat(padded).isEqualTo(byteArrayOf(0x00, 0x00, 0x01, 0x02))
    }

    @Test
    fun `formatAsUid formats with colons`() {
        val uid = byteArrayOf(0x04, 0x8A.toByte(), 0x2F, 0x1C)

        assertThat(uid.formatAsUid()).isEqualTo("04:8A:2F:1C")
    }

    // ==================== Byte Extensions ====================

    @Test
    fun `toUnsignedInt handles negative bytes`() {
        val byte = 0xFF.toByte()

        assertThat(byte.toUnsignedInt()).isEqualTo(255)
    }

    @Test
    fun `twoByteToInt combines bytes correctly`() {
        val result = twoByteToInt(0x12, 0x34)

        assertThat(result).isEqualTo(0x1234)
    }

    @Test
    fun `fourBytesToInt combines bytes correctly`() {
        val bytes = byteArrayOf(0x12, 0x34, 0x56, 0x78)

        assertThat(fourBytesToInt(bytes)).isEqualTo(0x12345678)
    }

    @Test
    fun `Int toByteArray converts correctly`() {
        val value = 0x12345678

        assertThat(value.toByteArray()).isEqualTo(byteArrayOf(0x12, 0x34, 0x56, 0x78))
    }

    @Test
    fun `safeSlice returns correct range`() {
        val bytes = byteArrayOf(0x00, 0x01, 0x02, 0x03, 0x04)

        assertThat(bytes.safeSlice(1, 3)).isEqualTo(byteArrayOf(0x01, 0x02, 0x03))
    }

    @Test
    fun `safeSlice returns null for out of bounds`() {
        val bytes = byteArrayOf(0x00, 0x01, 0x02)

        assertThat(bytes.safeSlice(2, 3)).isNull()
    }

    // ==================== MRZ Extensions ====================

    @Test
    fun `calculateMrzCheckDigit computes correctly for digits`() {
        val data = "123456789"

        // Calculation: 1*7 + 2*3 + 3*1 + 4*7 + 5*3 + 6*1 + 7*7 + 8*3 + 9*1
        // = 7 + 6 + 3 + 28 + 15 + 6 + 49 + 24 + 9 = 147 mod 10 = 7
        assertThat(data.calculateMrzCheckDigit()).isEqualTo(7)
    }

    @Test
    fun `cleanMrz replaces angle brackets with spaces`() {
        val mrz = "DOE<<JOHN<"

        assertThat(mrz.cleanMrz()).isEqualTo("DOE  JOHN")
    }

    @Test
    fun `formatMrzDate converts YYMMDD to readable format`() {
        val date = "901215"

        assertThat(date.formatMrzDate()).isEqualTo("15/12/1990")
    }

    @Test
    fun `formatMrzDate handles 2000s dates`() {
        val date = "250101"

        assertThat(date.formatMrzDate()).isEqualTo("01/01/2025")
    }
}
