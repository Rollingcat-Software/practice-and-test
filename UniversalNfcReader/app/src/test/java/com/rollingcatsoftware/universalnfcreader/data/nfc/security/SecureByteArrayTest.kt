package com.rollingcatsoftware.universalnfcreader.data.nfc.security

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for SecureByteArray.
 *
 * These tests verify:
 * 1. Basic operations (read, write, copy)
 * 2. Memory clearing on close
 * 3. State management (closed detection)
 * 4. Factory methods
 * 5. Security properties
 */
class SecureByteArrayTest {

    @Test
    fun `wrap creates SecureByteArray with correct size`() {
        val data = byteArrayOf(1, 2, 3, 4, 5)
        val secure = SecureByteArray.wrap(data)

        assertEquals(5, secure.size)
        assertFalse(secure.closed)

        secure.close()
    }

    @Test
    fun `allocate creates zero-filled array`() {
        val secure = SecureByteArray.allocate(10)

        assertEquals(10, secure.size)
        for (i in 0 until 10) {
            assertEquals(0.toByte(), secure[i])
        }

        secure.close()
    }

    @Test
    fun `copyOf creates independent copy`() {
        val original = byteArrayOf(1, 2, 3)
        val secure = SecureByteArray.copyOf(original)

        // Modify original - should not affect secure copy
        original[0] = 99

        assertEquals(1.toByte(), secure[0])

        secure.close()
    }

    @Test
    fun `get returns correct byte at index`() {
        val secure = SecureByteArray.wrap(byteArrayOf(10, 20, 30))

        assertEquals(10.toByte(), secure[0])
        assertEquals(20.toByte(), secure[1])
        assertEquals(30.toByte(), secure[2])

        secure.close()
    }

    @Test
    fun `set modifies byte at index`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))

        secure[1] = 99.toByte()

        assertEquals(99.toByte(), secure[1])

        secure.close()
    }

    @Test
    fun `toByteArray returns copy of data`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        val copy = secure.toByteArray()

        // Modify copy - should not affect secure
        copy[0] = 99

        assertEquals(1.toByte(), secure[0])

        secure.close()
    }

    @Test
    fun `copyOfRange returns correct subset`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3, 4, 5))
        val range = secure.copyOfRange(1, 4)

        assertArrayEquals(byteArrayOf(2, 3, 4), range)

        secure.close()
    }

    @Test
    fun `withData provides access to underlying data`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))

        val sum = secure.withData { data ->
            data.sumOf { it.toInt() }
        }

        assertEquals(6, sum)

        secure.close()
    }

    @Test
    fun `fill sets all bytes to specified value`() {
        val secure = SecureByteArray.allocate(5)
        secure.fill(42.toByte())

        for (i in 0 until 5) {
            assertEquals(42.toByte(), secure[i])
        }

        secure.close()
    }

    @Test
    fun `xorWith byte array produces correct result`() {
        val secure = SecureByteArray.wrap(byteArrayOf(0x0F, 0xF0.toByte(), 0x55))
        val other = byteArrayOf(0xF0.toByte(), 0x0F, 0xAA.toByte())

        secure.xorWith(other)

        assertEquals(0xFF.toByte(), secure[0])
        assertEquals(0xFF.toByte(), secure[1])
        assertEquals(0xFF.toByte(), secure[2])

        secure.close()
    }

    @Test
    fun `xorWith SecureByteArray produces correct result`() {
        val secure1 = SecureByteArray.wrap(byteArrayOf(0x0F, 0xF0.toByte()))
        val secure2 = SecureByteArray.wrap(byteArrayOf(0xF0.toByte(), 0x0F))

        secure1.xorWith(secure2)

        assertEquals(0xFF.toByte(), secure1[0])
        assertEquals(0xFF.toByte(), secure1[1])

        secure1.close()
        secure2.close()
    }

    @Test(expected = IllegalArgumentException::class)
    fun `xorWith throws on size mismatch`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        val other = byteArrayOf(1, 2)

        secure.xorWith(other)
    }

    @Test
    fun `close zeros out data`() {
        val data = byteArrayOf(1, 2, 3, 4, 5)
        val secure = SecureByteArray.wrap(data)

        secure.close()

        // The wrapped array should be zeroed
        for (byte in data) {
            assertEquals(0.toByte(), byte)
        }
        assertTrue(secure.closed)
    }

    @Test
    fun `close is idempotent`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))

        secure.close()
        secure.close() // Should not throw

        assertTrue(secure.closed)
    }

    @Test(expected = IllegalStateException::class)
    fun `get throws after close`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        secure.close()

        secure[0] // Should throw
    }

    @Test(expected = IllegalStateException::class)
    fun `set throws after close`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        secure.close()

        secure[0] = 99.toByte() // Should throw
    }

    @Test(expected = IllegalStateException::class)
    fun `size throws after close`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        secure.close()

        secure.size // Should throw
    }

    @Test(expected = IllegalStateException::class)
    fun `toByteArray throws after close`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        secure.close()

        secure.toByteArray() // Should throw
    }

    @Test(expected = IllegalStateException::class)
    fun `withData throws after close`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        secure.close()

        secure.withData { it.size } // Should throw
    }

    @Test
    fun `use block automatically closes`() {
        val data = byteArrayOf(1, 2, 3)
        val secure = SecureByteArray.wrap(data)

        secure.use { s ->
            assertEquals(3, s.size)
        }

        assertTrue(secure.closed)
        // Data should be zeroed
        for (byte in data) {
            assertEquals(0.toByte(), byte)
        }
    }

    @Test
    fun `fromHex creates correct array`() {
        val secure = SecureByteArray.fromHex("DEADBEEF")

        assertEquals(4, secure.size)
        assertEquals(0xDE.toByte(), secure[0])
        assertEquals(0xAD.toByte(), secure[1])
        assertEquals(0xBE.toByte(), secure[2])
        assertEquals(0xEF.toByte(), secure[3])

        secure.close()
    }

    @Test(expected = IllegalArgumentException::class)
    fun `fromHex throws on odd length`() {
        SecureByteArray.fromHex("ABC")
    }

    @Test
    fun `concat SecureByteArrays creates combined array`() {
        val a = SecureByteArray.wrap(byteArrayOf(1, 2))
        val b = SecureByteArray.wrap(byteArrayOf(3, 4))
        val c = SecureByteArray.wrap(byteArrayOf(5))

        val combined = SecureByteArray.concat(a, b, c)

        assertEquals(5, combined.size)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 5), combined.toByteArray())

        a.close()
        b.close()
        c.close()
        combined.close()
    }

    @Test
    fun `concat ByteArrays creates combined SecureByteArray`() {
        val combined = SecureByteArray.concat(
            byteArrayOf(1, 2),
            byteArrayOf(3, 4, 5)
        )

        assertEquals(5, combined.size)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 5), combined.toByteArray())

        combined.close()
    }

    @Test
    fun `random creates non-zero array`() {
        val secure = SecureByteArray.random(16)

        assertEquals(16, secure.size)

        // Extremely unlikely all 16 bytes are zero
        var hasNonZero = false
        secure.withData { data ->
            hasNonZero = data.any { it != 0.toByte() }
        }
        assertTrue("Random array should contain non-zero bytes", hasNonZero)

        secure.close()
    }

    @Test
    fun `equals returns true for identical content`() {
        val a = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        val b = SecureByteArray.wrap(byteArrayOf(1, 2, 3))

        assertEquals(a, b)

        a.close()
        b.close()
    }

    @Test
    fun `equals returns false for different content`() {
        val a = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        val b = SecureByteArray.wrap(byteArrayOf(1, 2, 4))

        assertNotEquals(a, b)

        a.close()
        b.close()
    }

    @Test
    fun `equals returns false when closed`() {
        val a = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        val b = SecureByteArray.wrap(byteArrayOf(1, 2, 3))

        a.close()

        assertNotEquals(a, b)

        b.close()
    }

    @Test
    fun `toString does not expose data`() {
        val secure = SecureByteArray.wrap(byteArrayOf(0xDE.toByte(), 0xAD.toByte()))

        val str = secure.toString()

        assertFalse("toString should not contain hex data", str.contains("DE"))
        assertFalse("toString should not contain hex data", str.contains("AD"))
        assertTrue("toString should contain size", str.contains("2"))

        secure.close()
    }

    @Test
    fun `toString indicates closed state`() {
        val secure = SecureByteArray.wrap(byteArrayOf(1, 2, 3))
        secure.close()

        val str = secure.toString()

        assertTrue("toString should indicate closed", str.contains("closed"))
    }

    @Test
    fun `copyFrom copies data correctly`() {
        val secure = SecureByteArray.allocate(5)
        val src = byteArrayOf(1, 2, 3)

        secure.copyFrom(src, destPos = 1)

        assertEquals(0.toByte(), secure[0])
        assertEquals(1.toByte(), secure[1])
        assertEquals(2.toByte(), secure[2])
        assertEquals(3.toByte(), secure[3])
        assertEquals(0.toByte(), secure[4])

        secure.close()
    }

    // Extension function tests

    @Test
    fun `secureClear zeros array`() {
        val data = byteArrayOf(1, 2, 3, 4, 5)

        data.secureClear()

        for (byte in data) {
            assertEquals(0.toByte(), byte)
        }
    }

    @Test
    fun `secureWipe zeros array`() {
        val data = byteArrayOf(1, 2, 3, 4, 5)

        data.secureWipe()

        for (byte in data) {
            assertEquals(0.toByte(), byte)
        }
    }
}
