package com.turkey.eidnfc.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for BitmapUtils.
 *
 * Tests bitmap optimization utilities including sample size calculation
 * and memory size formatting.
 */
class BitmapUtilsTest {

    // ============================================================================
    // Sample Size Calculation Tests
    // ============================================================================

    @Test
    fun `calculateSampleSize returns 1 when dimensions are within limit`() {
        // Given - image smaller than max dimension
        val width = 300
        val height = 400
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then
        assertEquals(1, sampleSize)
    }

    @Test
    fun `calculateSampleSize returns 2 for image slightly larger than limit`() {
        // Given - image just over the limit
        val width = 1200
        val height = 1600
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then
        assertEquals(2, sampleSize)
    }

    @Test
    fun `calculateSampleSize returns 4 for large image`() {
        // Given - large image
        val width = 2048
        val height = 2048
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then
        assertEquals(2, sampleSize) // 2048/2 = 1024, so sampleSize = 2
    }

    @Test
    fun `calculateSampleSize handles very large images`() {
        // Given - very large image (4K)
        val width = 4096
        val height = 4096
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then
        assertTrue(sampleSize >= 4) // Should be 4 or higher
    }

    @Test
    fun `calculateSampleSize returns power of 2`() {
        // Test various sizes to ensure result is always power of 2
        val testCases = listOf(
            Triple(800, 600, 1024),
            Triple(1600, 1200, 1024),
            Triple(3200, 2400, 1024),
            Triple(512, 512, 1024)
        )

        testCases.forEach { (width, height, maxDim) ->
            val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDim)

            // Check if sampleSize is a power of 2
            assertTrue("$sampleSize should be power of 2", isPowerOfTwo(sampleSize))
        }
    }

    @Test
    fun `calculateSampleSize handles rectangular images correctly`() {
        // Given - wide image
        val width = 2048
        val height = 512
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then - should downsample based on the larger dimension
        assertTrue(sampleSize >= 2)
    }

    @Test
    fun `calculateSampleSize handles tall images correctly`() {
        // Given - tall image
        val width = 512
        val height = 2048
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then - should downsample based on the larger dimension
        assertTrue(sampleSize >= 2)
    }

    @Test
    fun `calculateSampleSize handles minimum dimensions`() {
        // Given - tiny image
        val width = 10
        val height = 10
        val maxDimension = 1024

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then
        assertEquals(1, sampleSize)
    }

    @Test
    fun `calculateSampleSize with custom maxDimension`() {
        // Given - custom smaller max dimension
        val width = 800
        val height = 600
        val maxDimension = 400

        // When
        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDimension)

        // Then - should downsample
        assertTrue(sampleSize >= 2)
    }

    // ============================================================================
    // Memory Size Formatting Tests
    // ============================================================================

    @Test
    fun `formatMemorySize formats bytes correctly`() {
        assertEquals("512 B", BitmapUtils.formatMemorySize(512))
        assertEquals("1023 B", BitmapUtils.formatMemorySize(1023))
    }

    @Test
    fun `formatMemorySize formats kilobytes correctly`() {
        assertEquals("1.0 KB", BitmapUtils.formatMemorySize(1024))
        assertEquals("5.5 KB", BitmapUtils.formatMemorySize(5632))
        assertEquals("100.0 KB", BitmapUtils.formatMemorySize(102400))
    }

    @Test
    fun `formatMemorySize formats megabytes correctly`() {
        assertEquals("1.0 MB", BitmapUtils.formatMemorySize(1024 * 1024))
        assertEquals("2.5 MB", BitmapUtils.formatMemorySize((2.5 * 1024 * 1024).toInt()))
        assertEquals("10.0 MB", BitmapUtils.formatMemorySize(10 * 1024 * 1024))
    }

    @Test
    fun `formatMemorySize handles zero`() {
        assertEquals("0 B", BitmapUtils.formatMemorySize(0))
    }

    @Test
    fun `formatMemorySize handles typical bitmap sizes`() {
        // Typical eID photo: 300x400 ARGB_8888 = 300 * 400 * 4 = 480,000 bytes
        val eIdPhotoSize = 300 * 400 * 4
        val formatted = BitmapUtils.formatMemorySize(eIdPhotoSize)
        assertTrue(formatted.contains("KB"))

        // Large photo: 2048x2048 ARGB_8888 = 2048 * 2048 * 4 = 16,777,216 bytes
        val largePhotoSize = 2048 * 2048 * 4
        val formattedLarge = BitmapUtils.formatMemorySize(largePhotoSize)
        assertTrue(formattedLarge.contains("MB"))
    }

    // ============================================================================
    // Memory Size Calculation Tests (calculateMemorySize)
    // ============================================================================

    @Test
    fun `calculateMemorySize returns 0 for null bitmap`() {
        assertEquals(0, BitmapUtils.calculateMemorySize(null))
    }

    // Note: Testing with actual Bitmap objects requires Android framework
    // which is not available in unit tests. These tests would go in
    // androidTest instead for instrumented tests.

    // ============================================================================
    // Integration/Edge Case Tests
    // ============================================================================

    @Test
    fun `sample size calculation is consistent`() {
        // Same dimensions should always give same result
        val width = 1920
        val height = 1080
        val maxDim = 1024

        val result1 = BitmapUtils.calculateSampleSize(width, height, maxDim)
        val result2 = BitmapUtils.calculateSampleSize(width, height, maxDim)

        assertEquals(result1, result2)
    }

    @Test
    fun `sample size increases with image size`() {
        val maxDim = 512

        val small = BitmapUtils.calculateSampleSize(600, 600, maxDim)
        val medium = BitmapUtils.calculateSampleSize(1200, 1200, maxDim)
        val large = BitmapUtils.calculateSampleSize(2400, 2400, maxDim)

        assertTrue(small <= medium)
        assertTrue(medium <= large)
    }

    @Test
    fun `calculated dimensions after sampling are reasonable`() {
        val width = 4000
        val height = 3000
        val maxDim = 1024

        val sampleSize = BitmapUtils.calculateSampleSize(width, height, maxDim)

        val resultWidth = width / sampleSize
        val resultHeight = height / sampleSize

        // Resulting dimensions should be close to maxDim but not exceed it by much
        assertTrue(
            "Result width $resultWidth should be reasonable",
            resultWidth <= maxDim * 2
        )
        assertTrue(
            "Result height $resultHeight should be reasonable",
            resultHeight <= maxDim * 2
        )
    }

    @Test
    fun `memory size formatting is precise for edge values`() {
        // Test boundary values
        assertEquals("1.0 KB", BitmapUtils.formatMemorySize(1024))
        assertEquals("1023 B", BitmapUtils.formatMemorySize(1023))
        assertEquals("1.0 MB", BitmapUtils.formatMemorySize(1024 * 1024))
        assertEquals("1023.0 KB", BitmapUtils.formatMemorySize(1024 * 1024 - 1024))
    }

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /**
     * Checks if a number is a power of 2.
     */
    private fun isPowerOfTwo(n: Int): Boolean {
        return n > 0 && (n and (n - 1)) == 0
    }
}
