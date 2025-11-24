package com.turkey.eidnfc.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import timber.log.Timber
import kotlin.math.max

/**
 * Utility functions for optimized bitmap handling.
 *
 * Provides memory-efficient bitmap operations including:
 * - Downsampling to reduce memory usage
 * - Sample size calculation
 * - Bitmap recycling helpers
 */
object BitmapUtils {

    /**
     * Maximum dimensions for decoded bitmaps to prevent OutOfMemoryError.
     * eID photos are typically small (around 300x400), but this provides a safety limit.
     */
    private const val MAX_DIMENSION = 1024

    /**
     * Decodes byte array to bitmap with automatic downsampling if needed.
     *
     * This method is more memory-efficient than BitmapFactory.decodeByteArray
     * because it:
     * 1. First decodes bounds only (no memory allocation)
     * 2. Calculates appropriate sample size
     * 3. Decodes with sample size to reduce memory usage
     *
     * @param data The image data bytes
     * @param maxDimension Maximum width/height (defaults to MAX_DIMENSION)
     * @return Optimized bitmap, or null if decoding fails
     */
    fun decodeOptimized(
        data: ByteArray,
        maxDimension: Int = MAX_DIMENSION
    ): Bitmap? {
        return try {
            // First, decode just the bounds without allocating memory for pixels
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeByteArray(data, 0, data.size, options)

            val originalWidth = options.outWidth
            val originalHeight = options.outHeight

            if (originalWidth <= 0 || originalHeight <= 0) {
                Timber.e("Invalid bitmap dimensions: ${originalWidth}x${originalHeight}")
                return null
            }

            Timber.d("Original bitmap size: ${originalWidth}x${originalHeight}")

            // Calculate sample size for downsampling
            val sampleSize = calculateSampleSize(
                originalWidth,
                originalHeight,
                maxDimension
            )

            Timber.d("Using sample size: $sampleSize")

            // Decode with sample size for memory efficiency
            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888 // Good quality for photos
                inMutable = false // Immutable bitmaps use less memory
            }

            val bitmap = BitmapFactory.decodeByteArray(data, 0, data.size, decodeOptions)

            if (bitmap != null) {
                Timber.d("Decoded bitmap size: ${bitmap.width}x${bitmap.height}, " +
                        "bytes: ${bitmap.byteCount}")
            }

            bitmap

        } catch (e: OutOfMemoryError) {
            Timber.e(e, "Out of memory while decoding bitmap")
            // Try again with more aggressive downsampling
            tryAggressiveDownsampling(data, maxDimension / 2)
        } catch (e: Exception) {
            Timber.e(e, "Failed to decode bitmap")
            null
        }
    }

    /**
     * Calculates the optimal sample size for downsampling.
     *
     * Sample size is always a power of 2 for efficiency.
     * For example:
     * - sampleSize = 1: no downsampling
     * - sampleSize = 2: image is 1/2 the size
     * - sampleSize = 4: image is 1/4 the size
     *
     * @param width Original width
     * @param height Original height
     * @param maxDimension Maximum allowed dimension
     * @return Sample size (always a power of 2)
     */
    fun calculateSampleSize(
        width: Int,
        height: Int,
        maxDimension: Int
    ): Int {
        var sampleSize = 1

        if (width > maxDimension || height > maxDimension) {
            val halfWidth = width / 2
            val halfHeight = height / 2

            // Calculate the largest sampleSize that keeps dimensions above maxDimension
            while ((halfWidth / sampleSize) >= maxDimension ||
                (halfHeight / sampleSize) >= maxDimension
            ) {
                sampleSize *= 2
            }
        }

        return max(1, sampleSize)
    }

    /**
     * Tries decoding with more aggressive downsampling as a fallback.
     *
     * Used when normal decoding fails due to memory constraints.
     */
    private fun tryAggressiveDownsampling(
        data: ByteArray,
        maxDimension: Int
    ): Bitmap? {
        return try {
            Timber.d("Attempting aggressive downsampling (maxDim: $maxDimension)")

            val options = BitmapFactory.Options().apply {
                inSampleSize = 4 // Aggressively downsample by 4x
                inPreferredConfig = Bitmap.Config.RGB_565 // Use less memory
                inMutable = false
            }

            BitmapFactory.decodeByteArray(data, 0, data.size, options)

        } catch (e: Exception) {
            Timber.e(e, "Aggressive downsampling also failed")
            null
        }
    }

    /**
     * Safely recycles a bitmap if it's not null and not already recycled.
     *
     * @param bitmap The bitmap to recycle
     */
    fun recycleSafely(bitmap: Bitmap?) {
        try {
            if (bitmap != null && !bitmap.isRecycled) {
                bitmap.recycle()
                Timber.d("Bitmap recycled successfully")
            }
        } catch (e: Exception) {
            Timber.e(e, "Error recycling bitmap")
        }
    }

    /**
     * Calculates the memory size of a bitmap in bytes.
     *
     * @param bitmap The bitmap to measure
     * @return Size in bytes
     */
    fun calculateMemorySize(bitmap: Bitmap?): Int {
        return bitmap?.byteCount ?: 0
    }

    /**
     * Formats bitmap memory size as human-readable string.
     *
     * @param bytes Size in bytes
     * @return Formatted string (e.g., "1.5 MB", "500 KB")
     */
    fun formatMemorySize(bytes: Int): String {
        return when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> String.format("%.1f KB", bytes / 1024.0)
            else -> String.format("%.1f MB", bytes / (1024.0 * 1024.0))
        }
    }

    /**
     * Creates a scaled version of the bitmap.
     *
     * Uses FILTER=true for better quality when downscaling.
     *
     * @param bitmap Source bitmap
     * @param maxWidth Maximum width
     * @param maxHeight Maximum height
     * @return Scaled bitmap
     */
    fun createScaledBitmap(
        bitmap: Bitmap,
        maxWidth: Int,
        maxHeight: Int
    ): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        if (width <= maxWidth && height <= maxHeight) {
            return bitmap
        }

        val scale = minOf(
            maxWidth.toFloat() / width,
            maxHeight.toFloat() / height
        )

        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        Timber.d("Scaling bitmap from ${width}x${height} to ${newWidth}x${newHeight}")

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
}
