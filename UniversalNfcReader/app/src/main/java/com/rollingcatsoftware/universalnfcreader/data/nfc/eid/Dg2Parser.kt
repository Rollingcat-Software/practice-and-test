package com.rollingcatsoftware.universalnfcreader.data.nfc.eid

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.ByteArrayInputStream
import kotlin.math.max

/**
 * Parser for DG2 (Data Group 2) from Turkish eID card.
 *
 * DG2 contains the facial image encoded in JPEG2000 format.
 * The image is wrapped in ICAO LDS ASN.1 structure.
 *
 * Structure:
 * - Tag 0x75: DG2 wrapper
 * - Tag 0x7F61: Biometric Information Template
 * - Tag 0x7F60: Biometric Information Group Template
 * - Tag 0x5F2E: Biometric Data Block (contains image)
 */
object Dg2Parser {

    private const val TAG = "Dg2Parser"

    // ASN.1 Tags for DG2
    private const val TAG_DG2 = 0x75
    private const val TAG_BIOMETRIC_DATA_BLOCK = 0x5F2E
    private const val TAG_IMAGE_DATA = 0x7F2E

    // Image magic bytes
    private val JPEG2000_JP2_MAGIC = byteArrayOf(
        0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20
    )
    private val JPEG2000_CODESTREAM_MAGIC = byteArrayOf(
        0xFF.toByte(), 0x4F, 0xFF.toByte(), 0x51
    )
    private val JPEG_MAGIC = byteArrayOf(
        0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte()
    )

    // Maximum dimension for decoded bitmaps
    private const val MAX_DIMENSION = 1024

    /**
     * Parses DG2 data to extract the facial image.
     *
     * @param dg2Data Raw DG2 data bytes
     * @return Bitmap of the facial image, or null if parsing/decoding fails
     */
    fun parse(dg2Data: ByteArray): Bitmap? {
        return try {
            Log.d(TAG, "Parsing DG2 data (${dg2Data.size} bytes)")
            Log.d(TAG, "DG2 first 32 bytes: ${toHexString(dg2Data.take(32).toByteArray())}")

            // Extract image data from ASN.1 structure
            val imageData = extractImageData(dg2Data)
            if (imageData == null) {
                Log.e(TAG, "Failed to extract image data from DG2")
                return null
            }

            Log.d(TAG, "Extracted image data (${imageData.size} bytes)")
            Log.d(TAG, "Image data first 16 bytes: ${toHexString(imageData.take(16).toByteArray())}")

            // Decode to Bitmap
            decodeImage(imageData)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse DG2 data", e)
            null
        }
    }

    /**
     * Extracts image data from DG2 ASN.1 structure.
     */
    private fun extractImageData(data: ByteArray): ByteArray? {
        try {
            val stream = ByteArrayInputStream(data)

            // Read outer tag (should be 0x75 for DG2)
            val outerTag = readTag(stream)
            if (outerTag != TAG_DG2) {
                Log.w(TAG, "Unexpected DG2 tag: 0x${outerTag.toString(16)}, expected 0x75")
            }

            // Read length
            val outerLength = readLength(stream)
            Log.d(TAG, "DG2 content length: $outerLength bytes")

            // Navigate through the ASN.1 structure to find image data
            while (stream.available() > 0) {
                val tag = readTag(stream)
                val length = readLength(stream)

                Log.d(TAG, "Found tag: 0x${tag.toString(16)}, length: $length")

                when (tag) {
                    TAG_BIOMETRIC_DATA_BLOCK, TAG_IMAGE_DATA -> {
                        // This should contain the image data
                        val imageBytes = ByteArray(length)
                        stream.read(imageBytes)

                        // Check if it's a supported image format
                        val imageType = detectImageType(imageBytes)
                        if (imageType != null) {
                            Log.d(TAG, "Found $imageType image data")
                            return imageBytes
                        } else {
                            Log.w(TAG, "Data block is not a recognized image format, searching...")
                            // Try to find image within this block
                            val innerImage = findImageInRawData(imageBytes)
                            if (innerImage != null) {
                                return innerImage
                            }
                        }
                    }
                    else -> {
                        // For container tags, continue recursively
                        if (length > 0 && length < stream.available()) {
                            val subData = ByteArray(length)
                            stream.read(subData)

                            // Try to find image data recursively
                            val result = extractImageData(subData)
                            if (result != null) {
                                return result
                            }
                        } else {
                            if (length <= stream.available()) {
                                stream.skip(length.toLong())
                            }
                        }
                    }
                }
            }

            // If we haven't found it in the structure, search for image magic bytes
            Log.d(TAG, "Searching for image magic bytes in raw data...")
            return findImageInRawData(data)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract image data", e)
            return null
        }
    }

    /**
     * Searches for any supported image format magic bytes in raw data.
     */
    private fun findImageInRawData(data: ByteArray): ByteArray? {
        // Search for JPEG2000 JP2 container
        findMagicBytes(data, JPEG2000_JP2_MAGIC, "JPEG2000 JP2")?.let { return it }

        // Search for JPEG2000 codestream
        findMagicBytes(data, JPEG2000_CODESTREAM_MAGIC, "JPEG2000 codestream")?.let { return it }

        // Search for regular JPEG
        findMagicBytes(data, JPEG_MAGIC, "JPEG")?.let { return it }

        Log.w(TAG, "No recognized image format found in data")
        return null
    }

    /**
     * Searches for specific magic bytes in data.
     */
    private fun findMagicBytes(data: ByteArray, magic: ByteArray, formatName: String): ByteArray? {
        for (i in 0 until data.size - magic.size) {
            var match = true
            for (j in magic.indices) {
                if (data[i + j] != magic[j]) {
                    match = false
                    break
                }
            }
            if (match) {
                Log.d(TAG, "Found $formatName magic bytes at offset $i")
                return data.copyOfRange(i, data.size)
            }
        }
        return null
    }

    /**
     * Detects the image type from data.
     */
    private fun detectImageType(data: ByteArray): String? {
        if (data.size < 4) return null

        if (startsWith(data, JPEG2000_JP2_MAGIC)) return "JPEG2000 JP2"
        if (startsWith(data, JPEG2000_CODESTREAM_MAGIC)) return "JPEG2000 codestream"
        if (startsWith(data, JPEG_MAGIC)) return "JPEG"

        return null
    }

    /**
     * Checks if data starts with the given bytes.
     */
    private fun startsWith(data: ByteArray, prefix: ByteArray): Boolean {
        if (data.size < prefix.size) return false
        for (i in prefix.indices) {
            if (data[i] != prefix[i]) return false
        }
        return true
    }

    /**
     * Decodes image data to Android Bitmap with optimization.
     */
    private fun decodeImage(imageData: ByteArray): Bitmap? {
        return try {
            Log.d(TAG, "Attempting to decode image (${imageData.size} bytes)...")

            // First try direct decode
            val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)
            if (bitmap != null) {
                Log.d(TAG, "Successfully decoded image: ${bitmap.width}x${bitmap.height}")
                return bitmap
            }

            Log.w(TAG, "BitmapFactory.decodeByteArray returned null, trying optimized decode")
            decodeOptimized(imageData)

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Out of memory while decoding image", e)
            tryAggressiveDownsampling(imageData)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decode image", e)
            null
        }
    }

    /**
     * Decodes with automatic downsampling if needed.
     */
    private fun decodeOptimized(data: ByteArray): Bitmap? {
        return try {
            // First, decode just the bounds
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeByteArray(data, 0, data.size, options)

            val originalWidth = options.outWidth
            val originalHeight = options.outHeight

            if (originalWidth <= 0 || originalHeight <= 0) {
                Log.e(TAG, "Invalid bitmap dimensions: ${originalWidth}x${originalHeight}")
                return null
            }

            Log.d(TAG, "Original bitmap size: ${originalWidth}x${originalHeight}")

            // Calculate sample size for downsampling
            val sampleSize = calculateSampleSize(originalWidth, originalHeight, MAX_DIMENSION)
            Log.d(TAG, "Using sample size: $sampleSize")

            // Decode with sample size
            val decodeOptions = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inMutable = false
            }

            val bitmap = BitmapFactory.decodeByteArray(data, 0, data.size, decodeOptions)

            if (bitmap != null) {
                Log.d(TAG, "Decoded bitmap size: ${bitmap.width}x${bitmap.height}, bytes: ${bitmap.byteCount}")
            }

            bitmap

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Out of memory in optimized decode", e)
            tryAggressiveDownsampling(data)
        } catch (e: Exception) {
            Log.e(TAG, "Failed optimized decode", e)
            null
        }
    }

    /**
     * Calculates optimal sample size for downsampling.
     */
    private fun calculateSampleSize(width: Int, height: Int, maxDimension: Int): Int {
        var sampleSize = 1

        if (width > maxDimension || height > maxDimension) {
            val halfWidth = width / 2
            val halfHeight = height / 2

            while ((halfWidth / sampleSize) >= maxDimension ||
                (halfHeight / sampleSize) >= maxDimension
            ) {
                sampleSize *= 2
            }
        }

        return max(1, sampleSize)
    }

    /**
     * Tries aggressive downsampling as fallback.
     */
    private fun tryAggressiveDownsampling(data: ByteArray): Bitmap? {
        return try {
            Log.d(TAG, "Attempting aggressive downsampling")

            val options = BitmapFactory.Options().apply {
                inSampleSize = 4
                inPreferredConfig = Bitmap.Config.RGB_565
                inMutable = false
            }

            BitmapFactory.decodeByteArray(data, 0, data.size, options)

        } catch (e: Exception) {
            Log.e(TAG, "Aggressive downsampling also failed", e)
            null
        }
    }

    /**
     * Reads an ASN.1 tag from the stream.
     */
    private fun readTag(stream: ByteArrayInputStream): Int {
        val firstByte = stream.read()
        if (firstByte == -1) throw IllegalStateException("Unexpected end of stream")

        return if ((firstByte and 0x1F) == 0x1F) {
            var tag = firstByte shl 8
            tag = tag or stream.read()
            tag
        } else {
            firstByte
        }
    }

    /**
     * Reads an ASN.1 length from the stream.
     */
    private fun readLength(stream: ByteArrayInputStream): Int {
        val firstByte = stream.read()
        if (firstByte == -1) throw IllegalStateException("Unexpected end of stream")

        return if ((firstByte and 0x80) == 0) {
            firstByte
        } else {
            val numBytes = firstByte and 0x7F
            var length = 0
            repeat(numBytes) {
                length = (length shl 8) or stream.read()
            }
            length
        }
    }

    /**
     * Converts byte array to hex string for debugging.
     */
    private fun toHexString(bytes: ByteArray): String {
        return bytes.joinToString(" ") { String.format("%02X", it) }
    }
}
