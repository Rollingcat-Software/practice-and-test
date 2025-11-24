package com.turkey.eidnfc.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import timber.log.Timber
import java.io.ByteArrayInputStream

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
 * - Tag 0x5F2E: Biometric Data Block (contains JPEG2000 image)
 */
object Dg2Parser {

    // ASN.1 Tags for DG2
    private const val TAG_DG2 = 0x75
    private const val TAG_BIOMETRIC_INFO_TEMPLATE = 0x7F61
    private const val TAG_BIOMETRIC_INFO_GROUP_TEMPLATE = 0x7F60
    private const val TAG_BIOMETRIC_DATA_BLOCK = 0x5F2E
    private const val TAG_IMAGE_DATA = 0x7F2E

    // JPEG2000 JP2 container magic bytes (full format)
    private val JPEG2000_JP2_MAGIC = byteArrayOf(
        0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20
    )

    // JPEG2000 codestream magic bytes (.j2k, .j2c format) - FF 4F FF 51
    private val JPEG2000_CODESTREAM_MAGIC = byteArrayOf(
        0xFF.toByte(), 0x4F, 0xFF.toByte(), 0x51
    )

    // Regular JPEG magic bytes - FF D8 FF
    private val JPEG_MAGIC = byteArrayOf(
        0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte()
    )

    /**
     * Parses DG2 data to extract the facial image.
     *
     * @param dg2Data Raw DG2 data bytes
     * @return Bitmap of the facial image, or null if parsing/decoding fails
     */
    fun parse(dg2Data: ByteArray): Bitmap? {
        return try {
            Timber.d("Parsing DG2 data (${dg2Data.size} bytes)")
            Timber.d("DG2 first 32 bytes: ${toHexString(dg2Data.take(32).toByteArray())}")

            // Extract JPEG2000 image data
            val imageData = extractImageData(dg2Data)
            if (imageData == null) {
                Timber.e("Failed to extract image data from DG2")
                return null
            }

            Timber.d("Extracted image data (${imageData.size} bytes)")
            Timber.d("Image data first 16 bytes: ${toHexString(imageData.take(16).toByteArray())}")

            // Decode JPEG2000 to Bitmap
            decodeJpeg2000(imageData)

        } catch (e: Exception) {
            Timber.e(e, "Failed to parse DG2 data")
            null
        }
    }

    /**
     * Extracts JPEG2000 image data from DG2 ASN.1 structure.
     */
    private fun extractImageData(data: ByteArray): ByteArray? {
        try {
            val stream = ByteArrayInputStream(data)

            // Read outer tag (should be 0x75 for DG2)
            val outerTag = readTag(stream)
            if (outerTag != TAG_DG2) {
                Timber.w("Unexpected DG2 tag: 0x${outerTag.toString(16)}, expected 0x75")
            }

            // Read length
            val outerLength = readLength(stream)
            Timber.d("DG2 content length: $outerLength bytes")

            // Navigate through the ASN.1 structure to find image data
            while (stream.available() > 0) {
                val tag = readTag(stream)
                val length = readLength(stream)

                Timber.d("Found tag: 0x${tag.toString(16)}, length: $length")

                when (tag) {
                    TAG_BIOMETRIC_DATA_BLOCK, TAG_IMAGE_DATA -> {
                        // This should contain the image data
                        val imageBytes = ByteArray(length)
                        stream.read(imageBytes)

                        // Check if it's a supported image format
                        val imageType = detectImageType(imageBytes)
                        if (imageType != null) {
                            Timber.d("Found $imageType image data")
                            return imageBytes
                        } else {
                            Timber.w("Data block is not a recognized image format, continuing search...")
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
                            // Skip this field if it's too large or we can't process it
                            if (length <= stream.available()) {
                                stream.skip(length.toLong())
                            }
                        }
                    }
                }
            }

            // If we haven't found it in the structure, search for image magic bytes
            Timber.d("Searching for image magic bytes in raw data...")
            return findImageInRawData(data)

        } catch (e: Exception) {
            Timber.e(e, "Failed to extract image data")
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

        Timber.w("No recognized image format found in data")
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
                Timber.d("Found $formatName magic bytes at offset $i")
                return data.copyOfRange(i, data.size)
            }
        }
        return null
    }

    /**
     * Detects the image type from data.
     * Returns format name if recognized, null otherwise.
     */
    private fun detectImageType(data: ByteArray): String? {
        if (data.size < 4) return null

        // Check JPEG2000 JP2 container
        if (startsWith(data, JPEG2000_JP2_MAGIC)) {
            return "JPEG2000 JP2"
        }

        // Check JPEG2000 codestream
        if (startsWith(data, JPEG2000_CODESTREAM_MAGIC)) {
            return "JPEG2000 codestream"
        }

        // Check regular JPEG
        if (startsWith(data, JPEG_MAGIC)) {
            return "JPEG"
        }

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
     * Decodes JPEG2000 image data to Android Bitmap.
     *
     * Note: Android doesn't natively support JPEG2000.
     * This attempts to decode using BitmapFactory first (for regular JPEG),
     * then falls back to alternative methods.
     */
    private fun decodeJpeg2000(imageData: ByteArray): Bitmap? {
        return try {
            Timber.d("Attempting to decode image (${imageData.size} bytes)...")

            // Try using BitmapFactory first (works for JPEG, PNG, etc.)
            val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)

            if (bitmap != null) {
                Timber.d("Successfully decoded image: ${bitmap.width}x${bitmap.height}")
                return bitmap
            }

            Timber.w("BitmapFactory.decodeByteArray returned null, trying alternative methods")
            tryAlternativeDecoding(imageData)

        } catch (e: Exception) {
            Timber.e(e, "Failed to decode image")
            tryAlternativeDecoding(imageData)
        }
    }

    /**
     * Tries alternative decoding methods if JPEG2000 decoding fails.
     *
     * Some cards might use regular JPEG instead of JPEG2000.
     * Uses BitmapUtils for memory-efficient decoding.
     */
    private fun tryAlternativeDecoding(imageData: ByteArray): Bitmap? {
        return try {
            Timber.d("Trying alternative decoding (regular JPEG)...")

            // Try to decode as regular JPEG with optimization
            val bitmap = BitmapUtils.decodeOptimized(imageData)

            if (bitmap != null) {
                val memSize = BitmapUtils.formatMemorySize(bitmap.byteCount)
                Timber.d("Successfully decoded as regular JPEG: ${bitmap.width}x${bitmap.height}, size: $memSize")
            } else {
                Timber.e("Failed to decode image with alternative methods")
            }

            bitmap

        } catch (e: Exception) {
            Timber.e(e, "Alternative decoding failed")
            null
        }
    }

    /**
     * Reads an ASN.1 tag from the stream.
     */
    private fun readTag(stream: ByteArrayInputStream): Int {
        val firstByte = stream.read()
        if (firstByte == -1) throw IllegalStateException("Unexpected end of stream")

        // Check if this is a multi-byte tag
        return if ((firstByte and 0x1F) == 0x1F) {
            // Multi-byte tag
            var tag = firstByte shl 8
            tag = tag or stream.read()
            tag
        } else {
            // Single byte tag
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
            // Short form: length is in the first byte
            firstByte
        } else {
            // Long form: first byte indicates how many bytes encode the length
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
