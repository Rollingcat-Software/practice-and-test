package com.turkey.eidnfc.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import timber.log.Timber
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import java.io.ByteArrayOutputStream

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

    // JPEG2000 magic bytes
    private val JPEG2000_MAGIC_BYTES = byteArrayOf(
        0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20
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
                        // This should contain the JPEG2000 data
                        val imageBytes = ByteArray(length)
                        stream.read(imageBytes)

                        // Check if it's JPEG2000
                        if (isJpeg2000(imageBytes)) {
                            Timber.d("Found JPEG2000 image data")
                            return imageBytes
                        } else {
                            Timber.w("Data block is not JPEG2000, continuing search...")
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

            // If we haven't found it in the structure, search for JPEG2000 magic bytes
            Timber.d("Searching for JPEG2000 magic bytes in raw data...")
            return findJpeg2000InRawData(data)

        } catch (e: Exception) {
            Timber.e(e, "Failed to extract image data")
            return null
        }
    }

    /**
     * Searches for JPEG2000 magic bytes in raw data.
     */
    private fun findJpeg2000InRawData(data: ByteArray): ByteArray? {
        for (i in 0 until data.size - JPEG2000_MAGIC_BYTES.size) {
            var match = true
            for (j in JPEG2000_MAGIC_BYTES.indices) {
                if (data[i + j] != JPEG2000_MAGIC_BYTES[j]) {
                    match = false
                    break
                }
            }
            if (match) {
                Timber.d("Found JPEG2000 magic bytes at offset $i")
                return data.copyOfRange(i, data.size)
            }
        }
        return null
    }

    /**
     * Checks if data starts with JPEG2000 magic bytes.
     */
    private fun isJpeg2000(data: ByteArray): Boolean {
        if (data.size < JPEG2000_MAGIC_BYTES.size) {
            return false
        }
        for (i in JPEG2000_MAGIC_BYTES.indices) {
            if (data[i] != JPEG2000_MAGIC_BYTES[i]) {
                return false
            }
        }
        return true
    }

    /**
     * Decodes JPEG2000 image data to Android Bitmap.
     *
     * This uses JAI ImageIO library for JPEG2000 decoding.
     */
    private fun decodeJpeg2000(imageData: ByteArray): Bitmap? {
        return try {
            Timber.d("Attempting to decode JPEG2000 image...")

            // Try using JAI ImageIO for JPEG2000
            val inputStream = ByteArrayInputStream(imageData)
            val bufferedImage = ImageIO.read(inputStream)

            if (bufferedImage == null) {
                Timber.e("ImageIO.read returned null")
                return tryAlternativeDecoding(imageData)
            }

            // Convert BufferedImage to Android Bitmap
            val width = bufferedImage.width
            val height = bufferedImage.height
            Timber.d("Decoded image: ${width}x${height}")

            val pixels = IntArray(width * height)
            bufferedImage.getRGB(0, 0, width, height, pixels, 0, width)

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

            Timber.d("Successfully decoded JPEG2000 to Bitmap")
            bitmap

        } catch (e: Exception) {
            Timber.e(e, "Failed to decode JPEG2000 with JAI ImageIO")
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
