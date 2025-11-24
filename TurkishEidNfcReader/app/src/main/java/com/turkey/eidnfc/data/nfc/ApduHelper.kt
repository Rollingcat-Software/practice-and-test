package com.turkey.eidnfc.data.nfc

import timber.log.Timber

/**
 * APDU (Application Protocol Data Unit) Helper class for ISO 7816-4 commands.
 *
 * This class provides utilities to construct and parse APDU commands
 * used to communicate with the Turkish eID card via NFC.
 *
 * APDU Command Structure:
 * CLA | INS | P1 | P2 | Lc | Data | Le
 *
 * APDU Response Structure:
 * Data | SW1 | SW2
 */
object ApduHelper {

    // Turkish eID AID (Application Identifier)
    // A0 00 00 01 67 45 53 49 44
    val TURKISH_EID_AID = byteArrayOf(
        0xA0.toByte(), 0x00, 0x00, 0x01, 0x67,
        0x45, 0x53, 0x49, 0x44
    )

    // File IDs for Turkish eID
    object FileIds {
        val EF_CARD_ACCESS = byteArrayOf(0x01, 0x1C) // Public access
        val EF_SOD = byteArrayOf(0x01, 0x1D)         // Security Object Document
        val DG1 = byteArrayOf(0x01, 0x01)            // Personal data (requires PIN)
        val DG2 = byteArrayOf(0x01, 0x02)            // Facial image (requires PIN)
        val DG3 = byteArrayOf(0x01, 0x03)            // Fingerprints (restricted)
    }

    // APDU Status Words
    object StatusWord {
        const val SUCCESS = 0x9000
        const val WRONG_PIN_MASK = 0x63C0          // 63CX where X = remaining attempts
        const val SECURITY_NOT_SATISFIED = 0x6982
        const val AUTH_METHOD_BLOCKED = 0x6983
        const val FILE_NOT_FOUND = 0x6A82
        const val WRONG_PARAMETERS = 0x6A86
        const val WRONG_LENGTH = 0x6700
        const val CONDITIONS_NOT_SATISFIED = 0x6985
    }

    /**
     * Constructs a SELECT command to select an application or file.
     *
     * @param fileId The file identifier or AID to select
     * @param isAid True if selecting an application, false if selecting a file
     * @return The SELECT APDU command bytes
     */
    fun selectCommand(fileId: ByteArray, isAid: Boolean = false): ByteArray {
        val p1: Byte = if (isAid) 0x04 else 0x02
        val p2: Byte = if (isAid) 0x0C else 0x0C

        return byteArrayOf(
            0x00,                    // CLA: Inter-industry class
            0xA4.toByte(),          // INS: SELECT
            p1,                      // P1: Selection control
            p2,                      // P2: Selection options
            fileId.size.toByte()    // Lc: Length of data
        ) + fileId + byteArrayOf(0x00) // Le: Expected response length
    }

    /**
     * Constructs a SELECT AID command for Turkish eID.
     *
     * @return The SELECT AID APDU command bytes
     */
    fun selectEidAid(): ByteArray {
        return selectCommand(TURKISH_EID_AID, isAid = true)
    }

    /**
     * Constructs a READ BINARY command to read data from selected file.
     *
     * @param offset Offset to start reading from
     * @param length Number of bytes to read (0x00 means max, typically 256)
     * @return The READ BINARY APDU command bytes
     */
    fun readBinaryCommand(offset: Int = 0, length: Int = 0): ByteArray {
        val p1 = (offset shr 8).toByte()
        val p2 = (offset and 0xFF).toByte()

        return byteArrayOf(
            0x00,              // CLA
            0xB0.toByte(),    // INS: READ BINARY
            p1,                // P1: Offset high byte
            p2,                // P2: Offset low byte
            length.toByte()    // Le: Expected response length
        )
    }

    /**
     * Constructs a VERIFY PIN command (PIN authentication).
     *
     * For Turkish eID, PIN1 is 6 digits.
     *
     * @param pin The 6-digit PIN as a string
     * @return The VERIFY APDU command bytes, or null if PIN is invalid
     */
    fun verifyPinCommand(pin: String): ByteArray? {
        // Validate PIN format
        if (!isValidPin(pin)) {
            Timber.e("Invalid PIN format. Must be exactly 6 digits.")
            return null
        }

        // Convert PIN string to bytes
        val pinBytes = pin.toByteArray(Charsets.UTF_8)

        return byteArrayOf(
            0x00,              // CLA
            0x20,              // INS: VERIFY
            0x00,              // P1: No specific meaning for VERIFY
            0x81.toByte(),    // P2: Reference of PIN1
            0x06               // Lc: Length of PIN (6 bytes)
        ) + pinBytes
    }

    /**
     * Validates PIN format.
     *
     * @param pin The PIN to validate
     * @return True if PIN is valid (6 digits), false otherwise
     */
    fun isValidPin(pin: String): Boolean {
        return pin.length == 6 && pin.all { it.isDigit() }
    }

    /**
     * Parses APDU response and extracts status word.
     *
     * @param response The APDU response bytes
     * @return Pair of (data, statusWord)
     */
    fun parseResponse(response: ByteArray): Pair<ByteArray, Int> {
        if (response.size < 2) {
            Timber.e("Invalid APDU response: too short")
            return Pair(byteArrayOf(), 0x6F00) // General error
        }

        val sw1 = response[response.size - 2].toInt() and 0xFF
        val sw2 = response[response.size - 1].toInt() and 0xFF
        val statusWord = (sw1 shl 8) or sw2

        val data = if (response.size > 2) {
            response.copyOfRange(0, response.size - 2)
        } else {
            byteArrayOf()
        }

        return Pair(data, statusWord)
    }

    /**
     * Checks if the APDU response indicates success.
     *
     * @param statusWord The status word from APDU response
     * @return True if success (0x9000), false otherwise
     */
    fun isSuccess(statusWord: Int): Boolean {
        return statusWord == StatusWord.SUCCESS
    }

    /**
     * Gets remaining PIN attempts from status word.
     *
     * @param statusWord The status word from APDU response
     * @return Number of remaining attempts, or -1 if not a wrong PIN response
     */
    fun getRemainingPinAttempts(statusWord: Int): Int {
        return if ((statusWord and 0xFFF0) == StatusWord.WRONG_PIN_MASK) {
            statusWord and 0x000F
        } else {
            -1
        }
    }

    /**
     * Gets a human-readable description of the status word.
     *
     * @param statusWord The status word from APDU response
     * @return Description of the status
     */
    fun getStatusDescription(statusWord: Int): String {
        val remainingAttempts = getRemainingPinAttempts(statusWord)

        return when {
            statusWord == StatusWord.SUCCESS ->
                "Success"
            remainingAttempts >= 0 ->
                "Wrong PIN. $remainingAttempts attempt(s) remaining"
            statusWord == StatusWord.SECURITY_NOT_SATISFIED ->
                "Security condition not satisfied. PIN required"
            statusWord == StatusWord.AUTH_METHOD_BLOCKED ->
                "Authentication method blocked. Card is locked"
            statusWord == StatusWord.FILE_NOT_FOUND ->
                "File or application not found"
            statusWord == StatusWord.WRONG_PARAMETERS ->
                "Wrong parameters P1 or P2"
            statusWord == StatusWord.WRONG_LENGTH ->
                "Wrong length Lc"
            statusWord == StatusWord.CONDITIONS_NOT_SATISFIED ->
                "Conditions of use not satisfied"
            else ->
                String.format("Unknown status: 0x%04X", statusWord)
        }
    }

    /**
     * Converts byte array to hex string for logging.
     *
     * @param bytes The byte array to convert
     * @return Hex string representation
     */
    fun toHexString(bytes: ByteArray): String {
        return bytes.joinToString(" ") { String.format("%02X", it) }
    }

    /**
     * Logs APDU command for debugging.
     *
     * @param command The APDU command bytes
     */
    fun logCommand(command: ByteArray) {
        Timber.d("→ APDU Command: ${toHexString(command)}")
    }

    /**
     * Logs APDU response for debugging.
     *
     * @param response The APDU response bytes
     */
    fun logResponse(response: ByteArray) {
        val (data, statusWord) = parseResponse(response)
        Timber.d("← APDU Response: ${toHexString(response)}")
        Timber.d("  Status: ${getStatusDescription(statusWord)} (0x${String.format("%04X", statusWord)})")
        if (data.isNotEmpty()) {
            Timber.d("  Data length: ${data.size} bytes")
        }
    }

    /**
     * Reads complete file content by handling chunked responses.
     * This is useful when file size exceeds single APDU response limit.
     *
     * @param transceive Function to send APDU and receive response
     * @return Complete file data, or null if error
     */
    suspend fun readCompleteFile(
        transceive: suspend (ByteArray) -> ByteArray
    ): ByteArray? {
        val allData = mutableListOf<Byte>()
        var offset = 0

        while (true) {
            val command = readBinaryCommand(offset, 0)
            logCommand(command)

            val response = transceive(command)
            logResponse(response)

            val (data, statusWord) = parseResponse(response)

            if (!isSuccess(statusWord)) {
                if (offset == 0) {
                    // First read failed
                    Timber.e("Failed to read file: ${getStatusDescription(statusWord)}")
                    return null
                } else {
                    // Subsequent read failed - might have reached end of file
                    break
                }
            }

            if (data.isEmpty()) {
                // No more data
                break
            }

            allData.addAll(data.toList())
            offset += data.size

            // Check if we got less than requested (indicates end of file)
            if (data.size < 256) {
                break
            }
        }

        return allData.toByteArray()
    }
}
