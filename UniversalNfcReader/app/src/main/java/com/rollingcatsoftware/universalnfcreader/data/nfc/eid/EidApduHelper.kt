package com.rollingcatsoftware.universalnfcreader.data.nfc.eid

import android.util.Log

/**
 * APDU (Application Protocol Data Unit) Helper for ICAO MRTD (ePassport/eID) commands.
 *
 * Provides utilities to construct and parse APDU commands for Turkish eID card communication.
 * Based on ISO 7816-4 and ICAO Doc 9303.
 *
 * APDU Command Structure: CLA | INS | P1 | P2 | Lc | Data | Le
 * APDU Response Structure: Data | SW1 | SW2
 *
 * Security Note: This class only provides command construction.
 * Actual communication requires BAC authentication and secure messaging.
 */
object EidApduHelper {

    private const val TAG = "EidApduHelper"

    // ICAO MRTD AID (Application Identifier) - Standard for ePassports and eID cards
    val MRTD_AID = byteArrayOf(
        0xA0.toByte(), 0x00, 0x00, 0x02, 0x47, 0x10, 0x01
    )

    // File IDs for Turkish eID (Short File Identifiers)
    object FileIds {
        const val EF_CARD_ACCESS: Byte = 0x1C // Public access
        const val EF_SOD: Byte = 0x1D         // Security Object Document
        const val DG1: Byte = 0x01            // Personal data (requires BAC)
        const val DG2: Byte = 0x02            // Facial image (requires BAC)
        const val DG3: Byte = 0x03            // Fingerprints (restricted)
    }

    // APDU Status Words
    object StatusWord {
        const val SUCCESS = 0x9000
        const val MORE_DATA_AVAILABLE = 0x6100
        const val WRONG_LENGTH = 0x6700
        const val SECURITY_NOT_SATISFIED = 0x6982
        const val AUTH_METHOD_BLOCKED = 0x6983
        const val CONDITIONS_NOT_SATISFIED = 0x6985
        const val WRONG_PARAMETERS = 0x6A86
        const val FILE_NOT_FOUND = 0x6A82
        const val WRONG_PIN_MASK = 0x63C0      // 63CX where X = remaining attempts
    }

    /**
     * Constructs a SELECT command for AID or file selection.
     *
     * @param fileId The file identifier or AID to select
     * @param isAid True if selecting an application, false if selecting a file
     * @return The SELECT APDU command bytes
     */
    fun selectCommand(fileId: ByteArray, isAid: Boolean = false): ByteArray {
        val p1: Byte = if (isAid) 0x04 else 0x02
        val p2: Byte = 0x0C

        return byteArrayOf(
            0x00,                    // CLA: Inter-industry class
            0xA4.toByte(),          // INS: SELECT
            p1,                      // P1: Selection control
            p2,                      // P2: Selection options
            fileId.size.toByte()    // Lc: Length of data
        ) + fileId + byteArrayOf(0x00) // Le: Expected response length
    }

    /**
     * Constructs a SELECT AID command for ICAO MRTD application.
     */
    fun selectMrtdAid(): ByteArray = selectCommand(MRTD_AID, isAid = true)

    /**
     * Constructs a GET CHALLENGE command.
     * Returns 8 random bytes from the card for BAC authentication.
     */
    fun getChallengeCommand(): ByteArray = byteArrayOf(
        0x00,              // CLA
        0x84.toByte(),    // INS: GET CHALLENGE
        0x00,              // P1
        0x00,              // P2
        0x08               // Le: 8 bytes
    )

    /**
     * Constructs an EXTERNAL AUTHENTICATE command for BAC mutual authentication.
     *
     * @param cmdData 40 bytes: E.IFD (32 bytes) + M.IFD (8 bytes)
     */
    fun externalAuthenticateCommand(cmdData: ByteArray): ByteArray = byteArrayOf(
        0x00,                   // CLA
        0x82.toByte(),         // INS: EXTERNAL AUTHENTICATE
        0x00,                   // P1
        0x00,                   // P2
        cmdData.size.toByte()  // Lc
    ) + cmdData + byteArrayOf(0x28) // Le: 40 bytes expected

    /**
     * Constructs a READ BINARY command.
     *
     * @param offset Offset to start reading from
     * @param length Number of bytes to read (0x00 means max, typically 256)
     * @param useSfi True to use Short File Identifier in first read
     * @param sfi Short File Identifier (only used if useSfi is true)
     */
    fun readBinaryCommand(
        offset: Int = 0,
        length: Int = 0,
        useSfi: Boolean = false,
        sfi: Byte = 0
    ): ByteArray {
        val p1 = if (useSfi && offset == 0) {
            (0x80 or sfi.toInt()).toByte()
        } else {
            (offset shr 8).toByte()
        }
        val p2 = (offset and 0xFF).toByte()

        return byteArrayOf(
            0x00,              // CLA
            0xB0.toByte(),    // INS: READ BINARY
            p1,                // P1: Offset high byte or SFI
            p2,                // P2: Offset low byte
            length.toByte()    // Le: Expected response length
        )
    }

    /**
     * Parses APDU response and extracts data and status word.
     *
     * @param response The APDU response bytes
     * @return Pair of (data, statusWord)
     */
    fun parseResponse(response: ByteArray): Pair<ByteArray, Int> {
        if (response.size < 2) {
            Log.e(TAG, "Invalid APDU response: too short")
            return Pair(byteArrayOf(), 0x6F00)
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
     * Checks if status word indicates success.
     */
    fun isSuccess(statusWord: Int): Boolean = statusWord == StatusWord.SUCCESS

    /**
     * Gets remaining PIN attempts from status word (for 63CX responses).
     *
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
     */
    fun getStatusDescription(statusWord: Int): String {
        val remainingAttempts = getRemainingPinAttempts(statusWord)

        return when {
            statusWord == StatusWord.SUCCESS -> "Success"
            remainingAttempts >= 0 -> "Wrong PIN. $remainingAttempts attempt(s) remaining"
            statusWord == StatusWord.SECURITY_NOT_SATISFIED -> "Security condition not satisfied"
            statusWord == StatusWord.AUTH_METHOD_BLOCKED -> "Authentication method blocked"
            statusWord == StatusWord.FILE_NOT_FOUND -> "File not found"
            statusWord == StatusWord.WRONG_PARAMETERS -> "Wrong parameters"
            statusWord == StatusWord.WRONG_LENGTH -> "Wrong length"
            statusWord == StatusWord.CONDITIONS_NOT_SATISFIED -> "Conditions not satisfied"
            else -> String.format("Unknown status: 0x%04X", statusWord)
        }
    }

    /**
     * Converts byte array to hex string for logging.
     */
    fun toHexString(bytes: ByteArray): String =
        bytes.joinToString(" ") { String.format("%02X", it) }

    /**
     * Logs APDU command for debugging.
     */
    fun logCommand(command: ByteArray) {
        Log.d(TAG, "→ APDU: ${toHexString(command)}")
    }

    /**
     * Logs APDU response for debugging.
     */
    fun logResponse(response: ByteArray) {
        val (data, statusWord) = parseResponse(response)
        Log.d(TAG, "← APDU: ${toHexString(response)}")
        Log.d(TAG, "  Status: ${getStatusDescription(statusWord)} (0x${String.format("%04X", statusWord)})")
        if (data.isNotEmpty()) {
            Log.d(TAG, "  Data length: ${data.size} bytes")
        }
    }
}
