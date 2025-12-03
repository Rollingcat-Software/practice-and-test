package com.rollingcatsoftware.universalnfcreader.util

/**
 * Constants for NFC card operations.
 *
 * Security Note: No cryptographic keys should be stored here.
 * Only public identifiers and command templates.
 */
object Constants {

    // ==================== AIDs (Application Identifiers) ====================

    /** Turkish eID AID (MRTD/ICAO) */
    val AID_TURKISH_EID = byteArrayOf(
        0xA0.toByte(), 0x00, 0x00, 0x02, 0x47, 0x10, 0x01
    )

    /** MRTD Application AID (ISO 7816) */
    val AID_MRTD = byteArrayOf(
        0xA0.toByte(), 0x00, 0x00, 0x02, 0x47, 0x10, 0x01
    )

    /** MIFARE DESFire AID prefix */
    val AID_DESFIRE_PREFIX = byteArrayOf(
        0xD2.toByte(), 0x76, 0x00, 0x00, 0x85.toByte(), 0x01, 0x00
    )

    // ==================== File IDs ====================

    /** MRTD Data Groups */
    object MrtdFiles {
        const val EF_COM = 0x011E
        const val EF_SOD = 0x011D
        const val EF_DG1 = 0x0101 // MRZ data
        const val EF_DG2 = 0x0102 // Facial image
        const val EF_DG3 = 0x0103 // Fingerprints (restricted)
        const val EF_DG14 = 0x010E // Security options
        const val EF_CARD_ACCESS = 0x011C
    }

    // ==================== APDU Commands ====================

    /** SELECT command template */
    const val INS_SELECT = 0xA4.toByte()
    const val INS_READ_BINARY = 0xB0.toByte()
    const val INS_GET_CHALLENGE = 0x84.toByte()
    const val INS_EXTERNAL_AUTHENTICATE = 0x82.toByte()
    const val INS_INTERNAL_AUTHENTICATE = 0x88.toByte()
    const val INS_VERIFY = 0x20.toByte()

    /** P1-P2 parameters */
    const val P1_SELECT_BY_DF_NAME = 0x04.toByte()
    const val P1_SELECT_BY_EF_ID = 0x02.toByte()
    const val P2_FIRST_OR_ONLY = 0x0C.toByte()

    // ==================== Status Words ====================

    object StatusWord {
        const val SUCCESS = 0x9000

        // Warning statuses
        const val EOF_REACHED = 0x6282
        const val FILE_INVALIDATED = 0x6283
        const val WRONG_LENGTH = 0x6700

        // Error statuses
        const val SECURITY_NOT_SATISFIED = 0x6982
        const val AUTH_METHOD_BLOCKED = 0x6983
        const val REFERENCED_DATA_INVALID = 0x6984
        const val CONDITIONS_NOT_SATISFIED = 0x6985
        const val WRONG_DATA = 0x6A80
        const val FILE_NOT_FOUND = 0x6A82
        const val RECORD_NOT_FOUND = 0x6A83
        const val WRONG_P1P2 = 0x6A86
        const val INS_NOT_SUPPORTED = 0x6D00
        const val CLA_NOT_SUPPORTED = 0x6E00

        // PIN statuses (63CX where X = attempts remaining)
        const val WRONG_PIN_MASK = 0x63C0

        fun getRemainingAttempts(sw: Int): Int? {
            return if ((sw and 0xFFF0) == WRONG_PIN_MASK) sw and 0x000F else null
        }
    }

    // ==================== DESFire Commands ====================

    object DESFire {
        const val CMD_GET_VERSION = 0x60.toByte()
        const val CMD_GET_APPLICATION_IDS = 0x6A.toByte()
        const val CMD_SELECT_APPLICATION = 0x5A.toByte()
        const val CMD_GET_FILE_IDS = 0x6F.toByte()
        const val CMD_GET_FILE_SETTINGS = 0xF5.toByte()
        const val CMD_READ_DATA = 0xBD.toByte()
        const val CMD_READ_RECORDS = 0xBB.toByte()
        const val CMD_GET_VALUE = 0x6C.toByte()
        const val CMD_GET_FREE_MEMORY = 0x6E.toByte()
        const val CMD_AUTHENTICATE = 0x0A.toByte()
        const val CMD_AUTHENTICATE_ISO = 0x1A.toByte()
        const val CMD_AUTHENTICATE_AES = 0xAA.toByte()
        const val CMD_ADDITIONAL_FRAME = 0xAF.toByte()

        // Status codes
        const val STATUS_OK = 0x00.toByte()
        const val STATUS_NO_CHANGES = 0x0C.toByte()
        const val STATUS_OUT_OF_MEMORY = 0x0E.toByte()
        const val STATUS_ILLEGAL_COMMAND = 0x1C.toByte()
        const val STATUS_INTEGRITY_ERROR = 0x1E.toByte()
        const val STATUS_NO_SUCH_KEY = 0x40.toByte()
        const val STATUS_LENGTH_ERROR = 0x7E.toByte()
        const val STATUS_PERMISSION_DENIED = 0x9D.toByte()
        const val STATUS_PARAMETER_ERROR = 0x9E.toByte()
        const val STATUS_APPLICATION_NOT_FOUND = 0xA0.toByte()
        const val STATUS_AUTHENTICATION_ERROR = 0xAE.toByte()
        const val STATUS_ADDITIONAL_FRAME = 0xAF.toByte()
        const val STATUS_BOUNDARY_ERROR = 0xBE.toByte()
        const val STATUS_PICC_INTEGRITY_ERROR = 0xC1.toByte()
        const val STATUS_COMMAND_ABORTED = 0xCA.toByte()
        const val STATUS_PICC_DISABLED_ERROR = 0xCD.toByte()
        const val STATUS_COUNT_ERROR = 0xCE.toByte()
        const val STATUS_DUPLICATE_ERROR = 0xDE.toByte()
        const val STATUS_EEPROM_ERROR = 0xEE.toByte()
        const val STATUS_FILE_NOT_FOUND = 0xF0.toByte()
        const val STATUS_FILE_INTEGRITY_ERROR = 0xF1.toByte()

        // Application IDs
        val PICC_APPLICATION = byteArrayOf(0x00, 0x00, 0x00)
    }

    // ==================== Timeouts ====================

    object Timeout {
        const val DEFAULT_MS = 5000
        const val SHORT_MS = 2000
        const val LONG_MS = 10000
        const val AUTH_MS = 8000
    }

    // ==================== NXP Manufacturer Code ====================

    const val NXP_MANUFACTURER_CODE = 0x04.toByte()

    // ==================== Card Size Constants ====================

    object MifareClassic {
        const val BLOCK_SIZE = 16
        const val SECTORS_1K = 16
        const val SECTORS_4K = 40
        const val BLOCKS_PER_SECTOR = 4
        const val BLOCKS_PER_SECTOR_4K_LARGE = 16 // Sectors 32-39 have 16 blocks
    }

    object MifareUltralight {
        const val PAGE_SIZE = 4
        const val PAGES_ULTRALIGHT = 16
        const val PAGES_ULTRALIGHT_C = 48
        const val PAGES_NTAG_213 = 45
        const val PAGES_NTAG_215 = 135
        const val PAGES_NTAG_216 = 231
    }
}
