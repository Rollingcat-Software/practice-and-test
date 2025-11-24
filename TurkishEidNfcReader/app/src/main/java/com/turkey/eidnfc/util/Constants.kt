package com.turkey.eidnfc.util

/**
 * Application-wide constants.
 */
object Constants {

    /**
     * NFC-related constants.
     */
    object Nfc {
        // Timeouts
        const val CONNECTION_TIMEOUT_MS = 5000
        const val OPERATION_TIMEOUT_MS = 30000

        // PIN
        const val PIN_LENGTH = 6
        const val MIN_PIN_ATTEMPTS_WARNING = 3

        // APDU Status Words
        const val APDU_SUCCESS = 0x9000
        const val APDU_WRONG_PIN_MASK = 0x63C0
        const val APDU_SECURITY_NOT_SATISFIED = 0x6982
        const val APDU_AUTH_BLOCKED = 0x6983
        const val APDU_FILE_NOT_FOUND = 0x6A82
        const val APDU_WRONG_PARAMETERS = 0x6A86
        const val APDU_WRONG_LENGTH = 0x6700

        // File sizes (max expected sizes)
        const val MAX_DG1_SIZE = 10240  // 10KB
        const val MAX_DG2_SIZE = 102400 // 100KB
        const val MAX_SOD_SIZE = 51200  // 50KB

        // Image processing
        const val JPEG2000_MAGIC_BYTES_SIZE = 8
        const val PHOTO_MAX_DIMENSION = 800
    }

    /**
     * UI-related constants.
     */
    object Ui {
        const val DEBOUNCE_DELAY_MS = 300L
        const val SNACKBAR_DURATION_MS = 3000L
        const val ANIMATION_DURATION_MS = 300
        const val SHIMMER_ANIMATION_DURATION_MS = 1000
    }

    /**
     * Turkish eID specific constants.
     */
    object TurkishEid {
        const val TCKN_LENGTH = 11
        const val MRZ_LINE_LENGTH = 30
        const val MRZ_LINES = 3
    }

    /**
     * Logging tags.
     */
    object Tags {
        const val NFC = "NFC"
        const val APDU = "APDU"
        const val PARSER = "Parser"
        const val CRYPTO = "Crypto"
        const val UI = "UI"
    }
}
