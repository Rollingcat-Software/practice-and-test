package com.rollingcatsoftware.universalnfcreader.domain.model

import java.security.SecureRandom
import java.util.Arrays

/**
 * Sealed class representing authentication credentials for different card types.
 *
 * Security Notes:
 * - Credentials are stored in memory only during authentication
 * - Use [clear] to zero out sensitive data after use
 * - Never log or persist credentials in plain text
 */
sealed class AuthenticationData {
    /**
     * Clear sensitive data from memory.
     * Call this after authentication is complete.
     */
    abstract fun clear()

    /**
     * MRZ (Machine Readable Zone) data for BAC authentication.
     * Used by Turkish eID and other MRTD documents.
     *
     * @property documentNumber Document number (up to 9 characters)
     * @property dateOfBirth Date of birth in YYMMDD format
     * @property dateOfExpiry Date of expiry in YYMMDD format
     */
    data class MrzData(
        private var documentNumber: String,
        private var dateOfBirth: String,
        private var dateOfExpiry: String
    ) : AuthenticationData() {

        init {
            require(documentNumber.isNotBlank()) { "Document number cannot be blank" }
            require(dateOfBirth.matches(Regex("\\d{6}"))) { "Date of birth must be YYMMDD format" }
            require(dateOfExpiry.matches(Regex("\\d{6}"))) { "Date of expiry must be YYMMDD format" }
        }

        fun getDocumentNumber(): String = documentNumber
        fun getDateOfBirth(): String = dateOfBirth
        fun getDateOfExpiry(): String = dateOfExpiry

        override fun clear() {
            // Overwrite with random data before clearing
            val random = SecureRandom()
            documentNumber = random.nextInt().toString()
            dateOfBirth = "000000"
            dateOfExpiry = "000000"
        }

        override fun toString(): String =
            "MrzData(documentNumber=***, dateOfBirth=***, dateOfExpiry=***)"
    }

    /**
     * PIN authentication data.
     * Used for cards requiring numeric PIN.
     *
     * @property pin The PIN (typically 4-8 digits)
     */
    data class PinData(
        private var pin: CharArray
    ) : AuthenticationData() {

        init {
            require(pin.isNotEmpty()) { "PIN cannot be empty" }
            require(pin.size in 4..8) { "PIN must be 4-8 digits" }
            require(pin.all { it.isDigit() }) { "PIN must contain only digits" }
        }

        fun getPin(): CharArray = pin.copyOf()

        override fun clear() {
            Arrays.fill(pin, '0')
            pin = CharArray(0)
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is PinData) return false
            return pin.contentEquals(other.pin)
        }

        override fun hashCode(): Int = pin.contentHashCode()

        override fun toString(): String = "PinData(pin=****)"
    }

    /**
     * MIFARE key authentication data.
     * Used for MIFARE Classic cards (Key A or Key B).
     *
     * @property key 6-byte MIFARE key
     * @property keyType Whether this is Key A or Key B
     */
    data class MifareKeyData(
        private var key: ByteArray,
        val keyType: MifareKeyType
    ) : AuthenticationData() {

        init {
            require(key.size == 6) { "MIFARE key must be exactly 6 bytes" }
        }

        fun getKey(): ByteArray = key.copyOf()

        override fun clear() {
            Arrays.fill(key, 0x00.toByte())
            key = ByteArray(0)
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is MifareKeyData) return false
            return key.contentEquals(other.key) && keyType == other.keyType
        }

        override fun hashCode(): Int = 31 * key.contentHashCode() + keyType.hashCode()

        override fun toString(): String = "MifareKeyData(key=******, keyType=$keyType)"

        enum class MifareKeyType { KEY_A, KEY_B }

        companion object {
            /** Factory default key (all 0xFF) */
            val DEFAULT_KEY = byteArrayOf(
                0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte(),
                0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte()
            )

            /** NXP default key (all 0x00) */
            val NXP_DEFAULT_KEY = byteArrayOf(
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            )

            /** MAD key for sector 0 */
            val MAD_KEY = byteArrayOf(
                0xA0.toByte(), 0xA1.toByte(), 0xA2.toByte(),
                0xA3.toByte(), 0xA4.toByte(), 0xA5.toByte()
            )

            /** Common default keys to try */
            val COMMON_KEYS = listOf(
                DEFAULT_KEY,
                NXP_DEFAULT_KEY,
                MAD_KEY,
                byteArrayOf(
                    0xD3.toByte(),
                    0xF7.toByte(),
                    0xD3.toByte(),
                    0xF7.toByte(),
                    0xD3.toByte(),
                    0xF7.toByte()
                ),
                byteArrayOf(
                    0xA0.toByte(),
                    0xB0.toByte(),
                    0xC0.toByte(),
                    0xD0.toByte(),
                    0xE0.toByte(),
                    0xF0.toByte()
                ),
                byteArrayOf(
                    0xB0.toByte(),
                    0xB1.toByte(),
                    0xB2.toByte(),
                    0xB3.toByte(),
                    0xB4.toByte(),
                    0xB5.toByte()
                )
            )
        }
    }

    /**
     * DESFire key authentication data.
     * Used for MIFARE DESFire cards.
     *
     * @property key AES or 3DES key (16 or 24 bytes)
     * @property keyNumber Key slot number (0-13)
     */
    data class DesfireKeyData(
        private var key: ByteArray,
        val keyNumber: Int = 0
    ) : AuthenticationData() {

        init {
            require(key.size == 16 || key.size == 24) { "DESFire key must be 16 (AES) or 24 (3DES) bytes" }
            require(keyNumber in 0..13) { "Key number must be 0-13" }
        }

        fun getKey(): ByteArray = key.copyOf()

        override fun clear() {
            Arrays.fill(key, 0x00.toByte())
            key = ByteArray(0)
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is DesfireKeyData) return false
            return key.contentEquals(other.key) && keyNumber == other.keyNumber
        }

        override fun hashCode(): Int = 31 * key.contentHashCode() + keyNumber

        override fun toString(): String = "DesfireKeyData(key=****, keyNumber=$keyNumber)"

        companion object {
            /** Default DESFire key (all zeros) */
            val DEFAULT_KEY = ByteArray(16) { 0x00 }
        }
    }
}
