package com.rollingcatsoftware.universalnfcreader.data.nfc.eid

import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureByteArray
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.secureClear
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.secureWipe
import java.io.Closeable
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * BAC (Basic Access Control) Authentication for ICAO MRTD documents.
 *
 * BAC uses MRZ (Machine Readable Zone) data to derive encryption keys:
 * - Document number + check digit
 * - Date of birth (YYMMDD) + check digit
 * - Date of expiry (YYMMDD) + check digit
 *
 * Based on ICAO Doc 9303 Part 11.
 *
 * COMPLIANCE: se-checklist.md
 * - Section 1.1: "Clear PIN bytes from memory after use" - Uses SecureByteArray
 * - Section 4.1: "Use SecureRandom for any cryptographic operations"
 * - Section 4.2: "Never use predictable random sources"
 *
 * Security Note: All sensitive key material is wrapped in SecureByteArray
 * and automatically cleared when no longer needed.
 */
class BacAuthentication {

    companion object {
        private const val TAG = "BacAuthentication"

        // Constants for key derivation (ICAO Doc 9303)
        private val KENC_CONSTANT = byteArrayOf(0x00, 0x00, 0x00, 0x01)
        private val KMAC_CONSTANT = byteArrayOf(0x00, 0x00, 0x00, 0x02)

        // Zero IV for 3DES CBC
        private val ZERO_IV = ByteArray(8)
    }

    /**
     * MRZ key material for BAC authentication.
     *
     * @param documentNumber Document number (up to 9 characters)
     * @param dateOfBirth Date of birth in YYMMDD format
     * @param dateOfExpiry Date of expiry in YYMMDD format
     */
    data class MrzData(
        val documentNumber: String,
        val dateOfBirth: String,
        val dateOfExpiry: String
    ) {
        init {
            require(documentNumber.length in 1..9) { "Document number must be 1-9 characters" }
            require(dateOfBirth.length == 6 && dateOfBirth.all { it.isDigit() }) {
                "Date of birth must be 6 digits (YYMMDD)"
            }
            require(dateOfExpiry.length == 6 && dateOfExpiry.all { it.isDigit() }) {
                "Date of expiry must be 6 digits (YYMMDD)"
            }
        }

        /**
         * Clears sensitive data from memory.
         * Note: In Kotlin data classes, this creates a new instance with cleared values.
         */
        fun clear(): MrzData = MrzData("X".repeat(9), "000000", "000000")
    }

    /**
     * Session keys derived from BAC authentication.
     *
     * COMPLIANCE: Implements Closeable to ensure secure cleanup.
     * Use with Kotlin's use {} block for automatic cleanup.
     */
    class SessionKeys(
        encryptionKey: ByteArray,
        macKey: ByteArray,
        sendSequenceCounter: ByteArray
    ) : Closeable {
        // Wrap keys in SecureByteArray for secure memory management
        @PublishedApi internal val _encryptionKey = SecureByteArray.wrap(encryptionKey)
        @PublishedApi internal val _macKey = SecureByteArray.wrap(macKey)
        @PublishedApi internal val _sendSequenceCounter = SecureByteArray.wrap(sendSequenceCounter)

        /**
         * Gets a copy of the encryption key.
         * Caller is responsible for clearing the returned array.
         */
        val encryptionKey: ByteArray
            get() = _encryptionKey.toByteArray()

        /**
         * Gets a copy of the MAC key.
         * Caller is responsible for clearing the returned array.
         */
        val macKey: ByteArray
            get() = _macKey.toByteArray()

        /**
         * Gets a copy of the send sequence counter.
         * Caller is responsible for clearing the returned array.
         */
        val sendSequenceCounter: ByteArray
            get() = _sendSequenceCounter.toByteArray()

        /**
         * Performs an operation with direct access to the encryption key.
         * Preferred over [encryptionKey] as it doesn't create copies.
         */
        inline fun <R> withEncryptionKey(block: (ByteArray) -> R): R =
            _encryptionKey.withData(block)

        /**
         * Performs an operation with direct access to the MAC key.
         * Preferred over [macKey] as it doesn't create copies.
         */
        inline fun <R> withMacKey(block: (ByteArray) -> R): R =
            _macKey.withData(block)

        /**
         * Performs an operation with direct access to the SSC.
         */
        inline fun <R> withSsc(block: (ByteArray) -> R): R =
            _sendSequenceCounter.withData(block)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is SessionKeys) return false
            return _encryptionKey == other._encryptionKey &&
                    _macKey == other._macKey &&
                    _sendSequenceCounter == other._sendSequenceCounter
        }

        override fun hashCode(): Int {
            var result = _encryptionKey.hashCode()
            result = 31 * result + _macKey.hashCode()
            result = 31 * result + _sendSequenceCounter.hashCode()
            return result
        }

        /**
         * Securely clears all key material from memory.
         * Called automatically when used with use {} block.
         */
        override fun close() {
            _encryptionKey.close()
            _macKey.close()
            _sendSequenceCounter.close()
        }

        /**
         * Alias for close() for backward compatibility.
         */
        fun clear() = close()
    }

    /**
     * Derives the base encryption and MAC keys from MRZ data.
     *
     * COMPLIANCE: All intermediate values are securely cleared after use.
     *
     * @param mrzData MRZ key material
     * @return Pair of (Kenc, Kmac) base keys - caller must clear these after use
     */
    fun deriveKeys(mrzData: MrzData): Pair<ByteArray, ByteArray> {
        // Build MRZ_information string: docNo + checkDigit + dob + checkDigit + doe + checkDigit
        val docNoWithCheck = mrzData.documentNumber.padEnd(9, '<') +
                calculateCheckDigit(mrzData.documentNumber.padEnd(9, '<'))
        val dobWithCheck = mrzData.dateOfBirth + calculateCheckDigit(mrzData.dateOfBirth)
        val doeWithCheck = mrzData.dateOfExpiry + calculateCheckDigit(mrzData.dateOfExpiry)

        val mrzInfo = docNoWithCheck + dobWithCheck + doeWithCheck
        SecureLogger.logMrzData(TAG, mrzData.documentNumber, mrzData.dateOfBirth, mrzData.dateOfExpiry)

        // Calculate SHA-1 hash and take first 16 bytes as Kseed
        val sha1 = MessageDigest.getInstance("SHA-1")
        val mrzInfoBytes = mrzInfo.toByteArray(Charsets.UTF_8)
        val hash = sha1.digest(mrzInfoBytes)
        val kSeed = hash.copyOfRange(0, 16)

        SecureLogger.logHex(TAG, "Kseed derived", kSeed)

        // Derive Kenc and Kmac
        val kEnc = deriveKey(kSeed, KENC_CONSTANT)
        val kMac = deriveKey(kSeed, KMAC_CONSTANT)

        SecureLogger.logHex(TAG, "Kenc derived", kEnc)
        SecureLogger.logHex(TAG, "Kmac derived", kMac)

        // Securely clear all intermediate values
        kSeed.secureWipe()
        hash.secureClear()
        mrzInfoBytes.secureClear()

        return Pair(kEnc, kMac)
    }

    /**
     * Derives a 3DES key from seed and constant.
     */
    private fun deriveKey(kSeed: ByteArray, constant: ByteArray): ByteArray {
        val sha1 = MessageDigest.getInstance("SHA-1")
        val hash = sha1.digest(kSeed + constant)

        // Take first 16 bytes and adjust parity bits for DES
        val key = hash.copyOfRange(0, 16)
        adjustParityBits(key)

        // Expand to 24-byte 3DES key (K1, K2, K1)
        return key.copyOfRange(0, 8) + key.copyOfRange(8, 16) + key.copyOfRange(0, 8)
    }

    /**
     * Adjusts odd parity bits for DES key.
     */
    private fun adjustParityBits(key: ByteArray) {
        for (i in key.indices) {
            var b = key[i].toInt() and 0xFE
            var parity = 0
            for (j in 0..6) {
                parity = parity xor ((b shr j) and 1)
            }
            key[i] = (b or (1 - parity)).toByte()
        }
    }

    /**
     * Calculates MRZ check digit according to ICAO Doc 9303.
     */
    fun calculateCheckDigit(data: String): Char {
        val weights = intArrayOf(7, 3, 1)
        var sum = 0

        for (i in data.indices) {
            val c = data[i]
            val value = when {
                c in '0'..'9' -> c - '0'
                c in 'A'..'Z' -> c - 'A' + 10
                c == '<' -> 0
                else -> 0
            }
            sum += value * weights[i % 3]
        }

        return ('0' + (sum % 10))
    }

    /**
     * Performs BAC mutual authentication.
     *
     * COMPLIANCE: All intermediate cryptographic values are securely cleared.
     *
     * @param kEnc Encryption key
     * @param kMac MAC key
     * @param rndIcc Random challenge from card (8 bytes)
     * @param transceive Function to send APDU commands
     * @return SessionKeys if successful, null otherwise
     */
    suspend fun performMutualAuthentication(
        kEnc: ByteArray,
        kMac: ByteArray,
        rndIcc: ByteArray,
        transceive: suspend (ByteArray) -> ByteArray
    ): SessionKeys? {
        // Declare all intermediate values for cleanup in finally block
        var rndIfd: ByteArray? = null
        var kIfd: ByteArray? = null
        var s: ByteArray? = null
        var eIfd: ByteArray? = null
        var mIfd: ByteArray? = null
        var r: ByteArray? = null
        var kIcc: ByteArray? = null
        var kSeedSession: ByteArray? = null

        try {
            // Generate IFD random and key material using SecureRandom
            rndIfd = generateSecureRandom(8)
            kIfd = generateSecureRandom(16)

            SecureLogger.logHex(TAG, "RND.ICC", rndIcc)
            SecureLogger.logHex(TAG, "RND.IFD", rndIfd)
            SecureLogger.logHex(TAG, "K.IFD", kIfd)

            // S = RND.IFD || RND.ICC || K.IFD
            s = rndIfd + rndIcc + kIfd

            // Encrypt S with 3DES
            eIfd = encrypt3DES(s, kEnc)
            SecureLogger.logHex(TAG, "E.IFD", eIfd)

            // Calculate MAC
            mIfd = calculateMAC(eIfd, kMac)
            SecureLogger.logHex(TAG, "M.IFD", mIfd)

            // Build EXTERNAL AUTHENTICATE command
            val cmdData = eIfd + mIfd
            val command = EidApduHelper.externalAuthenticateCommand(cmdData)

            SecureLogger.logApduCommand(TAG, command)
            val response = transceive(command)
            SecureLogger.logApduResponse(TAG, response)

            val (data, statusWord) = EidApduHelper.parseResponse(response)

            if (!EidApduHelper.isSuccess(statusWord)) {
                SecureLogger.e(
                    TAG,
                    "BAC authentication failed: ${EidApduHelper.getStatusDescription(statusWord)}"
                )
                return null
            }

            if (data.size != 40) {
                SecureLogger.e(TAG, "Invalid response length: ${data.size}, expected 40")
                return null
            }

            // Parse response: E.ICC (32 bytes) || M.ICC (8 bytes)
            val eIcc = data.copyOfRange(0, 32)
            val mIcc = data.copyOfRange(32, 40)

            // Verify MAC
            val calculatedMac = calculateMAC(eIcc, kMac)
            if (!calculatedMac.contentEquals(mIcc)) {
                SecureLogger.e(TAG, "MAC verification failed")
                calculatedMac.secureClear()
                return null
            }
            calculatedMac.secureClear()

            // Decrypt E.ICC to get R = RND.ICC || RND.IFD || K.ICC
            r = decrypt3DES(eIcc, kEnc)
            val rndIccResponse = r.copyOfRange(0, 8)
            val rndIfdResponse = r.copyOfRange(8, 16)
            kIcc = r.copyOfRange(16, 32)

            // Verify RND.IFD matches what we sent (constant-time comparison)
            if (!constantTimeEquals(rndIfdResponse, rndIfd)) {
                SecureLogger.e(TAG, "RND.IFD verification failed")
                rndIccResponse.secureClear()
                rndIfdResponse.secureClear()
                return null
            }

            // Verify RND.ICC matches challenge (constant-time comparison)
            if (!constantTimeEquals(rndIccResponse, rndIcc)) {
                SecureLogger.e(TAG, "RND.ICC verification failed")
                rndIccResponse.secureClear()
                rndIfdResponse.secureClear()
                return null
            }

            rndIccResponse.secureClear()
            rndIfdResponse.secureClear()

            // Derive session keys: Kseed = K.IFD XOR K.ICC
            kSeedSession = ByteArray(16)
            for (i in 0..15) {
                kSeedSession!![i] = (kIfd[i].toInt() xor kIcc!![i].toInt()).toByte()
            }

            val ksEnc = deriveKey(kSeedSession!!, KENC_CONSTANT)
            val ksMac = deriveKey(kSeedSession!!, KMAC_CONSTANT)

            // Initial SSC = RND.ICC[4:8] || RND.IFD[4:8]
            val ssc = rndIcc.copyOfRange(4, 8) + rndIfd.copyOfRange(4, 8)

            SecureLogger.logHex(TAG, "Session KS.ENC", ksEnc)
            SecureLogger.logHex(TAG, "Session KS.MAC", ksMac)
            SecureLogger.logHex(TAG, "Initial SSC", ssc)

            SecureLogger.d(TAG, "BAC mutual authentication successful")
            return SessionKeys(ksEnc, ksMac, ssc)

        } catch (e: Exception) {
            SecureLogger.e(TAG, "BAC authentication error", e)
            return null
        } finally {
            // Securely clear ALL intermediate sensitive values
            rndIfd?.secureWipe()
            kIfd?.secureWipe()
            s?.secureWipe()
            eIfd?.secureClear()
            mIfd?.secureClear()
            r?.secureWipe()
            kIcc?.secureWipe()
            kSeedSession?.secureWipe()
        }
    }

    /**
     * Constant-time byte array comparison to prevent timing attacks.
     *
     * COMPLIANCE: se-checklist.md - Security consideration for cryptographic operations
     */
    private fun constantTimeEquals(a: ByteArray, b: ByteArray): Boolean {
        if (a.size != b.size) return false
        var result = 0
        for (i in a.indices) {
            result = result or (a[i].toInt() xor b[i].toInt())
        }
        return result == 0
    }

    /**
     * Encrypts data using 3DES CBC with zero IV.
     */
    private fun encrypt3DES(data: ByteArray, key: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("DESede/CBC/NoPadding")
        val keySpec = SecretKeySpec(key, "DESede")
        val ivSpec = IvParameterSpec(ZERO_IV)
        cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec)
        return cipher.doFinal(data)
    }

    /**
     * Decrypts data using 3DES CBC with zero IV.
     */
    private fun decrypt3DES(data: ByteArray, key: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("DESede/CBC/NoPadding")
        val keySpec = SecretKeySpec(key, "DESede")
        val ivSpec = IvParameterSpec(ZERO_IV)
        cipher.init(Cipher.DECRYPT_MODE, keySpec, ivSpec)
        return cipher.doFinal(data)
    }

    /**
     * Calculates ISO 9797-1 MAC Algorithm 3 (Retail MAC) with DES and 3DES.
     */
    private fun calculateMAC(data: ByteArray, key: ByteArray): ByteArray {
        // Pad data to multiple of 8 bytes (ISO 9797-1 padding method 2)
        val padded = padISO9797(data)

        val key1 = key.copyOfRange(0, 8)
        val key2 = key.copyOfRange(8, 16)

        var h = ByteArray(8)

        // Process all but last block with DES CBC
        val numBlocks = padded.size / 8
        for (i in 0 until numBlocks - 1) {
            val block = padded.copyOfRange(i * 8, (i + 1) * 8)
            for (j in 0..7) {
                h[j] = (h[j].toInt() xor block[j].toInt()).toByte()
            }
            h = encryptDES(h, key1)
        }

        // XOR with last block
        val lastBlock = padded.copyOfRange((numBlocks - 1) * 8, numBlocks * 8)
        for (j in 0..7) {
            h[j] = (h[j].toInt() xor lastBlock[j].toInt()).toByte()
        }

        // Final: encrypt with K1, decrypt with K2, encrypt with K1 (3DES)
        h = encryptDES(h, key1)
        h = decryptDES(h, key2)
        h = encryptDES(h, key1)

        return h
    }

    /**
     * Pads data according to ISO 9797-1 padding method 2.
     */
    private fun padISO9797(data: ByteArray): ByteArray {
        val padLength = 8 - (data.size % 8)
        val padded = ByteArray(data.size + padLength)
        System.arraycopy(data, 0, padded, 0, data.size)
        padded[data.size] = 0x80.toByte()
        return padded
    }

    private fun encryptDES(data: ByteArray, key: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("DES/ECB/NoPadding")
        val keySpec = SecretKeySpec(key, "DES")
        cipher.init(Cipher.ENCRYPT_MODE, keySpec)
        return cipher.doFinal(data)
    }

    private fun decryptDES(data: ByteArray, key: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("DES/ECB/NoPadding")
        val keySpec = SecretKeySpec(key, "DES")
        cipher.init(Cipher.DECRYPT_MODE, keySpec)
        return cipher.doFinal(data)
    }

    /**
     * Generates cryptographically secure random bytes.
     */
    private fun generateSecureRandom(length: Int): ByteArray {
        val random = SecureRandom()
        val bytes = ByteArray(length)
        random.nextBytes(bytes)
        return bytes
    }
}
