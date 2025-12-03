package com.turkey.eidnfc.data.nfc

import timber.log.Timber
import java.security.MessageDigest
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
 */
class BacAuthentication {

    companion object {
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
    }

    /**
     * Session keys derived from BAC authentication.
     */
    data class SessionKeys(
        val encryptionKey: ByteArray,
        val macKey: ByteArray,
        val sendSequenceCounter: ByteArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as SessionKeys
            return encryptionKey.contentEquals(other.encryptionKey) &&
                macKey.contentEquals(other.macKey) &&
                sendSequenceCounter.contentEquals(other.sendSequenceCounter)
        }

        override fun hashCode(): Int {
            var result = encryptionKey.contentHashCode()
            result = 31 * result + macKey.contentHashCode()
            result = 31 * result + sendSequenceCounter.contentHashCode()
            return result
        }
    }

    /**
     * Derives the base encryption and MAC keys from MRZ data.
     *
     * @param mrzData MRZ key material
     * @return Pair of (Kenc, Kmac) base keys
     */
    fun deriveKeys(mrzData: MrzData): Pair<ByteArray, ByteArray> {
        // Build MRZ_information string: docNo + checkDigit + dob + checkDigit + doe + checkDigit
        val docNoWithCheck =
            mrzData.documentNumber.padEnd(9, '<') + calculateCheckDigit(mrzData.documentNumber.padEnd(9, '<'))
        val dobWithCheck = mrzData.dateOfBirth + calculateCheckDigit(mrzData.dateOfBirth)
        val doeWithCheck = mrzData.dateOfExpiry + calculateCheckDigit(mrzData.dateOfExpiry)

        val mrzInfo = docNoWithCheck + dobWithCheck + doeWithCheck
        Timber.d("MRZ Info: $mrzInfo")

        // Calculate SHA-1 hash and take first 16 bytes as Kseed
        val sha1 = MessageDigest.getInstance("SHA-1")
        val hash = sha1.digest(mrzInfo.toByteArray(Charsets.UTF_8))
        val kSeed = hash.copyOfRange(0, 16)
        Timber.d("Kseed: ${ApduHelper.toHexString(kSeed)}")

        // Derive Kenc and Kmac
        val kEnc = deriveKey(kSeed, KENC_CONSTANT)
        val kMac = deriveKey(kSeed, KMAC_CONSTANT)

        Timber.d("Kenc: ${ApduHelper.toHexString(kEnc)}")
        Timber.d("Kmac: ${ApduHelper.toHexString(kMac)}")

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
        try {
            // Generate IFD random and key material
            val rndIfd = generateRandom(8)
            val kIfd = generateRandom(16)

            Timber.d("RND.ICC: ${ApduHelper.toHexString(rndIcc)}")
            Timber.d("RND.IFD: ${ApduHelper.toHexString(rndIfd)}")
            Timber.d("K.IFD: ${ApduHelper.toHexString(kIfd)}")

            // S = RND.IFD || RND.ICC || K.IFD
            val s = rndIfd + rndIcc + kIfd

            // Encrypt S with 3DES
            val eIfd = encrypt3DES(s, kEnc)
            Timber.d("E.IFD: ${ApduHelper.toHexString(eIfd)}")

            // Calculate MAC
            val mIfd = calculateMAC(eIfd, kMac)
            Timber.d("M.IFD: ${ApduHelper.toHexString(mIfd)}")

            // Build EXTERNAL AUTHENTICATE command
            val cmdData = eIfd + mIfd
            val command = byteArrayOf(
                0x00,                   // CLA
                0x82.toByte(),         // INS: EXTERNAL AUTHENTICATE
                0x00,                   // P1
                0x00,                   // P2
                cmdData.size.toByte()  // Lc
            ) + cmdData + byteArrayOf(0x28) // Le: 40 bytes expected

            ApduHelper.logCommand(command)
            val response = transceive(command)
            ApduHelper.logResponse(response)

            val (data, statusWord) = ApduHelper.parseResponse(response)

            if (!ApduHelper.isSuccess(statusWord)) {
                Timber.e("BAC authentication failed: ${ApduHelper.getStatusDescription(statusWord)}")
                return null
            }

            if (data.size != 40) {
                Timber.e("Invalid response length: ${data.size}, expected 40")
                return null
            }

            // Parse response: E.ICC (32 bytes) || M.ICC (8 bytes)
            val eIcc = data.copyOfRange(0, 32)
            val mIcc = data.copyOfRange(32, 40)

            // Verify MAC
            val calculatedMac = calculateMAC(eIcc, kMac)
            if (!calculatedMac.contentEquals(mIcc)) {
                Timber.e("MAC verification failed")
                return null
            }

            // Decrypt E.ICC to get R = RND.ICC || RND.IFD || K.ICC
            val r = decrypt3DES(eIcc, kEnc)
            val rndIccResponse = r.copyOfRange(0, 8)
            val rndIfdResponse = r.copyOfRange(8, 16)
            val kIcc = r.copyOfRange(16, 32)

            // Verify RND.IFD matches what we sent
            if (!rndIfdResponse.contentEquals(rndIfd)) {
                Timber.e("RND.IFD verification failed")
                return null
            }

            // Verify RND.ICC matches challenge
            if (!rndIccResponse.contentEquals(rndIcc)) {
                Timber.e("RND.ICC verification failed")
                return null
            }

            // Derive session keys: Kseed = K.IFD XOR K.ICC
            val kSeedSession = ByteArray(16)
            for (i in 0..15) {
                kSeedSession[i] = (kIfd[i].toInt() xor kIcc[i].toInt()).toByte()
            }

            val ksEnc = deriveKey(kSeedSession, KENC_CONSTANT)
            val ksMac = deriveKey(kSeedSession, KMAC_CONSTANT)

            // Initial SSC = RND.ICC[4:8] || RND.IFD[4:8]
            val ssc = rndIcc.copyOfRange(4, 8) + rndIfd.copyOfRange(4, 8)

            Timber.d("Session KS.ENC: ${ApduHelper.toHexString(ksEnc)}")
            Timber.d("Session KS.MAC: ${ApduHelper.toHexString(ksMac)}")
            Timber.d("Initial SSC: ${ApduHelper.toHexString(ssc)}")

            return SessionKeys(ksEnc, ksMac, ssc)

        } catch (e: Exception) {
            Timber.e(e, "BAC authentication error")
            return null
        }
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

        // ISO 9797-1 MAC Algorithm 3:
        // 1. Process all blocks except last with single DES (first 8 bytes of key)
        // 2. Process last block with full 3DES

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
        // Remaining bytes are already 0x00
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

    private fun generateRandom(length: Int): ByteArray {
        val random = java.security.SecureRandom()
        val bytes = ByteArray(length)
        random.nextBytes(bytes)
        return bytes
    }
}
