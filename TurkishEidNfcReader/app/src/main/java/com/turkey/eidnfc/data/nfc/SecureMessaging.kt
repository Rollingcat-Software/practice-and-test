package com.turkey.eidnfc.data.nfc

import timber.log.Timber
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * Secure Messaging (SM) wrapper for ICAO MRTD documents.
 *
 * After BAC authentication, all commands must be wrapped with secure messaging:
 * - Commands are encrypted with the session encryption key
 * - Commands include a MAC for integrity verification
 * - Send Sequence Counter (SSC) prevents replay attacks
 *
 * Based on ICAO Doc 9303 Part 11.
 */
class SecureMessaging(
    private val encryptionKey: ByteArray,
    private val macKey: ByteArray,
    initialSsc: ByteArray
) {
    // Send Sequence Counter - incremented for each command/response
    private var ssc: ByteArray = initialSsc.copyOf()

    companion object {
        private val ZERO_IV = ByteArray(8)
    }

    /**
     * Wraps an APDU command with secure messaging.
     *
     * @param command Plain APDU command
     * @return Secure messaging wrapped APDU
     */
    fun wrapCommand(command: ByteArray): ByteArray {
        if (command.size < 4) {
            throw IllegalArgumentException("Invalid APDU command")
        }

        // Increment SSC
        incrementSsc()

        val cla = command[0]
        val ins = command[1]
        val p1 = command[2]
        val p2 = command[3]

        // Mask CLA byte for secure messaging (set bit 3)
        val maskedCla = (cla.toInt() or 0x0C).toByte()

        // Build command header for MAC calculation
        val cmdHeader = byteArrayOf(maskedCla, ins, p1, p2)
        val paddedHeader = padISO9797(cmdHeader)

        // Check if there's command data (Lc and data)
        val hasData = command.size > 5
        val data = if (hasData) {
            val lc = command[4].toInt() and 0xFF
            command.copyOfRange(5, 5 + lc)
        } else {
            byteArrayOf()
        }

        // Check if Le is present
        val hasLe = if (hasData) {
            command.size > 5 + (command[4].toInt() and 0xFF)
        } else {
            command.size > 4
        }
        val le = if (hasLe) {
            if (hasData) {
                command[command.size - 1]
            } else {
                command[4]
            }
        } else {
            null
        }

        // Build secure messaging data objects
        val do87 = if (data.isNotEmpty()) {
            buildDO87(data)
        } else {
            byteArrayOf()
        }

        val do97 = if (le != null) {
            buildDO97(le)
        } else {
            byteArrayOf()
        }

        // Calculate MAC over: SSC || padded header || DO87 || DO97
        val macInput = ssc + paddedHeader + padISO9797(do87 + do97)
        val mac = calculateMAC(macInput, macKey)
        val do8e = buildDO8E(mac)

        // Build protected APDU
        val protectedData = do87 + do97 + do8e
        val newLc = protectedData.size

        val protectedApdu = byteArrayOf(
            maskedCla, ins, p1, p2,
            newLc.toByte()
        ) + protectedData + byteArrayOf(0x00) // Le = 0x00 for extended response

        Timber.d("Protected APDU: ${ApduHelper.toHexString(protectedApdu)}")
        return protectedApdu
    }

    /**
     * Unwraps a secure messaging response.
     *
     * @param response Secure messaging wrapped response
     * @return Pair of (plaintext data, status word) or null if verification fails
     */
    fun unwrapResponse(response: ByteArray): Pair<ByteArray, Int>? {
        if (response.size < 2) {
            Timber.e("Response too short")
            return null
        }

        // Increment SSC for response
        incrementSsc()

        val sw1 = response[response.size - 2].toInt() and 0xFF
        val sw2 = response[response.size - 1].toInt() and 0xFF
        val statusWord = (sw1 shl 8) or sw2

        if (statusWord != 0x9000) {
            // Error response - may not be protected
            return Pair(byteArrayOf(), statusWord)
        }

        // Parse response data objects
        val responseData = response.copyOfRange(0, response.size - 2)

        var offset = 0
        var encryptedData: ByteArray? = null
        var responseSw: ByteArray? = null
        var responseMac: ByteArray? = null

        while (offset < responseData.size) {
            val tag = responseData[offset].toInt() and 0xFF
            offset++

            val length = if ((responseData[offset].toInt() and 0xFF) == 0x81) {
                offset++
                responseData[offset++].toInt() and 0xFF
            } else if ((responseData[offset].toInt() and 0xFF) == 0x82) {
                offset++
                val len1 = responseData[offset++].toInt() and 0xFF
                val len2 = responseData[offset++].toInt() and 0xFF
                (len1 shl 8) or len2
            } else {
                responseData[offset++].toInt() and 0xFF
            }

            val value = responseData.copyOfRange(offset, offset + length)
            offset += length

            when (tag) {
                0x87 -> {
                    // Encrypted data (skip padding indicator byte)
                    if (value.isNotEmpty() && value[0] == 0x01.toByte()) {
                        encryptedData = value.copyOfRange(1, value.size)
                    }
                }
                0x99 -> responseSw = value
                0x8E -> responseMac = value
            }
        }

        // Verify MAC
        if (responseMac == null) {
            Timber.e("No MAC in response")
            return null
        }

        // Build MAC input: SSC || DO87 || DO99
        val macInputParts = mutableListOf<Byte>()
        macInputParts.addAll(ssc.toList())

        // Reconstruct DO87 and DO99 for MAC verification
        var macOffset = 0
        while (macOffset < responseData.size) {
            val tag = responseData[macOffset].toInt() and 0xFF
            if (tag == 0x8E) break // Stop before DO8E

            val startOffset = macOffset
            macOffset++

            val lengthByte = responseData[macOffset].toInt() and 0xFF
            if (lengthByte == 0x81) {
                macOffset += 2
                val len = responseData[macOffset - 1].toInt() and 0xFF
                macOffset += len
            } else if (lengthByte == 0x82) {
                macOffset += 3
                val len = ((responseData[macOffset - 2].toInt() and 0xFF) shl 8) or
                        (responseData[macOffset - 1].toInt() and 0xFF)
                macOffset += len
            } else {
                macOffset++
                macOffset += lengthByte
            }

            macInputParts.addAll(responseData.copyOfRange(startOffset, macOffset).toList())
        }

        val macInput = padISO9797(macInputParts.toByteArray())
        val calculatedMac = calculateMAC(macInput, macKey)

        if (!calculatedMac.contentEquals(responseMac)) {
            Timber.e("MAC verification failed")
            Timber.e("Expected: ${ApduHelper.toHexString(calculatedMac)}")
            Timber.e("Received: ${ApduHelper.toHexString(responseMac)}")
            return null
        }

        // Decrypt data if present
        val plainData = if (encryptedData != null && encryptedData.isNotEmpty()) {
            val decrypted = decrypt3DES(encryptedData, encryptionKey)
            removePadding(decrypted)
        } else {
            byteArrayOf()
        }

        // Get status word from DO99 if present
        val finalSw = if (responseSw != null && responseSw.size >= 2) {
            ((responseSw[0].toInt() and 0xFF) shl 8) or (responseSw[1].toInt() and 0xFF)
        } else {
            statusWord
        }

        return Pair(plainData, finalSw)
    }

    /**
     * Builds DO'87 (encrypted data) TLV.
     */
    private fun buildDO87(data: ByteArray): ByteArray {
        // Pad and encrypt data
        val padded = padISO9797(data)
        val encrypted = encrypt3DES(padded, encryptionKey)

        // Build TLV: 87 || L || 01 || encrypted data
        val value = byteArrayOf(0x01) + encrypted
        return buildTLV(0x87, value)
    }

    /**
     * Builds DO'97 (expected length) TLV.
     */
    private fun buildDO97(le: Byte): ByteArray {
        return byteArrayOf(0x97.toByte(), 0x01, le)
    }

    /**
     * Builds DO'8E (MAC) TLV.
     */
    private fun buildDO8E(mac: ByteArray): ByteArray {
        return byteArrayOf(0x8E.toByte(), mac.size.toByte()) + mac
    }

    /**
     * Builds a TLV with proper length encoding.
     */
    private fun buildTLV(tag: Int, value: ByteArray): ByteArray {
        val length = value.size
        return when {
            length < 0x80 -> byteArrayOf(tag.toByte(), length.toByte()) + value
            length < 0x100 -> byteArrayOf(tag.toByte(), 0x81.toByte(), length.toByte()) + value
            else -> byteArrayOf(
                tag.toByte(), 0x82.toByte(),
                (length shr 8).toByte(), (length and 0xFF).toByte()
            ) + value
        }
    }

    /**
     * Increments the Send Sequence Counter.
     */
    private fun incrementSsc() {
        for (i in ssc.size - 1 downTo 0) {
            ssc[i] = (ssc[i].toInt() + 1).toByte()
            if (ssc[i] != 0.toByte()) break
        }
        Timber.d("SSC: ${ApduHelper.toHexString(ssc)}")
    }

    /**
     * ISO 9797-1 padding method 2.
     */
    private fun padISO9797(data: ByteArray): ByteArray {
        val padLength = 8 - (data.size % 8)
        val padded = ByteArray(data.size + padLength)
        System.arraycopy(data, 0, padded, 0, data.size)
        padded[data.size] = 0x80.toByte()
        return padded
    }

    /**
     * Removes ISO 9797-1 padding.
     */
    private fun removePadding(data: ByteArray): ByteArray {
        var i = data.size - 1
        while (i >= 0 && data[i] == 0x00.toByte()) {
            i--
        }
        if (i >= 0 && data[i] == 0x80.toByte()) {
            return data.copyOfRange(0, i)
        }
        return data
    }

    /**
     * ISO 9797-1 MAC Algorithm 3 (Retail MAC).
     */
    private fun calculateMAC(data: ByteArray, key: ByteArray): ByteArray {
        val key1 = key.copyOfRange(0, 8)
        val key2 = key.copyOfRange(8, 16)

        var h = ByteArray(8)

        val numBlocks = data.size / 8
        for (i in 0 until numBlocks - 1) {
            val block = data.copyOfRange(i * 8, (i + 1) * 8)
            for (j in 0..7) {
                h[j] = (h[j].toInt() xor block[j].toInt()).toByte()
            }
            h = encryptDES(h, key1)
        }

        val lastBlock = data.copyOfRange((numBlocks - 1) * 8, numBlocks * 8)
        for (j in 0..7) {
            h[j] = (h[j].toInt() xor lastBlock[j].toInt()).toByte()
        }

        h = encryptDES(h, key1)
        h = decryptDES(h, key2)
        h = encryptDES(h, key1)

        return h
    }

    private fun encrypt3DES(data: ByteArray, key: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("DESede/CBC/NoPadding")
        val keySpec = SecretKeySpec(key, "DESede")
        val ivSpec = IvParameterSpec(ZERO_IV)
        cipher.init(Cipher.ENCRYPT_MODE, keySpec, ivSpec)
        return cipher.doFinal(data)
    }

    private fun decrypt3DES(data: ByteArray, key: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("DESede/CBC/NoPadding")
        val keySpec = SecretKeySpec(key, "DESede")
        val ivSpec = IvParameterSpec(ZERO_IV)
        cipher.init(Cipher.DECRYPT_MODE, keySpec, ivSpec)
        return cipher.doFinal(data)
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
}
