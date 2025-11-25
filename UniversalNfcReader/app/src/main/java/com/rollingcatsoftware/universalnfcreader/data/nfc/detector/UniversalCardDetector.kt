package com.rollingcatsoftware.universalnfcreader.data.nfc.detector

import android.nfc.Tag
import android.nfc.tech.IsoDep
import android.nfc.tech.MifareClassic
import android.nfc.tech.MifareUltralight
import android.nfc.tech.Ndef
import android.util.Log
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.util.Constants
import com.rollingcatsoftware.universalnfcreader.util.getStatusWord
import com.rollingcatsoftware.universalnfcreader.util.isSuccess
import com.rollingcatsoftware.universalnfcreader.util.toHexString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * Universal card detector that orchestrates multiple specific detectors.
 *
 * Detection order (by priority):
 * 1. Turkish eID (most specific - exact AID match)
 * 2. Istanbulkart (DESFire + UID pattern)
 * 3. Student Cards (MIFARE patterns)
 * 4. Generic MIFARE types
 * 5. Generic NFC types (fallback)
 */
class UniversalCardDetector : CardDetector {

    companion object {
        private const val TAG = "UniversalCardDetector"
    }

    override suspend fun detectCardType(tag: Tag): CardType = withContext(Dispatchers.IO) {
        val techList = tag.techList.toList()
        Log.d(TAG, "Detecting card type. Technologies: ${techList.joinToString()}")
        Log.d(TAG, "Tag UID: ${tag.id.toHexString()}")

        // Check for NDEF first (formatted tags)
        if (techList.any { it.contains("Ndef") }) {
            val ndef = Ndef.get(tag)
            if (ndef != null) {
                Log.d(TAG, "NDEF tag detected")
                return@withContext CardType.NDEF
            }
        }

        // Check for IsoDep (smart cards)
        if (techList.any { it.contains("IsoDep") }) {
            val isoDepType = detectIsoDepCard(tag)
            if (isoDepType != CardType.UNKNOWN) {
                return@withContext isoDepType
            }
        }

        // Check for MIFARE Classic
        if (techList.any { it.contains("MifareClassic") }) {
            return@withContext detectMifareClassic(tag)
        }

        // Check for MIFARE Ultralight
        if (techList.any { it.contains("MifareUltralight") }) {
            return@withContext detectMifareUltralight(tag)
        }

        // Check for NfcV (ISO 15693)
        if (techList.any { it.contains("NfcV") }) {
            Log.d(TAG, "ISO 15693 (NfcV) card detected")
            return@withContext CardType.ISO_15693
        }

        // Check for NfcF (FeliCa)
        if (techList.any { it.contains("NfcF") }) {
            Log.d(TAG, "FeliCa card detected")
            return@withContext CardType.FELICA
        }

        // Generic NFC-A
        if (techList.any { it.contains("NfcA") }) {
            Log.d(TAG, "Generic NFC-A card detected")
            return@withContext CardType.ISO_14443_A
        }

        // Generic NFC-B
        if (techList.any { it.contains("NfcB") }) {
            Log.d(TAG, "Generic NFC-B card detected")
            return@withContext CardType.ISO_14443_B
        }

        Log.d(TAG, "Unknown card type")
        CardType.UNKNOWN
    }

    /**
     * Detect specific IsoDep card type by trying AID selection.
     */
    private suspend fun detectIsoDepCard(tag: Tag): CardType = withContext(Dispatchers.IO) {
        val isoDep = IsoDep.get(tag) ?: return@withContext CardType.UNKNOWN

        try {
            isoDep.connect()
            isoDep.timeout = Constants.Timeout.SHORT_MS

            // Try Turkish eID AID first
            if (trySelectAid(isoDep, Constants.AID_TURKISH_EID)) {
                Log.d(TAG, "Turkish eID detected via AID selection")
                return@withContext CardType.TURKISH_EID
            }

            // Try MRTD AID (other passports/IDs)
            if (trySelectAid(isoDep, Constants.AID_MRTD)) {
                Log.d(TAG, "MRTD card detected")
                return@withContext CardType.TURKISH_EID // Generic MRTD handling
            }

            // Check for DESFire (Istanbulkart, etc.)
            val desfireType = detectDesfireCard(isoDep, tag)
            if (desfireType != CardType.UNKNOWN) {
                return@withContext desfireType
            }

            // Generic IsoDep
            Log.d(TAG, "Generic IsoDep card detected")
            CardType.ISO_14443_A

        } catch (e: IOException) {
            Log.e(TAG, "Error detecting IsoDep card: ${e.message}")
            CardType.UNKNOWN
        } finally {
            try {
                if (isoDep.isConnected) isoDep.close()
            } catch (e: IOException) {
                Log.w(TAG, "Error closing IsoDep: ${e.message}")
            }
        }
    }

    /**
     * Try to select an application by AID.
     */
    private fun trySelectAid(isoDep: IsoDep, aid: ByteArray): Boolean {
        val selectCommand = byteArrayOf(
            0x00, // CLA
            0xA4.toByte(), // INS: SELECT
            0x04, // P1: Select by DF name
            0x0C, // P2: First or only occurrence
            aid.size.toByte(), // Lc
            *aid, // AID
            0x00 // Le
        )

        return try {
            val response = isoDep.transceive(selectCommand)
            Log.d(TAG, "SELECT AID ${aid.toHexString()} response: ${response.toHexString()}")
            response.isSuccess()
        } catch (e: IOException) {
            Log.d(TAG, "SELECT AID ${aid.toHexString()} failed: ${e.message}")
            false
        }
    }

    /**
     * Detect DESFire card type (Istanbulkart, student cards, generic).
     */
    private fun detectDesfireCard(isoDep: IsoDep, tag: Tag): CardType {
        // Send native DESFire GET_VERSION command wrapped in ISO 7816
        val getVersionCmd = byteArrayOf(
            0x90.toByte(), // CLA: DESFire native wrapped
            0x60, // INS: GetVersion
            0x00, // P1
            0x00, // P2
            0x00  // Le
        )

        return try {
            val response = isoDep.transceive(getVersionCmd)
            val sw = response.getStatusWord()

            if (sw == 0x91AF || sw == 0x9100) {
                // This is a DESFire card
                Log.d(TAG, "DESFire card detected")

                // Check UID pattern for Istanbulkart (NXP manufacturer, specific pattern)
                val uid = tag.id
                if (uid.isNotEmpty() && uid[0] == Constants.NXP_MANUFACTURER_CODE) {
                    // Could be Istanbulkart - check further
                    // Istanbulkart typically has 7-byte UID starting with 04
                    if (uid.size == 7) {
                        Log.d(TAG, "Potential Istanbulkart detected (7-byte NXP UID)")
                        return CardType.ISTANBULKART
                    }
                }

                // Generic DESFire
                CardType.MIFARE_DESFIRE
            } else {
                CardType.UNKNOWN
            }
        } catch (e: IOException) {
            Log.d(TAG, "DESFire detection failed: ${e.message}")
            CardType.UNKNOWN
        }
    }

    /**
     * Detect MIFARE Classic card variant.
     */
    private fun detectMifareClassic(tag: Tag): CardType {
        val mifare = MifareClassic.get(tag) ?: return CardType.UNKNOWN

        return try {
            val type = when (mifare.type) {
                MifareClassic.TYPE_CLASSIC -> {
                    when (mifare.size) {
                        MifareClassic.SIZE_1K -> {
                            Log.d(TAG, "MIFARE Classic 1K detected")
                            CardType.MIFARE_CLASSIC_1K
                        }

                        MifareClassic.SIZE_4K -> {
                            Log.d(TAG, "MIFARE Classic 4K detected")
                            CardType.MIFARE_CLASSIC_4K
                        }

                        MifareClassic.SIZE_MINI -> {
                            Log.d(TAG, "MIFARE Mini detected")
                            CardType.MIFARE_CLASSIC_1K
                        }

                        else -> {
                            Log.d(TAG, "MIFARE Classic (unknown size) detected")
                            CardType.MIFARE_CLASSIC_1K
                        }
                    }
                }

                MifareClassic.TYPE_PLUS -> {
                    Log.d(TAG, "MIFARE Plus detected")
                    CardType.MIFARE_CLASSIC_1K
                }

                MifareClassic.TYPE_PRO -> {
                    Log.d(TAG, "MIFARE Pro detected")
                    CardType.MIFARE_CLASSIC_4K
                }

                else -> {
                    Log.d(TAG, "Unknown MIFARE Classic type")
                    CardType.MIFARE_CLASSIC_1K
                }
            }

            // Check for student card patterns
            // Student cards often have specific UID patterns or sector layouts
            // For now, return generic MIFARE type
            type

        } catch (e: Exception) {
            Log.e(TAG, "Error detecting MIFARE Classic: ${e.message}")
            CardType.MIFARE_CLASSIC_1K
        }
    }

    /**
     * Detect MIFARE Ultralight variant.
     */
    private fun detectMifareUltralight(tag: Tag): CardType {
        val ultralight = MifareUltralight.get(tag) ?: return CardType.UNKNOWN

        return try {
            val type = when (ultralight.type) {
                MifareUltralight.TYPE_ULTRALIGHT -> {
                    Log.d(TAG, "MIFARE Ultralight detected")
                    CardType.MIFARE_ULTRALIGHT
                }

                MifareUltralight.TYPE_ULTRALIGHT_C -> {
                    Log.d(TAG, "MIFARE Ultralight C detected")
                    CardType.MIFARE_ULTRALIGHT_C
                }

                else -> {
                    // Could be NTAG - check by reading pages
                    Log.d(TAG, "Unknown Ultralight type, assuming standard")
                    CardType.MIFARE_ULTRALIGHT
                }
            }
            type
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting MIFARE Ultralight: ${e.message}")
            CardType.MIFARE_ULTRALIGHT
        }
    }

    override fun getSupportedTechnologies(tag: Tag): List<String> {
        return tag.techList.toList()
    }
}
