package com.rollingcatsoftware.universalnfcreader.data.nfc.detector

import android.nfc.Tag
import android.nfc.tech.IsoDep
import android.nfc.tech.MifareClassic
import android.nfc.tech.MifareUltralight
import android.nfc.tech.Ndef
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
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
        SecureLogger.d(TAG, "Detecting card type. Technologies: ${techList.joinToString()}")
        SecureLogger.d(TAG, "Tag UID: ${tag.id.toHexString()}")

        // Check for NDEF first (formatted tags)
        if (techList.any { it.contains("Ndef") }) {
            val ndef = Ndef.get(tag)
            if (ndef != null) {
                SecureLogger.d(TAG, "NDEF tag detected")
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
            SecureLogger.d(TAG, "ISO 15693 (NfcV) card detected")
            return@withContext CardType.ISO_15693
        }

        // Check for NfcF (FeliCa)
        if (techList.any { it.contains("NfcF") }) {
            SecureLogger.d(TAG, "FeliCa card detected")
            return@withContext CardType.FELICA
        }

        // Generic NFC-A
        if (techList.any { it.contains("NfcA") }) {
            SecureLogger.d(TAG, "Generic NFC-A card detected")
            return@withContext CardType.ISO_14443_A
        }

        // Generic NFC-B
        if (techList.any { it.contains("NfcB") }) {
            SecureLogger.d(TAG, "Generic NFC-B card detected")
            return@withContext CardType.ISO_14443_B
        }

        SecureLogger.d(TAG, "Unknown card type")
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

            // Try standard MRTD AID (e-Passports and eID cards use the same AID)
            // A0 00 00 02 47 10 01 is the ICAO MRTD application ID
            if (trySelectAid(isoDep, Constants.AID_MRTD)) {
                SecureLogger.d(TAG, "MRTD document detected via ICAO AID")

                // Try to determine document type by reading EF.CardAccess or ATR
                // Turkish eID cards typically have different historical bytes
                val historicalBytes = isoDep.historicalBytes
                val isTurkishEid = historicalBytes != null && detectTurkishEid(historicalBytes, isoDep)

                return@withContext if (isTurkishEid) {
                    SecureLogger.d(TAG, "Detected as Turkish eID based on card profile")
                    CardType.TURKISH_EID
                } else {
                    SecureLogger.d(TAG, "Detected as e-Passport (generic MRTD)")
                    CardType.PASSPORT
                }
            }

            // Check for DESFire (Istanbulkart, etc.)
            val desfireType = detectDesfireCard(isoDep, tag)
            if (desfireType != CardType.UNKNOWN) {
                return@withContext desfireType
            }

            // Generic IsoDep
            SecureLogger.d(TAG, "Generic IsoDep card detected")
            CardType.ISO_14443_A

        } catch (e: IOException) {
            SecureLogger.e(TAG, "Error detecting IsoDep card: ${e.message}")
            CardType.UNKNOWN
        } finally {
            try {
                if (isoDep.isConnected) isoDep.close()
            } catch (e: IOException) {
                SecureLogger.w(TAG, "Error closing IsoDep: ${e.message}")
            }
        }
    }

    /**
     * Attempts to detect if this is a Turkish eID card.
     *
     * Turkish eID cards (TC Kimlik Kartı) have specific characteristics:
     * 1. Historical bytes may contain "TCKK" or Turkish identifiers
     * 2. EF.CardAccess contains PACE-GM or DH parameters specific to Turkey
     * 3. Card profile typically indicates TD1 format
     *
     * Note: This is a heuristic - some cards may not be distinguishable without
     * reading DG1 to check MRZ format (TD1 = 90 chars vs TD3 = 88 chars)
     */
    private fun detectTurkishEid(historicalBytes: ByteArray, isoDep: IsoDep): Boolean {
        try {
            // Check historical bytes for Turkish indicators
            val histStr = String(historicalBytes, Charsets.ISO_8859_1)
            if (histStr.contains("TCKK") || histStr.contains("TUR") || histStr.contains("TC")) {
                SecureLogger.d(TAG, "Historical bytes indicate Turkish eID")
                return true
            }

            // Try to read EF.CardAccess (public file, doesn't require auth)
            // Turkish eID cards have specific PACE parameters
            val readCardAccess = byteArrayOf(
                0x00, 0xB0.toByte(), 0x9C.toByte(), 0x00, 0x00 // READ BINARY with SFI 1C
            )

            val response = isoDep.transceive(readCardAccess)
            if (response.size > 10) {
                // If EF.CardAccess is present and readable, analyze content
                // Turkish eID typically uses PACE-GM with specific OIDs
                val responseHex = response.joinToString("") { "%02X".format(it) }
                // Check for Turkish-specific OID pattern (approximate)
                if (responseHex.contains("0409") || responseHex.contains("0410")) {
                    SecureLogger.d(TAG, "EF.CardAccess suggests Turkish eID (PACE-GM parameters)")
                    return true
                }
            }

            // Default: If we can't determine, assume passport (more common globally)
            return false

        } catch (e: Exception) {
            SecureLogger.d(TAG, "Could not determine if Turkish eID: ${e.message}")
            return false
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
            SecureLogger.d(TAG, "SELECT AID ${aid.toHexString()} response: ${response.toHexString()}")
            response.isSuccess()
        } catch (e: IOException) {
            SecureLogger.d(TAG, "SELECT AID ${aid.toHexString()} failed: ${e.message}")
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
                SecureLogger.d(TAG, "DESFire card detected")

                // Check UID pattern for Istanbulkart (NXP manufacturer, specific pattern)
                val uid = tag.id
                if (uid.isNotEmpty() && uid[0] == Constants.NXP_MANUFACTURER_CODE) {
                    // Could be Istanbulkart - check further
                    // Istanbulkart typically has 7-byte UID starting with 04
                    if (uid.size == 7) {
                        SecureLogger.d(TAG, "Potential Istanbulkart detected (7-byte NXP UID)")
                        return CardType.ISTANBULKART
                    }
                }

                // Generic DESFire
                CardType.MIFARE_DESFIRE
            } else {
                CardType.UNKNOWN
            }
        } catch (e: IOException) {
            SecureLogger.d(TAG, "DESFire detection failed: ${e.message}")
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
                            SecureLogger.d(TAG, "MIFARE Classic 1K detected")
                            CardType.MIFARE_CLASSIC_1K
                        }

                        MifareClassic.SIZE_4K -> {
                            SecureLogger.d(TAG, "MIFARE Classic 4K detected")
                            CardType.MIFARE_CLASSIC_4K
                        }

                        MifareClassic.SIZE_MINI -> {
                            SecureLogger.d(TAG, "MIFARE Mini detected")
                            CardType.MIFARE_CLASSIC_1K
                        }

                        else -> {
                            SecureLogger.d(TAG, "MIFARE Classic (unknown size) detected")
                            CardType.MIFARE_CLASSIC_1K
                        }
                    }
                }

                MifareClassic.TYPE_PLUS -> {
                    SecureLogger.d(TAG, "MIFARE Plus detected")
                    CardType.MIFARE_CLASSIC_1K
                }

                MifareClassic.TYPE_PRO -> {
                    SecureLogger.d(TAG, "MIFARE Pro detected")
                    CardType.MIFARE_CLASSIC_4K
                }

                else -> {
                    SecureLogger.d(TAG, "Unknown MIFARE Classic type")
                    CardType.MIFARE_CLASSIC_1K
                }
            }

            // Check for student card patterns
            // Student cards often have specific UID patterns or sector layouts
            // For now, return generic MIFARE type
            type

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error detecting MIFARE Classic: ${e.message}")
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
                    SecureLogger.d(TAG, "MIFARE Ultralight detected")
                    CardType.MIFARE_ULTRALIGHT
                }

                MifareUltralight.TYPE_ULTRALIGHT_C -> {
                    SecureLogger.d(TAG, "MIFARE Ultralight C detected")
                    CardType.MIFARE_ULTRALIGHT_C
                }

                else -> {
                    // Could be NTAG - check by reading pages
                    SecureLogger.d(TAG, "Unknown Ultralight type, assuming standard")
                    CardType.MIFARE_ULTRALIGHT
                }
            }
            type
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error detecting MIFARE Ultralight: ${e.message}")
            CardType.MIFARE_ULTRALIGHT
        }
    }

    override fun getSupportedTechnologies(tag: Tag): List<String> {
        return tag.techList.toList()
    }
}
