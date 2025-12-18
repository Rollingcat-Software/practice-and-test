package com.rollingcatsoftware.universalnfcreader.domain.model

/**
 * Enumeration of all supported NFC card types.
 *
 * Card detection follows a priority order based on specificity:
 * 1. Turkish eID (most specific - requires exact AID match)
 * 2. Istanbulkart (DESFire + specific UID pattern)
 * 3. Student Cards (MIFARE Classic/DESFire with common patterns)
 * 4. Generic MIFARE types (by technology)
 * 5. Generic NFC types (fallback)
 */
enum class CardType(
    val displayName: String,
    val description: String,
    val requiresAuthentication: Boolean
) {
    // Passport - ICAO 9303 TD3 format with BAC/PACE authentication
    PASSPORT(
        displayName = "Passport",
        description = "e-Passport (ICAO 9303 MRTD)",
        requiresAuthentication = true
    ),

    // Turkish eID - ISO 7816-4 with BAC authentication
    TURKISH_EID(
        displayName = "Turkish eID",
        description = "Turkish National Identity Card (MRTD)",
        requiresAuthentication = true
    ),

    // Transport Cards
    ISTANBULKART(
        displayName = "Istanbulkart",
        description = "Istanbul Transport Card (MIFARE DESFire)",
        requiresAuthentication = false // Can read structure without keys
    ),

    // Student Cards
    STUDENT_CARD_CLASSIC(
        displayName = "Student Card (Classic)",
        description = "MIFARE Classic based student ID",
        requiresAuthentication = false // Tries default keys
    ),

    STUDENT_CARD_DESFIRE(
        displayName = "Student Card (DESFire)",
        description = "MIFARE DESFire based student ID",
        requiresAuthentication = false // Can read structure
    ),

    // MIFARE Family
    MIFARE_CLASSIC_1K(
        displayName = "MIFARE Classic 1K",
        description = "1KB memory, 16 sectors",
        requiresAuthentication = false // Tries default keys
    ),

    MIFARE_CLASSIC_4K(
        displayName = "MIFARE Classic 4K",
        description = "4KB memory, 40 sectors",
        requiresAuthentication = false // Tries default keys
    ),

    MIFARE_DESFIRE(
        displayName = "MIFARE DESFire",
        description = "High security, multiple applications",
        requiresAuthentication = false // Can read structure
    ),

    MIFARE_ULTRALIGHT(
        displayName = "MIFARE Ultralight",
        description = "64 bytes memory, no authentication",
        requiresAuthentication = false
    ),

    MIFARE_ULTRALIGHT_C(
        displayName = "MIFARE Ultralight C",
        description = "192 bytes memory, 3DES authentication",
        requiresAuthentication = false // Basic read without auth
    ),

    // ISO Standards
    ISO_14443_A(
        displayName = "ISO 14443-A",
        description = "Generic NFC-A card",
        requiresAuthentication = false
    ),

    ISO_14443_B(
        displayName = "ISO 14443-B",
        description = "Generic NFC-B card",
        requiresAuthentication = false
    ),

    ISO_15693(
        displayName = "ISO 15693 (NfcV)",
        description = "Vicinity card",
        requiresAuthentication = false
    ),

    // NDEF Formatted
    NDEF(
        displayName = "NDEF Tag",
        description = "NFC Forum formatted tag",
        requiresAuthentication = false
    ),

    // FeliCa (Sony)
    FELICA(
        displayName = "FeliCa",
        description = "Sony FeliCa NFC-F card",
        requiresAuthentication = false
    ),

    // Unknown/Generic
    UNKNOWN(
        displayName = "Unknown Card",
        description = "Unidentified NFC card",
        requiresAuthentication = false
    );

    companion object {
        /**
         * Get card type for MIFARE Classic based on SAK byte.
         */
        fun fromMifareClassicSak(sak: Short): CardType {
            return when (sak.toInt()) {
                0x08 -> MIFARE_CLASSIC_1K  // Classic 1K
                0x09 -> MIFARE_CLASSIC_1K  // Mini
                0x18 -> MIFARE_CLASSIC_4K  // Classic 4K
                0x28 -> MIFARE_CLASSIC_1K  // Emulated
                0x38 -> MIFARE_CLASSIC_4K  // Emulated
                else -> MIFARE_CLASSIC_1K
            }
        }
    }
}
