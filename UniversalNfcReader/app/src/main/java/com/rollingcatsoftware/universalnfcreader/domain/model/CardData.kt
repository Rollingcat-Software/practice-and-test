package com.rollingcatsoftware.universalnfcreader.domain.model

import android.graphics.Bitmap

/**
 * Base sealed class for all card data types.
 *
 * Each card type has its own data class that extends this sealed class,
 * enabling type-safe handling of different card formats.
 *
 * Security Note: Sensitive data should be cleared from memory after use.
 * Use [clearSensitiveData] to zero out sensitive fields.
 */
sealed class CardData {
    /** Unique identifier of the card */
    abstract val uid: String

    /** Detected card type */
    abstract val cardType: CardType

    /** Timestamp when the card was read */
    abstract val readTimestamp: Long

    /** List of NFC technologies supported by the card */
    abstract val technologies: List<String>

    /** Raw data map for debugging/advanced use */
    abstract val rawData: Map<String, Any>

    /**
     * Clear sensitive data from memory.
     * Override in subclasses that contain sensitive information.
     */
    open fun clearSensitiveData() {
        // Default implementation does nothing
        // Subclasses with sensitive data should override
    }
}

/**
 * Turkish eID card data (MRTD format).
 *
 * Contains personal information read from DG1 and optional photo from DG2.
 * Requires MRZ-based BAC authentication.
 */
data class TurkishEidData(
    override val uid: String,
    override val cardType: CardType = CardType.TURKISH_EID,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // DG1 - Personal Data (MRZ)
    val documentNumber: String = "",
    val surname: String = "",
    val givenNames: String = "",
    val nationality: String = "",
    val dateOfBirth: String = "",
    val sex: String = "",
    val dateOfExpiry: String = "",
    val personalNumber: String = "",

    // DG2 - Facial Image
    val photo: Bitmap? = null,

    // Authentication info
    val bacSuccessful: Boolean = false,
    val sodValid: Boolean? = null
) : CardData() {
    override fun clearSensitiveData() {
        // In Kotlin data classes, we can't mutate.
        // Caller should null out their reference to this object.
        photo?.recycle()
    }
}

/**
 * Istanbulkart transport card data (MIFARE DESFire).
 *
 * Balance and transaction history require proprietary keys from IBB.
 * Only UID and card structure can be read without authentication.
 */
data class IstanbulkartData(
    override val uid: String,
    override val cardType: CardType = CardType.ISTANBULKART,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // DESFire info (publicly readable)
    val desfireVersion: DesfireVersion? = null,
    val applicationIds: List<String> = emptyList(),
    val freeMemory: Int? = null,

    // Protected data (requires keys)
    val balance: Double? = null,
    val lastTransaction: TransactionInfo? = null,
    val expiryDate: String? = null
) : CardData()

/**
 * Student card data (MIFARE Classic or DESFire).
 */
data class StudentCardData(
    override val uid: String,
    override val cardType: CardType,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // Student info (may require default keys)
    val studentId: String? = null,
    val studentName: String? = null,
    val department: String? = null,
    val universityName: String? = null,
    val validUntil: String? = null,

    // Card-specific
    val sectorsRead: Int = 0,
    val totalSectors: Int = 0
) : CardData()

/**
 * Generic MIFARE Classic card data.
 */
data class MifareClassicData(
    override val uid: String,
    override val cardType: CardType,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // Card info
    val sak: Int = 0,
    val atqa: ByteArray = ByteArray(0),
    val sectorCount: Int = 0,
    val blockCount: Int = 0,
    val size: Int = 0,

    // Read data
    val sectorsRead: List<SectorData> = emptyList(),
    val accessibleSectors: Int = 0
) : CardData() {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MifareClassicData) return false
        return uid == other.uid && cardType == other.cardType
    }

    override fun hashCode(): Int = uid.hashCode()
}

/**
 * Generic MIFARE DESFire card data.
 */
data class MifareDesfireData(
    override val uid: String,
    override val cardType: CardType = CardType.MIFARE_DESFIRE,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // DESFire info
    val version: DesfireVersion? = null,
    val applicationIds: List<String> = emptyList(),
    val freeMemory: Int? = null,
    val cardSize: Int? = null
) : CardData()

/**
 * MIFARE Ultralight card data.
 */
data class MifareUltralightData(
    override val uid: String,
    override val cardType: CardType = CardType.MIFARE_ULTRALIGHT,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // Card info
    val pageCount: Int = 0,
    val ultralightType: UltralightType = UltralightType.UNKNOWN,

    // Memory content
    val pages: List<ByteArray> = emptyList(),
    val ndefMessage: String? = null
) : CardData()

/**
 * NDEF formatted tag data.
 */
data class NdefData(
    override val uid: String,
    override val cardType: CardType = CardType.NDEF,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // NDEF info
    val records: List<NdefRecord> = emptyList(),
    val isWritable: Boolean = false,
    val maxSize: Int = 0,
    val usedSize: Int = 0
) : CardData()

/**
 * ISO 15693 (NfcV) card data.
 */
data class Iso15693Data(
    override val uid: String,
    override val cardType: CardType = CardType.ISO_15693,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // NfcV info
    val dsfId: Int = 0,
    val responseFlags: Int = 0,
    val blockSize: Int = 0,
    val blockCount: Int = 0,
    val manufacturer: String = "",

    // Read data
    val blocks: List<ByteArray> = emptyList()
) : CardData()

/**
 * Generic NFC card data (fallback).
 */
data class GenericCardData(
    override val uid: String,
    override val cardType: CardType = CardType.UNKNOWN,
    override val readTimestamp: Long = System.currentTimeMillis(),
    override val technologies: List<String> = emptyList(),
    override val rawData: Map<String, Any> = emptyMap(),

    // Basic info
    val atqa: ByteArray? = null,
    val sak: Short? = null,
    val ats: ByteArray? = null,
    val historicalBytes: ByteArray? = null
) : CardData() {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GenericCardData) return false
        return uid == other.uid
    }

    override fun hashCode(): Int = uid.hashCode()
}

// Supporting data classes

/**
 * MIFARE DESFire version information.
 */
data class DesfireVersion(
    val hardwareVendorId: Int,
    val hardwareType: Int,
    val hardwareSubType: Int,
    val hardwareMajorVersion: Int,
    val hardwareMinorVersion: Int,
    val hardwareStorageSize: Int,
    val hardwareProtocol: Int,
    val softwareVendorId: Int,
    val softwareType: Int,
    val softwareSubType: Int,
    val softwareMajorVersion: Int,
    val softwareMinorVersion: Int,
    val softwareStorageSize: Int,
    val softwareProtocol: Int,
    val uid: ByteArray,
    val batchNumber: ByteArray,
    val productionWeek: Int,
    val productionYear: Int
) {
    val storageSizeBytes: Int
        get() = when (hardwareStorageSize) {
            0x16 -> 2048
            0x18 -> 4096
            0x1A -> 8192
            else -> 0
        }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DesfireVersion) return false
        return uid.contentEquals(other.uid)
    }

    override fun hashCode(): Int = uid.contentHashCode()
}

/**
 * MIFARE Classic sector data.
 */
data class SectorData(
    val sectorNumber: Int,
    val blocks: List<ByteArray>,
    val keyType: KeyType,
    val accessBits: ByteArray?
) {
    enum class KeyType { KEY_A, KEY_B, BOTH, NONE }
}

/**
 * Transaction information (for transport cards).
 */
data class TransactionInfo(
    val timestamp: Long,
    val amount: Double,
    val transactionType: String,
    val location: String?
)

/**
 * NDEF record.
 */
data class NdefRecord(
    val tnf: Short,
    val type: ByteArray,
    val id: ByteArray,
    val payload: ByteArray,
    val payloadAsString: String?
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is NdefRecord) return false
        return tnf == other.tnf && type.contentEquals(other.type)
    }

    override fun hashCode(): Int = 31 * tnf.hashCode() + type.contentHashCode()
}

/**
 * MIFARE Ultralight type.
 */
enum class UltralightType {
    ULTRALIGHT,
    ULTRALIGHT_C,
    ULTRALIGHT_EV1_MF0UL11,
    ULTRALIGHT_EV1_MF0UL21,
    NTAG_213,
    NTAG_215,
    NTAG_216,
    UNKNOWN
}
