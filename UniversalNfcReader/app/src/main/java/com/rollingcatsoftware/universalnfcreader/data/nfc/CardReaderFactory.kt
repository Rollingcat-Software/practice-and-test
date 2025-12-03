package com.rollingcatsoftware.universalnfcreader.data.nfc

import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.CardReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.GenericCardReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.MifareClassicReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.MifareDesfireReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.MifareUltralightReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.NdefReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.TurkishEidReader
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Factory for creating card readers based on card type.
 *
 * Follows the Factory Pattern to centralize reader instantiation
 * and ensure proper dependency injection.
 *
 * Usage:
 * ```kotlin
 * val factory = CardReaderFactory()
 * val reader = factory.createReader(CardType.ISTANBULKART)
 * val result = reader?.readCard(tag)
 * ```
 */
@Singleton
class CardReaderFactory @Inject constructor() {

    // Lazy initialization of readers (they are stateless, so can be reused)
    private val turkishEidReader by lazy { TurkishEidReader() }
    private val desfireReader by lazy { MifareDesfireReader() }
    private val classicReader by lazy { MifareClassicReader() }
    private val ultralightReader by lazy { MifareUltralightReader() }
    private val ndefReader by lazy { NdefReader() }
    private val genericReader by lazy { GenericCardReader() }

    /**
     * Create a reader appropriate for the given card type.
     *
     * @param cardType The type of card to read
     * @return A CardReader instance, or null if no reader supports this type
     */
    fun createReader(cardType: CardType): CardReader? {
        return when (cardType) {
            // Turkish eID - requires BAC authentication with MRZ data
            CardType.TURKISH_EID -> turkishEidReader

            // DESFire-based cards
            CardType.ISTANBULKART,
            CardType.STUDENT_CARD_DESFIRE,
            CardType.MIFARE_DESFIRE -> desfireReader

            // MIFARE Classic cards
            CardType.MIFARE_CLASSIC_1K,
            CardType.MIFARE_CLASSIC_4K,
            CardType.STUDENT_CARD_CLASSIC -> classicReader

            // Ultralight cards
            CardType.MIFARE_ULTRALIGHT,
            CardType.MIFARE_ULTRALIGHT_C -> ultralightReader

            // NDEF formatted tags
            CardType.NDEF -> ndefReader

            // Generic/Unknown cards
            CardType.ISO_14443_A,
            CardType.ISO_14443_B,
            CardType.ISO_15693,
            CardType.FELICA,
            CardType.UNKNOWN -> genericReader
        }
    }

    /**
     * Get all available readers.
     */
    fun getAllReaders(): List<CardReader> {
        return listOf(
            turkishEidReader,
            desfireReader,
            classicReader,
            ultralightReader,
            ndefReader,
            genericReader
        )
    }

    /**
     * Check if a specific card type is supported.
     */
    fun isSupported(cardType: CardType): Boolean {
        return createReader(cardType) != null
    }

    /**
     * Get all supported card types.
     */
    fun getSupportedCardTypes(): List<CardType> {
        return CardType.entries.filter { isSupported(it) }
    }
}
