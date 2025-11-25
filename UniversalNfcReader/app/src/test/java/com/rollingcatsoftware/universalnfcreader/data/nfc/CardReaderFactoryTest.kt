package com.rollingcatsoftware.universalnfcreader.data.nfc

import com.google.common.truth.Truth.assertThat
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.GenericCardReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.MifareClassicReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.MifareDesfireReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.MifareUltralightReader
import com.rollingcatsoftware.universalnfcreader.data.nfc.reader.NdefReader
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [CardReaderFactory].
 */
class CardReaderFactoryTest {

    private lateinit var factory: CardReaderFactory

    @Before
    fun setup() {
        factory = CardReaderFactory()
    }

    @Test
    fun `createReader returns DESFire reader for Istanbulkart`() {
        val reader = factory.createReader(CardType.ISTANBULKART)

        assertThat(reader).isInstanceOf(MifareDesfireReader::class.java)
    }

    @Test
    fun `createReader returns DESFire reader for MIFARE_DESFIRE`() {
        val reader = factory.createReader(CardType.MIFARE_DESFIRE)

        assertThat(reader).isInstanceOf(MifareDesfireReader::class.java)
    }

    @Test
    fun `createReader returns Classic reader for MIFARE_CLASSIC_1K`() {
        val reader = factory.createReader(CardType.MIFARE_CLASSIC_1K)

        assertThat(reader).isInstanceOf(MifareClassicReader::class.java)
    }

    @Test
    fun `createReader returns Classic reader for MIFARE_CLASSIC_4K`() {
        val reader = factory.createReader(CardType.MIFARE_CLASSIC_4K)

        assertThat(reader).isInstanceOf(MifareClassicReader::class.java)
    }

    @Test
    fun `createReader returns Ultralight reader for MIFARE_ULTRALIGHT`() {
        val reader = factory.createReader(CardType.MIFARE_ULTRALIGHT)

        assertThat(reader).isInstanceOf(MifareUltralightReader::class.java)
    }

    @Test
    fun `createReader returns NDEF reader for NDEF`() {
        val reader = factory.createReader(CardType.NDEF)

        assertThat(reader).isInstanceOf(NdefReader::class.java)
    }

    @Test
    fun `createReader returns Generic reader for ISO_14443_A`() {
        val reader = factory.createReader(CardType.ISO_14443_A)

        assertThat(reader).isInstanceOf(GenericCardReader::class.java)
    }

    @Test
    fun `createReader returns Generic reader for UNKNOWN`() {
        val reader = factory.createReader(CardType.UNKNOWN)

        assertThat(reader).isInstanceOf(GenericCardReader::class.java)
    }

    @Test
    fun `createReader returns null for Turkish eID (not implemented)`() {
        val reader = factory.createReader(CardType.TURKISH_EID)

        assertThat(reader).isNull()
    }

    @Test
    fun `isSupported returns true for supported types`() {
        assertThat(factory.isSupported(CardType.ISTANBULKART)).isTrue()
        assertThat(factory.isSupported(CardType.MIFARE_CLASSIC_1K)).isTrue()
        assertThat(factory.isSupported(CardType.NDEF)).isTrue()
    }

    @Test
    fun `isSupported returns false for unsupported types`() {
        assertThat(factory.isSupported(CardType.TURKISH_EID)).isFalse()
    }

    @Test
    fun `getAllReaders returns all reader instances`() {
        val readers = factory.getAllReaders()

        assertThat(readers).hasSize(5)
        assertThat(readers.map { it::class }).containsExactly(
            MifareDesfireReader::class,
            MifareClassicReader::class,
            MifareUltralightReader::class,
            NdefReader::class,
            GenericCardReader::class
        )
    }

    @Test
    fun `getSupportedCardTypes returns expected types`() {
        val supportedTypes = factory.getSupportedCardTypes()

        // Turkish eID is not supported yet
        assertThat(supportedTypes).doesNotContain(CardType.TURKISH_EID)

        // These should be supported
        assertThat(supportedTypes).contains(CardType.ISTANBULKART)
        assertThat(supportedTypes).contains(CardType.MIFARE_CLASSIC_1K)
        assertThat(supportedTypes).contains(CardType.MIFARE_DESFIRE)
        assertThat(supportedTypes).contains(CardType.NDEF)
    }

    @Test
    fun `same reader instance is returned for multiple calls`() {
        val reader1 = factory.createReader(CardType.ISTANBULKART)
        val reader2 = factory.createReader(CardType.ISTANBULKART)

        assertThat(reader1).isSameInstanceAs(reader2)
    }

    @Test
    fun `different reader types return different instances`() {
        val desfireReader = factory.createReader(CardType.MIFARE_DESFIRE)
        val classicReader = factory.createReader(CardType.MIFARE_CLASSIC_1K)

        assertThat(desfireReader).isNotSameInstanceAs(classicReader)
    }
}
