package com.rollingcatsoftware.universalnfcreader.di

import com.rollingcatsoftware.universalnfcreader.data.nfc.CardReaderFactory
import com.rollingcatsoftware.universalnfcreader.data.nfc.NfcCardReadingService
import com.rollingcatsoftware.universalnfcreader.data.nfc.detector.CardDetector
import com.rollingcatsoftware.universalnfcreader.data.nfc.detector.UniversalCardDetector
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module providing application-level dependencies.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideCardDetector(): CardDetector {
        return UniversalCardDetector()
    }

    @Provides
    @Singleton
    fun provideCardReaderFactory(): CardReaderFactory {
        return CardReaderFactory()
    }

    @Provides
    @Singleton
    fun provideNfcCardReadingService(
        detector: CardDetector,
        factory: CardReaderFactory
    ): NfcCardReadingService {
        return NfcCardReadingService(detector, factory)
    }
}
