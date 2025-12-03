package com.turkey.eidnfc.di

import android.content.Context
import android.nfc.NfcAdapter
import com.turkey.eidnfc.data.nfc.NfcCardReader
import com.turkey.eidnfc.util.Dg1Parser
import com.turkey.eidnfc.util.Dg2Parser
import com.turkey.eidnfc.util.SodValidator
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module for providing application-level dependencies.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    /**
     * Provides NFC adapter instance.
     * Returns null if device doesn't support NFC.
     */
    @Provides
    @Singleton
    fun provideNfcAdapter(@ApplicationContext context: Context): NfcAdapter? {
        return NfcAdapter.getDefaultAdapter(context)
    }

    /**
     * Provides NFC card reader instance.
     */
    @Provides
    @Singleton
    fun provideNfcCardReader(): NfcCardReader {
        return NfcCardReader()
    }

    /**
     * Provides DG1 parser instance.
     */
    @Provides
    @Singleton
    fun provideDg1Parser(): Dg1Parser {
        return Dg1Parser
    }

    /**
     * Provides DG2 parser instance.
     */
    @Provides
    @Singleton
    fun provideDg2Parser(): Dg2Parser {
        return Dg2Parser
    }

    /**
     * Provides SOD validator instance.
     */
    @Provides
    @Singleton
    fun provideSodValidator(): SodValidator {
        return SodValidator
    }
}
