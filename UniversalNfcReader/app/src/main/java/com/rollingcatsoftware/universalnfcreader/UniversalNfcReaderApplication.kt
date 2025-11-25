package com.rollingcatsoftware.universalnfcreader

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class for Universal NFC Reader.
 *
 * Initializes Hilt dependency injection.
 */
@HiltAndroidApp
class UniversalNfcReaderApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        // Initialize any application-level components here
    }
}
