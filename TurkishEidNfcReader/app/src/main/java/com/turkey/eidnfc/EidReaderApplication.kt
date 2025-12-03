package com.turkey.eidnfc

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import timber.log.Timber

/**
 * Application class for Turkish eID NFC Reader.
 *
 * Initializes Hilt dependency injection and Timber logging.
 */
@HiltAndroidApp
class EidReaderApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        initializeTimber()
    }

    private fun initializeTimber() {
        if (BuildConfig.DEBUG) {
            // Plant debug tree for development
            Timber.plant(Timber.DebugTree())
            Timber.d("Timber initialized in DEBUG mode")
        } else {
            // Plant release tree that only logs errors and warnings
            Timber.plant(ReleaseTree())
            Timber.d("Timber initialized in RELEASE mode")
        }
    }

    /**
     * Release tree that only logs errors and warnings.
     * In production, you might want to send these to a crash reporting service.
     */
    private class ReleaseTree : Timber.Tree() {
        override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
            if (priority == android.util.Log.ERROR || priority == android.util.Log.WARN) {
                // In production, send to crash reporting service (e.g., Firebase Crashlytics)
                // For now, just use Android Log
                if (t != null) {
                    android.util.Log.e(tag, message, t)
                } else {
                    android.util.Log.e(tag, message)
                }
            }
        }
    }
}
