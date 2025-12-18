package com.rollingcatsoftware.universalnfcreader

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.nfc.NfcAdapter
import android.nfc.Tag
import android.nfc.tech.IsoDep
import android.nfc.tech.MifareClassic
import android.nfc.tech.MifareUltralight
import android.nfc.tech.Ndef
import android.nfc.tech.NfcA
import android.nfc.tech.NfcB
import android.nfc.tech.NfcF
import android.nfc.tech.NfcV
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.fragment.app.FragmentActivity
import com.rollingcatsoftware.universalnfcreader.security.BiometricAuthManager
import com.rollingcatsoftware.universalnfcreader.ui.MainViewModel
import com.rollingcatsoftware.universalnfcreader.ui.screens.AppScreen
import com.rollingcatsoftware.universalnfcreader.ui.screens.AuthenticationScreen
import com.rollingcatsoftware.universalnfcreader.ui.theme.UniversalNFCReaderTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * Main activity handling NFC tag discovery and UI rendering.
 *
 * Uses foreground dispatch to ensure this app receives NFC tags when visible.
 * Requires biometric authentication on launch if device supports it.
 *
 * Note: Samsung devices have issues with NFC Reader Mode - we use foreground
 * dispatch for better compatibility on Samsung devices.
 */
@AndroidEntryPoint
class MainActivity : FragmentActivity() {

    companion object {
        private const val TAG = "MainActivity"

        // Samsung device detection
        private val isSamsungDevice: Boolean by lazy {
            Build.MANUFACTURER.equals("samsung", ignoreCase = true)
        }
    }

    private val viewModel: MainViewModel by viewModels()

    private var nfcAdapter: NfcAdapter? = null
    private var pendingIntent: PendingIntent? = null
    private var intentFilters: Array<IntentFilter>? = null
    private var techLists: Array<Array<String>>? = null
    private var isUsingReaderMode = false

    private lateinit var biometricAuthManager: BiometricAuthManager

    /**
     * BroadcastReceiver for NFC adapter state changes.
     * Listens for NFC being enabled/disabled and updates UI accordingly.
     */
    private val nfcStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == NfcAdapter.ACTION_ADAPTER_STATE_CHANGED) {
                val state = intent.getIntExtra(
                    NfcAdapter.EXTRA_ADAPTER_STATE,
                    NfcAdapter.STATE_OFF
                )
                SecureLogger.d(TAG, "NFC adapter state changed: $state")

                when (state) {
                    NfcAdapter.STATE_ON -> {
                        SecureLogger.d(TAG, "NFC enabled - updating status and enabling dispatch")
                        updateNfcStatus()
                        // Re-enable foreground dispatch now that NFC is on
                        enableForegroundDispatch()
                    }
                    NfcAdapter.STATE_OFF -> {
                        SecureLogger.d(TAG, "NFC disabled - updating status")
                        updateNfcStatus()
                    }
                    NfcAdapter.STATE_TURNING_ON -> {
                        SecureLogger.d(TAG, "NFC turning on...")
                    }
                    NfcAdapter.STATE_TURNING_OFF -> {
                        SecureLogger.d(TAG, "NFC turning off...")
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setupNfc()
        setupBiometricAuth()

        setContent {
            UniversalNFCReaderTheme {
                val authState by biometricAuthManager.authenticationState.collectAsState()

                when (authState) {
                    is BiometricAuthManager.AuthenticationState.Authenticated -> {
                        // User is authenticated - show main app with navigation
                        AppScreen(
                            viewModel = viewModel,
                            onOpenNfcSettings = { openNfcSettings() },
                            onRefreshNfcStatus = { updateNfcStatus() }
                        )
                    }

                    else -> {
                        // Show authentication screen
                        AuthenticationScreen(
                            authState = authState,
                            onAuthenticate = { promptBiometricAuth() },
                            onExit = { finish() }
                        )
                    }
                }
            }
        }

        // Handle intent if activity was launched from NFC tag
        handleIntent(intent)

        // Register NFC state change receiver
        registerNfcStateReceiver()
    }

    /**
     * Setup biometric authentication.
     */
    private fun setupBiometricAuth() {
        biometricAuthManager = BiometricAuthManager(this)

        // Check if biometric authentication is required
        if (biometricAuthManager.isAuthenticationRequired()) {
            // Prompt for authentication
            promptBiometricAuth()
        } else {
            // No biometrics available - allow access
            SecureLogger.d(TAG, "Biometric authentication not available - allowing access")
            biometricAuthManager.markAsAuthenticated()
        }
    }

    /**
     * Prompt for biometric authentication.
     */
    private fun promptBiometricAuth() {
        biometricAuthManager.authenticate(
            activity = this,
            onSuccess = {
                SecureLogger.d(TAG, "Biometric authentication successful")
            },
            onError = { message ->
                SecureLogger.e(TAG, "Biometric authentication error: $message")
            },
            onCancel = {
                SecureLogger.d(TAG, "Biometric authentication cancelled")
            }
        )
    }

    /**
     * Register broadcast receiver for NFC adapter state changes.
     * This allows the app to react when NFC is enabled/disabled from
     * quick settings, notification panel, or system settings.
     */
    private fun registerNfcStateReceiver() {
        val filter = IntentFilter(NfcAdapter.ACTION_ADAPTER_STATE_CHANGED)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(nfcStateReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(nfcStateReceiver, filter)
        }
        SecureLogger.d(TAG, "NFC state receiver registered")
    }

    /**
     * Unregister NFC state broadcast receiver.
     */
    private fun unregisterNfcStateReceiver() {
        try {
            unregisterReceiver(nfcStateReceiver)
            SecureLogger.d(TAG, "NFC state receiver unregistered")
        } catch (e: IllegalArgumentException) {
            // Receiver was not registered
            SecureLogger.w(TAG, "NFC state receiver was not registered")
        }
    }

    /**
     * Setup NFC foreground dispatch.
     */
    private fun setupNfc() {
        nfcAdapter = NfcAdapter.getDefaultAdapter(this)

        if (nfcAdapter == null) {
            SecureLogger.w(TAG, "NFC is not available on this device")
            viewModel.updateNfcStatus(isAvailable = false, isEnabled = false)
            return
        }

        // Create pending intent for foreground dispatch
        val intent = Intent(this, javaClass).apply {
            addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }

        pendingIntent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            PendingIntent.getActivity(
                this, 0, intent,
                PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )
        } else {
            PendingIntent.getActivity(
                this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT
            )
        }

        // Setup intent filters for all NFC actions
        intentFilters = arrayOf(
            IntentFilter(NfcAdapter.ACTION_NDEF_DISCOVERED),
            IntentFilter(NfcAdapter.ACTION_TECH_DISCOVERED),
            IntentFilter(NfcAdapter.ACTION_TAG_DISCOVERED)
        )

        // Setup tech lists for tech discovered filter
        techLists = arrayOf(
            arrayOf(IsoDep::class.java.name),
            arrayOf(NfcA::class.java.name),
            arrayOf(NfcB::class.java.name),
            arrayOf(NfcF::class.java.name),
            arrayOf(NfcV::class.java.name),
            arrayOf(MifareClassic::class.java.name),
            arrayOf(MifareUltralight::class.java.name),
            arrayOf(Ndef::class.java.name)
        )
    }

    override fun onResume() {
        super.onResume()
        enableForegroundDispatch()
        updateNfcStatus()
    }

    override fun onPause() {
        super.onPause()
        disableForegroundDispatch()
    }

    override fun onDestroy() {
        super.onDestroy()
        unregisterNfcStateReceiver()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    /**
     * Enable NFC tag detection for this activity.
     *
     * Samsung devices have issues with Reader Mode causing tags not to be detected
     * on second tap. For Samsung, we use foreground dispatch which works more reliably.
     * For other devices, we use Reader Mode with platform sounds enabled.
     */
    private fun enableForegroundDispatch() {
        nfcAdapter?.let { adapter ->
            if (adapter.isEnabled) {
                if (isSamsungDevice) {
                    // Samsung devices: Use foreground dispatch (more reliable for re-reads)
                    enableForegroundDispatchMode()
                } else {
                    // Other devices: Try Reader Mode first, fallback to foreground dispatch
                    enableReaderModeWithSounds()
                }
            }
        }
    }

    /**
     * Enable Reader Mode with platform sounds (for non-Samsung devices).
     */
    private fun enableReaderModeWithSounds() {
        nfcAdapter?.let { adapter ->
            try {
                // Reader Mode flags - NO FLAG_READER_NO_PLATFORM_SOUNDS to keep sounds
                // FLAG_READER_SKIP_NDEF_CHECK is CRITICAL for passport/eID reading:
                // - Samsung devices check for NDEF first, which blocks encrypted MRTD cards
                // - Passports are not NDEF formatted, so the check fails and blocks reading
                val flags = NfcAdapter.FLAG_READER_NFC_A or
                        NfcAdapter.FLAG_READER_NFC_B or
                        NfcAdapter.FLAG_READER_NFC_F or
                        NfcAdapter.FLAG_READER_NFC_V or
                        NfcAdapter.FLAG_READER_SKIP_NDEF_CHECK

                // Bundle with extra options
                val options = Bundle().apply {
                    // Presence check delay helps prevent "Tag was lost" errors
                    putInt(NfcAdapter.EXTRA_READER_PRESENCE_CHECK_DELAY, 250)
                }

                adapter.enableReaderMode(
                    this,
                    { tag ->
                        // Handle tag on main thread
                        runOnUiThread {
                            SecureLogger.d(TAG, "Reader mode: Tag detected: ${tag.id.joinToString("") { b -> "%02X".format(b) }}")
                            handleTagDetected(tag)
                        }
                    },
                    flags,
                    options
                )
                isUsingReaderMode = true
                SecureLogger.d(TAG, "NFC Reader Mode enabled with sounds")
            } catch (e: Exception) {
                SecureLogger.e(TAG, "Error enabling reader mode: ${e.message}")
                enableForegroundDispatchMode()
            }
        }
    }

    /**
     * Enable foreground dispatch mode (works better on Samsung devices).
     */
    private fun enableForegroundDispatchMode() {
        nfcAdapter?.let { adapter ->
            if (adapter.isEnabled) {
                try {
                    adapter.enableForegroundDispatch(
                        this,
                        pendingIntent,
                        intentFilters,
                        techLists
                    )
                    isUsingReaderMode = false
                    SecureLogger.d(TAG, "Foreground dispatch enabled" + if (isSamsungDevice) " (Samsung device)" else "")
                } catch (e: Exception) {
                    SecureLogger.e(TAG, "Error enabling foreground dispatch: ${e.message}")
                }
            }
        }
    }

    /**
     * Handle detected NFC tag.
     *
     * Note: We do NOT automatically reset the reader mode here because:
     * 1. Passport/eID reading can take 30-45 seconds
     * 2. Resetting mid-read would interrupt the operation
     * 3. The reader mode will naturally allow re-detection when the card is removed and re-tapped
     */
    private fun handleTagDetected(tag: Tag) {
        viewModel.onTagDetected(tag)
    }

    /**
     * Disable NFC tag detection.
     */
    private fun disableForegroundDispatch() {
        nfcAdapter?.let { adapter ->
            try {
                if (isUsingReaderMode) {
                    adapter.disableReaderMode(this)
                    SecureLogger.d(TAG, "Reader mode disabled")
                } else {
                    adapter.disableForegroundDispatch(this)
                    SecureLogger.d(TAG, "Foreground dispatch disabled")
                }
            } catch (e: Exception) {
                SecureLogger.w(TAG, "Error disabling NFC: ${e.message}")
            }
        }
    }

    /**
     * Handle NFC intent.
     */
    private fun handleIntent(intent: Intent?) {
        intent ?: return

        val action = intent.action
        SecureLogger.d(TAG, "Received intent with action: $action")

        when (action) {
            NfcAdapter.ACTION_NDEF_DISCOVERED,
            NfcAdapter.ACTION_TECH_DISCOVERED,
            NfcAdapter.ACTION_TAG_DISCOVERED -> {
                val tag: Tag? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(NfcAdapter.EXTRA_TAG, Tag::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(NfcAdapter.EXTRA_TAG)
                }

                tag?.let {
                    SecureLogger.d(TAG, "Intent: Tag detected: ${it.id.joinToString("") { b -> "%02X".format(b) }}")
                    handleTagDetected(it)
                }
            }
        }
    }

    /**
     * Update NFC availability status.
     */
    private fun updateNfcStatus() {
        val adapter = nfcAdapter
        val isAvailable = adapter != null
        val isEnabled = adapter?.isEnabled == true

        viewModel.updateNfcStatus(isAvailable, isEnabled)

        SecureLogger.d(TAG, "NFC status: available=$isAvailable, enabled=$isEnabled")
    }

    /**
     * Open NFC settings.
     */
    fun openNfcSettings() {
        try {
            startActivity(Intent(Settings.ACTION_NFC_SETTINGS))
        } catch (e: Exception) {
            // Fallback to wireless settings if NFC settings not available
            try {
                startActivity(Intent(Settings.ACTION_WIRELESS_SETTINGS))
            } catch (e2: Exception) {
                SecureLogger.e(TAG, "Could not open settings: ${e2.message}")
            }
        }
    }
}
