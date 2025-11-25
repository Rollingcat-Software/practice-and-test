package com.rollingcatsoftware.universalnfcreader

import android.app.PendingIntent
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
import android.util.Log
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.rollingcatsoftware.universalnfcreader.security.BiometricAuthManager
import com.rollingcatsoftware.universalnfcreader.ui.MainViewModel
import com.rollingcatsoftware.universalnfcreader.ui.screens.AppScreen
import com.rollingcatsoftware.universalnfcreader.ui.screens.AuthenticationScreen
import com.rollingcatsoftware.universalnfcreader.ui.theme.UniversalNFCReaderTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

/**
 * Main activity handling NFC tag discovery and UI rendering.
 *
 * Uses foreground dispatch to ensure this app receives NFC tags when visible.
 * Requires biometric authentication on launch if device supports it.
 */
@AndroidEntryPoint
class MainActivity : FragmentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private val viewModel: MainViewModel by viewModels()

    private var nfcAdapter: NfcAdapter? = null
    private var pendingIntent: PendingIntent? = null
    private var intentFilters: Array<IntentFilter>? = null
    private var techLists: Array<Array<String>>? = null

    private lateinit var biometricAuthManager: BiometricAuthManager

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
                            onOpenNfcSettings = { openNfcSettings() }
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

        // Observe NFC state changes
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.RESUMED) {
                updateNfcStatus()
            }
        }
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
            Log.d(TAG, "Biometric authentication not available - allowing access")
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
                Log.d(TAG, "Biometric authentication successful")
            },
            onError = { message ->
                Log.e(TAG, "Biometric authentication error: $message")
            },
            onCancel = {
                Log.d(TAG, "Biometric authentication cancelled")
            }
        )
    }

    /**
     * Setup NFC foreground dispatch.
     */
    private fun setupNfc() {
        nfcAdapter = NfcAdapter.getDefaultAdapter(this)

        if (nfcAdapter == null) {
            Log.w(TAG, "NFC is not available on this device")
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

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    /**
     * Enable NFC foreground dispatch for this activity.
     */
    private fun enableForegroundDispatch() {
        nfcAdapter?.let { adapter ->
            if (adapter.isEnabled) {
                try {
                    adapter.enableForegroundDispatch(
                        this,
                        pendingIntent,
                        intentFilters,
                        techLists
                    )
                    Log.d(TAG, "Foreground dispatch enabled")
                } catch (e: Exception) {
                    Log.e(TAG, "Error enabling foreground dispatch: ${e.message}")
                }
            }
        }
    }

    /**
     * Disable NFC foreground dispatch.
     */
    private fun disableForegroundDispatch() {
        nfcAdapter?.let { adapter ->
            try {
                adapter.disableForegroundDispatch(this)
                Log.d(TAG, "Foreground dispatch disabled")
            } catch (e: Exception) {
                Log.w(TAG, "Error disabling foreground dispatch: ${e.message}")
            }
        }
    }

    /**
     * Handle NFC intent.
     */
    private fun handleIntent(intent: Intent?) {
        intent ?: return

        val action = intent.action
        Log.d(TAG, "Received intent with action: $action")

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
                    Log.d(TAG, "Tag detected: ${it.id.joinToString("") { b -> "%02X".format(b) }}")
                    viewModel.onTagDetected(it)
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

        Log.d(TAG, "NFC status: available=$isAvailable, enabled=$isEnabled")
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
                Log.e(TAG, "Could not open settings: ${e2.message}")
            }
        }
    }
}
