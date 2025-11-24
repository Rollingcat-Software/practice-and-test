package com.turkey.eidnfc

import android.app.PendingIntent
import android.content.Intent
import android.content.IntentFilter
import android.nfc.NfcAdapter
import android.nfc.Tag
import android.nfc.tech.IsoDep
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.turkey.eidnfc.ui.MainScreen
import com.turkey.eidnfc.ui.MainViewModel
import com.turkey.eidnfc.ui.theme.TurkishEidNfcReaderTheme
import timber.log.Timber

/**
 * Main Activity for Turkish eID NFC Reader.
 *
 * This activity:
 * 1. Checks for NFC availability
 * 2. Enables foreground NFC dispatch
 * 3. Handles NFC tag detection
 * 4. Passes tags to ViewModel for processing
 */
class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()
    private var nfcAdapter: NfcAdapter? = null
    private var pendingIntent: PendingIntent? = null
    private var intentFilters: Array<IntentFilter>? = null
    private var techLists: Array<Array<String>>? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Timber for logging
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        }

        Timber.d("MainActivity onCreate")

        // Enable edge-to-edge display
        enableEdgeToEdge()

        // Check NFC availability
        nfcAdapter = NfcAdapter.getDefaultAdapter(this)

        if (nfcAdapter == null) {
            Timber.e("NFC is not available on this device")
            Toast.makeText(
                this,
                "NFC is not available on this device",
                Toast.LENGTH_LONG
            ).show()
        }

        // Setup NFC foreground dispatch
        setupNfcForegroundDispatch()

        // Setup UI
        setContent {
            TurkishEidNfcReaderTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val uiState by viewModel.uiState.collectAsState()
                    val pin by viewModel.pin.collectAsState()

                    MainScreen(
                        uiState = uiState,
                        pin = pin,
                        onPinChanged = viewModel::onPinChanged,
                        onResetClick = viewModel::resetState
                    )
                }
            }
        }

        // Handle intent if app was launched by NFC tag
        handleIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        Timber.d("MainActivity onResume")

        // Check if NFC is enabled
        nfcAdapter?.let { adapter ->
            if (!adapter.isEnabled) {
                Timber.w("NFC is disabled")
                Toast.makeText(
                    this,
                    "Please enable NFC in device settings",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                // Enable foreground dispatch
                enableForegroundDispatch()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        Timber.d("MainActivity onPause")

        // Disable foreground dispatch
        disableForegroundDispatch()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        Timber.d("MainActivity onNewIntent: ${intent.action}")
        handleIntent(intent)
    }

    /**
     * Sets up NFC foreground dispatch system.
     */
    private fun setupNfcForegroundDispatch() {
        // Create a PendingIntent to handle NFC intents
        val intent = Intent(this, javaClass).apply {
            addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
        }

        pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        // Create intent filters for NFC discovery
        val isoDepFilter = IntentFilter(NfcAdapter.ACTION_TECH_DISCOVERED)
        intentFilters = arrayOf(isoDepFilter)

        // Specify IsoDep technology
        techLists = arrayOf(arrayOf(IsoDep::class.java.name))
    }

    /**
     * Enables NFC foreground dispatch.
     */
    private fun enableForegroundDispatch() {
        nfcAdapter?.enableForegroundDispatch(
            this,
            pendingIntent,
            intentFilters,
            techLists
        )
        Timber.d("NFC foreground dispatch enabled")
    }

    /**
     * Disables NFC foreground dispatch.
     */
    private fun disableForegroundDispatch() {
        nfcAdapter?.disableForegroundDispatch(this)
        Timber.d("NFC foreground dispatch disabled")
    }

    /**
     * Handles NFC intent and extracts tag.
     */
    private fun handleIntent(intent: Intent?) {
        if (intent == null) {
            Timber.w("Intent is null")
            return
        }

        val action = intent.action
        Timber.d("Handling intent with action: $action")

        when (action) {
            NfcAdapter.ACTION_TECH_DISCOVERED,
            NfcAdapter.ACTION_TAG_DISCOVERED -> {
                val tag: Tag? = intent.getParcelableExtra(NfcAdapter.EXTRA_TAG)

                if (tag != null) {
                    Timber.d("NFC tag detected")
                    Timber.d("Tag ID: ${tag.id.joinToString(":") { String.format("%02X", it) }}")
                    Timber.d("Tech list: ${tag.techList.joinToString(", ")}")

                    // Check if IsoDep is supported
                    if (tag.techList.contains(IsoDep::class.java.name)) {
                        Timber.d("IsoDep supported, processing tag")
                        viewModel.onTagDetected(tag)
                    } else {
                        Timber.e("IsoDep not supported by this tag")
                        Toast.makeText(
                            this,
                            "This card does not support IsoDep",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                } else {
                    Timber.w("Tag is null in intent")
                }
            }
            else -> {
                Timber.d("Intent action not related to NFC: $action")
            }
        }
    }
}
