# Universal NFC Card Reader

A modular, generic Android NFC card reader that can detect and read multiple card types including Turkish eID, Istanbulkart, student cards, MIFARE cards, and more.

## 🎯 Features

### Supported Card Types

- **Turkish eID (ePassport)** - ISO 7816-4 with BAC authentication
- **Istanbulkart** - MIFARE DESFire transport card (basic info without keys)
- **Student Cards** - MIFARE Classic, MIFARE DESFire
- **MIFARE Classic** - 1K/4K cards (with default keys)
- **MIFARE DESFire** - Version info, application list
- **MIFARE Ultralight** - Simple memory cards
- **ISO 15693** - NfcV tags
- **NDEF** - NFC Forum formatted tags
- **Generic NFC-A/B/F** - Basic UID and tech info

### Key Capabilities

✅ **Automatic Card Detection** - Identifies card type automatically
✅ **Modular Architecture** - Easy to add new card types
✅ **Factory Pattern** - Clean reader instantiation
✅ **Unified Data Model** - Consistent CardData interface
✅ **No Authentication** - Reads public data without keys
✅ **Authentication Support** - Can read protected data with credentials
✅ **UID Reading** - Always reads card unique identifier
✅ **Technology Detection** - Identifies all supported NFC technologies

## 🏗️ Architecture

### Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        Presentation Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ MainActivity │  │  MainScreen  │  │  CardDetails │      │
│  │  (Activity)  │  │   (Compose)  │  │   (Compose)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│           │                  │                  │            │
│           └──────────────────┴──────────────────┘            │
│                              │                                │
│                    ┌─────────▼─────────┐                    │
│                    │   MainViewModel    │                    │
│                    └─────────┬─────────┘                    │
└──────────────────────────────┼──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                        Domain Layer                          │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  ReadCardUseCase │        │   CardData       │          │
│  │  (Business Logic)│        │   (Models)       │          │
│  └──────────────────┘        └──────────────────┘          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                         Data Layer                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              NfcCardReadingService                     │  │
│  │         (Orchestrates detection & reading)             │  │
│  └─────────────────┬─────────────────────────────────────┘  │
│                    │                                          │
│      ┌─────────────┼─────────────┬─────────────┐            │
│      │             │             │             │            │
│  ┌───▼──────┐  ┌──▼────────┐  ┌─▼───────┐  ┌─▼────────┐   │
│  │  Card    │  │  Reader   │  │  Data   │  │  Card    │   │
│  │ Detector │  │  Factory  │  │ Parsers │  │ Readers  │   │
│  └──────────┘  └───────────┘  └─────────┘  └──────────┘   │
│                                                              │
│  ┌───────────────── Card Readers ────────────────────────┐  │
│  │  TurkishEid │ Istanbulkart │ StudentCard │ MIFARE │   │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Card Detector (`CardDetector.kt`)

Identifies card type by examining:
- Technology list (IsoDep, MifareClassic, etc.)
- UID patterns
- AID selection attempts
- Manufacturer codes

```kotlin
interface CardDetector {
    fun detectCardType(tag: Tag): CardType
    fun getSupportedTechnologies(tag: Tag): List<String>
}
```

**Implementations:**
- `UniversalCardDetector` - Orchestrates all specific detectors
- `TurkishEidDetector` - Detects MRTD AID
- `IstanbulkartDetector` - Detects DESFire + UID pattern
- `StudentCardDetector` - Detects common student card patterns

#### 2. Card Reader Factory (`CardReaderFactory.kt`)

Creates appropriate reader based on detected card type using Factory Pattern.

```kotlin
interface CardReaderFactory {
    fun createReader(cardType: CardType): CardReader?
}
```

#### 3. Card Reader Interface (`CardReader.kt`)

Abstract interface for all card readers.

```kotlin
interface CardReader {
    suspend fun readCard(tag: Tag): Result<CardData>
    fun getSupportedCardTypes(): List<CardType>
    fun requiresAuthentication(): Boolean
}
```

**Implementations:**
- `TurkishEidReader` - Reads Turkish eID (requires MRZ for BAC)
- `IstanbulkartReader` - Reads Istanbulkart (UID + structure only)
- `StudentCardClassicReader` - Reads MIFARE Classic student cards
- `StudentCardDesfireReader` - Reads DESFire student cards
- `MifareClassicReader` - Generic MIFARE Classic reader
- `MifareDesfireReader` - Generic DESFire reader
- `MifareUltralightReader` - MIFARE Ultralight reader
- `Iso15693Reader` - NfcV tag reader
- `NdefReader` - NDEF formatted tag reader

#### 4. Data Parsers (`CardDataParser.kt`)

Parse raw card data into structured models.

```kotlin
interface CardDataParser<T> {
    fun parse(rawData: ByteArray): T?
    fun validate(data: T): Boolean
}
```

#### 5. Unified Data Model (`CardData.kt`)

All card readers return `CardData` sealed class:

```kotlin
sealed class CardData {
    abstract val uid: String
    abstract val cardType: CardType
    abstract val readTimestamp: Long
    abstract val rawData: Map<String, Any>
}

// Specific implementations:
data class TurkishEidData(...)
data class IstanbulkartData(...)
data class StudentCardData(...)
data class GenericCardData(...)
```

## 📦 Project Structure

```
UniversalNfcReader/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/universal/nfcreader/
│   │   │   │   ├── data/
│   │   │   │   │   ├── nfc/
│   │   │   │   │   │   ├── readers/          # Card-specific readers
│   │   │   │   │   │   │   ├── TurkishEidReader.kt
│   │   │   │   │   │   │   ├── IstanbulkartReader.kt
│   │   │   │   │   │   │   ├── StudentCardClassicReader.kt
│   │   │   │   │   │   │   ├── MifareClassicReader.kt
│   │   │   │   │   │   │   ├── MifareDesfireReader.kt
│   │   │   │   │   │   │   └── ...
│   │   │   │   │   │   ├── detectors/        # Card type detectors
│   │   │   │   │   │   │   ├── CardDetector.kt
│   │   │   │   │   │   │   ├── TurkishEidDetector.kt
│   │   │   │   │   │   │   ├── IstanbulkartDetector.kt
│   │   │   │   │   │   │   └── ...
│   │   │   │   │   │   ├── parsers/          # Data parsers
│   │   │   │   │   │   │   ├── CardDataParser.kt
│   │   │   │   │   │   │   ├── MrzParser.kt
│   │   │   │   │   │   │   └── ...
│   │   │   │   │   │   ├── CardReader.kt     # Reader interface
│   │   │   │   │   │   ├── CardReaderFactory.kt
│   │   │   │   │   │   ├── NfcCardReadingService.kt
│   │   │   │   │   │   └── UidReader.kt
│   │   │   │   │   └── repository/
│   │   │   │   │       └── CardRepository.kt
│   │   │   │   ├── domain/
│   │   │   │   │   ├── model/
│   │   │   │   │   │   ├── CardData.kt       # Sealed class for all card types
│   │   │   │   │   │   ├── CardType.kt       # Enum of supported types
│   │   │   │   │   │   ├── CardError.kt      # Error types
│   │   │   │   │   │   └── Result.kt         # Result wrapper
│   │   │   │   │   └── usecase/
│   │   │   │   │       ├── ReadCardUseCase.kt
│   │   │   │   │       └── DetectCardTypeUseCase.kt
│   │   │   │   ├── ui/
│   │   │   │   │   ├── screens/
│   │   │   │   │   │   ├── MainActivity.kt
│   │   │   │   │   │   ├── MainScreen.kt     # Card reading screen
│   │   │   │   │   │   └── CardDetailsScreen.kt
│   │   │   │   │   ├── components/
│   │   │   │   │   │   ├── CardInfoCard.kt
│   │   │   │   │   │   ├── TechnologyChip.kt
│   │   │   │   │   │   └── UidDisplay.kt
│   │   │   │   │   ├── theme/
│   │   │   │   │   │   ├── Color.kt
│   │   │   │   │   │   ├── Theme.kt
│   │   │   │   │   │   └── Type.kt
│   │   │   │   │   └── MainViewModel.kt
│   │   │   │   └── di/
│   │   │   │       └── AppModule.kt          # Hilt DI
│   │   │   ├── AndroidManifest.xml
│   │   │   └── res/
│   │   ├── test/                             # Unit tests
│   │   └── androidTest/                      # UI tests
│   └── build.gradle
├── build.gradle
├── settings.gradle
└── README.md                                 # This file
```

## 🚀 Usage

### Basic Card Reading

```kotlin
class MainActivity : ComponentActivity() {

    private val nfcAdapter: NfcAdapter? by lazy {
        NfcAdapter.getDefaultAdapter(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            UniversalNfcReaderTheme {
                MainScreen(
                    onCardDetected = { tag ->
                        handleNfcTag(tag)
                    }
                )
            }
        }
    }

    private fun handleNfcTag(tag: Tag) {
        lifecycleScope.launch {
            val service = NfcCardReadingService(this@MainActivity)
            when (val result = service.readCard(tag)) {
                is CardReadResult.Success -> {
                    // Display card data
                    displayCardData(result.cardData)
                }
                is CardReadResult.AuthenticationRequired -> {
                    // Show authentication dialog
                    showAuthDialog(result.cardType)
                }
                is CardReadResult.UnsupportedCard -> {
                    // Show unsupported message
                    showError("Unsupported card: ${result.cardType}")
                }
                // Handle other cases...
            }
        }
    }
}
```

### Card Type Detection Only

```kotlin
val detector = UniversalCardDetector()
val cardType = detector.detectCardType(tag)
val technologies = detector.getSupportedTechnologies(tag)

when (cardType) {
    CardType.TURKISH_EID -> println("Turkish eID detected")
    CardType.ISTANBULKART -> println("Istanbulkart detected")
    CardType.MIFARE_CLASSIC -> println("MIFARE Classic detected")
    else -> println("Unknown card")
}
```

### Reading with Authentication

```kotlin
val service = NfcCardReadingService(context)
val mrzData = AuthenticationData.MrzData(
    documentNumber = "U12345678",
    dateOfBirth = "900115",
    dateOfExpiry = "301231"
)

val result = service.readCardWithAuth(tag, mrzData)
```

## 📊 Card Type Capabilities

### What Can Be Read WITHOUT Authentication

| Card Type | UID | Tech Info | Balance | Personal Data | Transaction History |
|-----------|-----|-----------|---------|---------------|---------------------|
| Turkish eID | ✅ | ✅ | N/A | ❌ (needs MRZ) | N/A |
| Istanbulkart | ✅ | ✅ | ❌ (needs keys) | N/A | ❌ (needs keys) |
| Student Card (Classic) | ✅ | ✅ | ⚠️ (if default keys) | ⚠️ (if default keys) | ⚠️ (if default keys) |
| Student Card (DESFire) | ✅ | ✅ | ❌ (needs keys) | ❌ (needs keys) | ❌ (needs keys) |
| MIFARE Classic | ✅ | ✅ | ⚠️ (if default keys) | ⚠️ (if default keys) | ⚠️ (if default keys) |
| MIFARE DESFire | ✅ | ✅ | ❌ (needs keys) | ❌ (needs keys) | ❌ (needs keys) |
| MIFARE Ultralight | ✅ | ✅ | ✅ | ✅ (if stored) | N/A |
| ISO 15693 | ✅ | ✅ | ⚠️ (depends) | ⚠️ (depends) | ⚠️ (depends) |
| NDEF | ✅ | ✅ | N/A | ✅ (if formatted) | N/A |

Legend:
- ✅ Always readable
- ❌ Requires authentication/keys
- ⚠️ May be readable with default/common keys
- N/A Not applicable

## 🔐 Security Considerations

### Data Protection

- **Turkish eID**: Uses BAC (Basic Access Control) with MRZ-derived keys
- **Istanbulkart**: DESFire authentication keys are proprietary
- **Student Cards**: Keys may be default or custom per university
- **MIFARE Classic**: Crypto-1 is broken but still provides basic security

### Best Practices

1. **Never log sensitive data** (PINs, keys, personal information)
2. **Clear sensitive data** from memory after use
3. **Use secure storage** for authentication credentials
4. **Validate all input** before processing
5. **Handle errors gracefully** without leaking information

## 🧪 Testing

### Unit Tests

```bash
./gradlew test
```

Tests cover:
- Card detection logic
- Reader factory
- Data parsers
- UID extraction
- Error handling

### Android Instrumentation Tests

```bash
./gradlew connectedAndroidTest
```

Tests require:
- Physical device with NFC
- Sample NFC cards for testing

## 📚 Dependencies

```gradle
dependencies {
    // Kotlin
    implementation "org.jetbrains.kotlin:kotlin-stdlib:1.9.0"
    implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3"

    // Jetpack Compose
    implementation "androidx.compose.ui:ui:1.5.4"
    implementation "androidx.compose.material3:material3:1.1.2"
    implementation "androidx.activity:activity-compose:1.8.0"

    // Hilt DI
    implementation "com.google.dagger:hilt-android:2.48"
    kapt "com.google.dagger:hilt-compiler:2.48"

    // Timber logging
    implementation "com.jakewharton.timber:timber:5.0.1"

    // Testing
    testImplementation "junit:junit:4.13.2"
    testImplementation "io.mockk:mockk:1.13.8"
}
```

## 🎨 UI Screenshots

```
┌──────────────────────────────────────────┐
│  Universal NFC Card Reader               │
├──────────────────────────────────────────┤
│                                          │
│          [NFC Icon]                      │
│                                          │
│      Hold card near device               │
│                                          │
├──────────────────────────────────────────┤
│  Supported Cards:                        │
│  ✓ Turkish eID                           │
│  ✓ Istanbulkart                          │
│  ✓ Student Cards                         │
│  ✓ MIFARE Cards                          │
│  ✓ Generic NFC Tags                      │
└──────────────────────────────────────────┘
```

## 🤝 Contributing

### Adding a New Card Type

1. **Create Detector**
   ```kotlin
   class MyCardDetector : SpecificCardDetector {
       override fun detect(tag: Tag): CardType {
           // Detection logic
       }
   }
   ```

2. **Create Reader**
   ```kotlin
   class MyCardReader : CardReader {
       override suspend fun readCard(tag: Tag): Result<CardData> {
           // Reading logic
       }
   }
   ```

3. **Create Data Model**
   ```kotlin
   data class MyCardData(
       override val uid: String,
       // Custom fields
   ) : CardData()
   ```

4. **Register in Factory**
   ```kotlin
   override fun createReader(cardType: CardType): CardReader? {
       return when (cardType) {
           CardType.MY_CARD -> MyCardReader()
           // ...
       }
   }
   ```

## 📖 References

### Standards

- **ISO 14443-3**: Contactless cards (Type A/B)
- **ISO 14443-4**: Transmission protocol (IsoDep)
- **ISO 7816-4**: Smart card APDU commands
- **ISO 15693**: Vicinity cards (NfcV)
- **ICAO Doc 9303**: Machine Readable Travel Documents (MRTD)

### Documentation

- [Android NFC Guide](https://developer.android.com/guide/topics/connectivity/nfc)
- [MIFARE Documentation](https://www.nxp.com/products/rfid-nfc/mifare-ics:MIFARE_ICS)
- [NFC Forum Specifications](https://nfc-forum.org/our-work/specifications-and-application-documents/specifications/)

## 📄 License

```
Copyright 2025 Universal NFC Reader

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 🙋 FAQ

**Q: Can I read Istanbulkart balance?**
A: Not without the proprietary DESFire keys from Istanbul transport authority. You can read UID and application structure.

**Q: What if my student card uses custom keys?**
A: The app tries common default keys. For custom keys, you'd need to provide them through the authentication interface.

**Q: Does this work on all Android devices?**
A: Works on Android devices with NFC hardware (Android 4.4+). Check `NfcAdapter.getDefaultAdapter(context) != null`.

**Q: Can I write to cards?**
A: Currently read-only. Writing requires additional permissions and authentication.

**Q: Is Turkish eID reading included?**
A: Yes, with BAC authentication using MRZ data (same as the Turkish eID Reader project).

---

**Built with ❤️ for the NFC community**
