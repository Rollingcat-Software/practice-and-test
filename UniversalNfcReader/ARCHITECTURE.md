# Universal NFC Reader - Architecture Deep Dive

## 🎯 Design Principles

1. **Open/Closed Principle** - Open for extension (new card types), closed for modification
2. **Single Responsibility** - Each reader handles one card type
3. **Dependency Inversion** - Depend on abstractions, not concretions
4. **Factory Pattern** - Centralized reader creation
5. **Strategy Pattern** - Different reading strategies for different cards

## 🏛️ Architectural Patterns

### 1. Detector Pattern

**Purpose**: Identify card type from NFC tag

```
┌─────────────────────────────────────────────────────────────┐
│                     UniversalCardDetector                    │
│  (Orchestrates multiple specific detectors)                  │
└───────────────────┬─────────────────────────────────────────┘
                    │
         ┌──────────┼──────────┬──────────┬──────────┐
         │          │          │          │          │
    ┌────▼────┐ ┌──▼────┐ ┌───▼────┐ ┌───▼────┐ ┌──▼────┐
    │ Turkish │ │Istanbul│ │Student │ │MIFARE  │ │  NDEF │
    │   eID   │ │  kart  │ │  Card  │ │Classic │ │Detector│
    │Detector │ │Detector│ │Detector│ │Detector│ └────────┘
    └─────────┘ └────────┘ └────────┘ └────────┘
```

**How It Works:**

1. Tag arrives → UniversalCardDetector receives it
2. Checks technology list (IsoDep, MifareClassic, etc.)
3. Delegates to specific detectors in priority order
4. Each detector tries to identify the card (AID selection, UID pattern, etc.)
5. First successful detection wins
6. Returns `CardType` enum value

**Example Detection Flow for Istanbulkart:**

```kotlin
// 1. Check if tag supports IsoDep
if (tag.techList.contains("android.nfc.tech.IsoDep")) {

    // 2. Try DESFire AID selection
    val isoDep = IsoDep.get(tag)
    isoDep.connect()

    // 3. SELECT DESFire application
    val desfireAid = byteArrayOf(0xD2, 0x76, 0x00, 0x00, 0x85, 0x01, 0x00)
    val response = isoDep.transceive(selectCommand(desfireAid))

    // 4. Check response status
    if (response.statusWord == 0x9100) {

        // 5. Check UID pattern (NXP manufacturer)
        if (tag.id[0] == 0x04.toByte()) {

            // 6. Confirmed Istanbulkart!
            return CardType.ISTANBULKART
        }
    }
}
```

### 2. Factory Pattern

**Purpose**: Create appropriate reader based on card type

```
┌─────────────────────────────────────────────────────────────┐
│                     CardReaderFactory                        │
│  createReader(cardType: CardType): CardReader?               │
└───────────────────┬─────────────────────────────────────────┘
                    │
         ┌──────────┼──────────┬──────────┬──────────┐
         │          │          │          │          │
    ┌────▼────┐ ┌──▼────┐ ┌───▼────┐ ┌───▼────┐ ┌──▼────┐
    │ Turkish │ │Istanbul│ │Student │ │MIFARE  │ │  NDEF │
    │   eID   │ │  kart  │ │  Card  │ │Classic │ │ Reader│
    │ Reader  │ │ Reader │ │ Reader │ │ Reader │ └────────┘
    └─────────┘ └────────┘ └────────┘ └────────┘
         │          │          │          │
         └──────────┴──────────┴──────────┴────────────┐
                                                        │
                                                 ┌──────▼──────┐
                                                 │ CardReader  │
                                                 │ (interface) │
                                                 └─────────────┘
```

**Benefits:**

- Single point of reader instantiation
- Easy to add new card types (just add new case)
- Centralized dependency injection
- Type-safe reader creation

**Usage:**

```kotlin
val factory = CardReaderFactory(context)
val reader = factory.createReader(CardType.TURKISH_EID)
    ?: throw UnsupportedCardException()

val result = reader.readCard(tag)
```

### 3. Strategy Pattern

**Purpose**: Different reading strategies for different authentication methods

```
┌─────────────────────────────────────────────────────────────┐
│                       CardReader                             │
│  readCard(tag: Tag): Result<CardData>                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
         ┌──────────┼──────────────┐
         │          │              │
    ┌────▼─────┐ ┌─▼──────────┐ ┌─▼────────────┐
    │ No Auth  │ │ BAC Auth   │ │  Key Auth    │
    │ Strategy │ │ Strategy   │ │  Strategy    │
    └──────────┘ └────────────┘ └──────────────┘
         │            │              │
         ▼            ▼              ▼
    Read UID     MRZ → Keys    Try default keys
    Read tech    Challenge     Authenticate
    info         Authenticate  Read sectors
                 Read DGs
```

**Implementation:**

```kotlin
interface CardReader {
    suspend fun readCard(tag: Tag): Result<CardData>
    fun requiresAuthentication(): Boolean
}

// No authentication needed
class MifareUltralightReader : CardReader {
    override fun requiresAuthentication() = false
    override suspend fun readCard(tag: Tag): Result<CardData> {
        // Direct read
    }
}

// Requires BAC authentication
class TurkishEidReader : CardReader, AuthenticatedCardReader {
    override fun requiresAuthentication() = true
    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData.MrzData
    ): Result<CardData> {
        // BAC authentication → Read
    }
}

// Tries default keys, falls back to auth if needed
class MifareClassicReader : CardReader {
    override fun requiresAuthentication() = false // Optional
    override suspend fun readCard(tag: Tag): Result<CardData> {
        // Try default keys
        // If fail, return partial data
    }
}
```

### 4. Chain of Responsibility

**Purpose**: Try multiple authentication methods in sequence

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Default Keys │─NO──▶│ Common Keys  │─NO──▶│ Ask User for │
│   Handler    │      │   Handler    │      │  Key Handler │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
      YES                   YES                   YES
       │                     │                     │
       ▼                     ▼                     ▼
    Read Data             Read Data            Read Data
```

**Implementation:**

```kotlin
class MifareClassicAuthChain {

    private val authHandlers = listOf(
        DefaultKeysHandler(),
        CommonKeysHandler(),
        UserProvidedKeysHandler()
    )

    suspend fun authenticate(mifare: MifareClassic, sector: Int): ByteArray? {
        for (handler in authHandlers) {
            val result = handler.tryAuthenticate(mifare, sector)
            if (result != null) {
                return handler.readData(mifare, sector)
            }
        }
        return null // All methods failed
    }
}
```

## 📊 Data Flow

### Complete Card Reading Flow

```
1. Tag Detected (NFC)
         │
         ▼
2. NfcCardReadingService.readCard(tag)
         │
         ├──▶ UniversalCardDetector.detectCardType(tag)
         │         │
         │         ├──▶ TurkishEidDetector.detect(tag)
         │         ├──▶ IstanbulkartDetector.detect(tag)
         │         └──▶ ... other detectors
         │         │
         │         ▼
         │    CardType.ISTANBULKART
         │
         ├──▶ CardReaderFactory.createReader(CardType.ISTANBULKART)
         │         │
         │         ▼
         │    IstanbulkartReader instance
         │
         ├──▶ IstanbulkartReader.readCard(tag)
         │         │
         │         ├──▶ Connect to IsoDep
         │         ├──▶ GET_VERSION (DESFire)
         │         ├──▶ GET_APPLICATION_IDS
         │         ├──▶ Read UID
         │         └──▶ Parse data
         │         │
         │         ▼
         │    IstanbulkartData(uid, version, appIds, ...)
         │
         ▼
3. CardReadResult.Success(IstanbulkartData)
         │
         ▼
4. UI Display
```

### Error Handling Flow

```
Tag Detected
     │
     ▼
Try Read
     │
     ├──▶ Connection Lost ──▶ CardError.ConnectionLost
     │                             │
     ├──▶ Timeout ──────────▶ CardError.Timeout
     │                             │
     ├──▶ Auth Required ────▶ CardError.AuthenticationRequired
     │                             │
     ├──▶ Auth Failed ──────▶ CardError.AuthenticationFailed
     │                             │
     └──▶ Unknown ──────────▶ CardError.UnknownError(message)
                                   │
                                   ▼
                            CardReadResult.Failure(cardType, error)
                                   │
                                   ▼
                            UI shows actionable error message
```

## 🔧 Component Responsibilities

### NfcCardReadingService

**Responsibility**: Orchestrate entire reading process

```kotlin
class NfcCardReadingService(private val context: Context) {

    private val detector = UniversalCardDetector()
    private val factory = CardReaderFactory(context)

    suspend fun readCard(tag: Tag): CardReadResult {
        // 1. Detect type
        val cardType = detector.detectCardType(tag)

        // 2. Create reader
        val reader = factory.createReader(cardType)
            ?: return CardReadResult.UnsupportedCard(cardType)

        // 3. Check auth requirement
        if (reader.requiresAuthentication()) {
            return CardReadResult.AuthenticationRequired(cardType)
        }

        // 4. Read card
        return when (val result = reader.readCard(tag)) {
            is Result.Success -> CardReadResult.Success(result.data)
            is Result.Error -> CardReadResult.Failure(cardType, result.error)
        }
    }
}
```

**Inputs:**
- NFC Tag from Android NFC system

**Outputs:**
- `CardReadResult.Success(CardData)` - Successfully read
- `CardReadResult.AuthenticationRequired` - Needs credentials
- `CardReadResult.UnsupportedCard` - Unknown card type
- `CardReadResult.Failure(error)` - Read failed
- `CardReadResult.Exception` - Unexpected error

### UniversalCardDetector

**Responsibility**: Identify card type

**Decision Tree:**

```
Tag
 │
 ├─ Has NDEF? ─YES─▶ CardType.NDEF
 │     │
 │    NO
 │     ▼
 ├─ Has IsoDep?
 │     │
 │    YES ─▶ Try Turkish eID AID ─YES─▶ CardType.TURKISH_EID
 │     │           │
 │     │          NO
 │     │           ▼
 │     │      Try DESFire AID ─YES─▶ Check UID pattern ─YES─▶ CardType.ISTANBULKART
 │     │           │                       │
 │     │          NO                      NO
 │     │           ▼                       ▼
 │     │      CardType.ISO_DEP_UNKNOWN   CardType.MIFARE_DESFIRE
 │     │
 │    NO
 │     ▼
 ├─ Has MifareClassic? ─YES─▶ CardType.MIFARE_CLASSIC
 │     │
 │    NO
 │     ▼
 ├─ Has MifareUltralight? ─YES─▶ CardType.MIFARE_ULTRALIGHT
 │     │
 │    NO
 │     ▼
 └─ CardType.UNKNOWN
```

### CardReader Implementations

Each reader is responsible for:

1. **Technology Selection** - Get correct tag technology (IsoDep, MifareClassic, etc.)
2. **Connection Management** - Connect, set timeout, close
3. **Command Execution** - Send APDU commands or tag-specific commands
4. **Response Parsing** - Extract data from responses
5. **Error Handling** - Map exceptions to `CardError`
6. **Data Construction** - Build `CardData` object

**Example: IstanbulkartReader**

```kotlin
class IstanbulkartReader : CardReader {

    override suspend fun readCard(tag: Tag): Result<CardData> {
        val isoDep = IsoDep.get(tag) ?: return Result.Error(CardError.UnsupportedCard)

        return try {
            // 1. Technology selection
            isoDep.connect()
            isoDep.timeout = 3000

            // 2. Command execution
            val version = getDESFireVersion(isoDep)
            val appIds = getApplicationIds(isoDep)
            val uid = tag.id

            // 3. Data construction
            val data = IstanbulkartData(
                uid = uid.toHexString(),
                cardType = CardType.ISTANBULKART,
                readTimestamp = System.currentTimeMillis(),
                desfireVersion = version,
                applicationIds = appIds.map { it.toHexString() },
                rawData = mapOf(
                    "version" to version,
                    "appIds" to appIds
                )
            )

            Result.Success(data)

        } catch (e: TagLostException) {
            // 4. Error handling
            Result.Error(CardError.ConnectionLost)
        } catch (e: IOException) {
            Result.Error(CardError.Timeout)
        } finally {
            // 5. Connection management
            isoDep.close()
        }
    }

    override fun getSupportedCardTypes() = listOf(CardType.ISTANBULKART)
    override fun requiresAuthentication() = false
}
```

## 📐 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          <<interface>>                          │
│                          CardDetector                           │
│  + detectCardType(tag: Tag): CardType                           │
│  + getSupportedTechnologies(tag: Tag): List<String>             │
└────────────────────────────▲────────────────────────────────────┘
                             │
                             │implements
                             │
                ┌────────────┴────────────┐
                │ UniversalCardDetector   │
                │ - detectors: List<...>  │
                └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          <<interface>>                          │
│                            CardReader                           │
│  + readCard(tag: Tag): Result<CardData>                         │
│  + getSupportedCardTypes(): List<CardType>                      │
│  + requiresAuthentication(): Boolean                            │
└────────────────────────────▲────────────────────────────────────┘
                             │
                ┌────────────┼────────────┬────────────┐
                │            │            │            │
       ┌────────┴──────┐  ┌─┴──────────┐ ┌┴──────────┐ ┌──────────┐
       │ TurkishEid    │  │Istanbulkart│ │StudentCard│ │  MIFARE  │
       │    Reader     │  │   Reader   │ │  Reader   │ │  Reader  │
       └───────────────┘  └────────────┘ └───────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         <<sealed class>>                        │
│                            CardData                             │
│  + uid: String                                                  │
│  + cardType: CardType                                           │
│  + readTimestamp: Long                                          │
│  + rawData: Map<String, Any>                                    │
└────────────────────────────▲────────────────────────────────────┘
                             │
                ┌────────────┼────────────┬────────────┐
                │            │            │            │
       ┌────────┴──────┐  ┌─┴──────────┐ ┌┴──────────┐ ┌──────────┐
       │ TurkishEid    │  │Istanbulkart│ │StudentCard│ │ Generic  │
       │    Data       │  │    Data    │ │   Data    │ │   Data   │
       └───────────────┘  └────────────┘ └───────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      CardReaderFactory                          │
│  + createReader(cardType: CardType): CardReader?                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   NfcCardReadingService                         │
│  - detector: CardDetector                                       │
│  - factory: CardReaderFactory                                   │
│  + readCard(tag: Tag): CardReadResult                           │
│  + readCardWithAuth(tag, auth): CardReadResult                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Sequence Diagrams

### Successful Card Read

```
User        MainActivity    ViewModel    NfcService    Detector    Factory    Reader    Tag
 │               │             │            │             │           │          │        │
 │  Hold card    │             │            │             │           │          │        │
 ├──────────────▶│             │            │             │           │          │        │
 │               │ onTagDetected()          │             │           │          │        │
 │               ├────────────▶│            │             │           │          │        │
 │               │             │ readCard() │             │           │          │        │
 │               │             ├───────────▶│             │           │          │        │
 │               │             │            │ detectCardType()        │          │        │
 │               │             │            ├────────────▶│           │          │        │
 │               │             │            │             │ tryDetect()          │        │
 │               │             │            │             ├──────────────────────▶│
 │               │             │            │             │  SELECT AID           │
 │               │             │            │             │◀──────────────────────┤
 │               │             │            │   CardType  │           │          │        │
 │               │             │            │◀────────────┤           │          │        │
 │               │             │            │ createReader()          │          │        │
 │               │             │            ├──────────────────────▶│          │        │
 │               │             │            │            Reader      │          │        │
 │               │             │            │◀──────────────────────┤          │        │
 │               │             │            │ readCard()                         │        │
 │               │             │            ├───────────────────────────────────▶│
 │               │             │            │                        GET_VERSION  │
 │               │             │            │                                    ├───────▶│
 │               │             │            │                          Response  │        │
 │               │             │            │                                    │◀───────┤
 │               │             │            │                         CardData   │        │
 │               │             │            │◀───────────────────────────────────┤
 │               │             │ Success(CardData)                   │          │        │
 │               │             │◀───────────┤            │           │          │        │
 │               │ update UI   │            │             │           │          │        │
 │               │◀────────────┤            │             │           │          │        │
 │  See card     │             │            │             │           │          │        │
 │  details      │             │            │             │           │          │        │
 │◀──────────────┤             │            │             │           │          │        │
```

### Authentication Required Flow

```
User        MainActivity    ViewModel    NfcService    Reader    Dialog
 │               │             │            │            │          │
 │  Hold card    │             │            │            │          │
 ├──────────────▶│             │            │            │          │
 │               │ onTagDetected()          │            │          │
 │               ├────────────▶│            │            │          │
 │               │             │ readCard() │            │          │
 │               │             ├───────────▶│            │          │
 │               │             │            │ requiresAuthentication()
 │               │             │            ├───────────▶│
 │               │             │            │    true    │          │
 │               │             │            │◀───────────┤          │
 │               │             │ AuthenticationRequired  │          │
 │               │             │◀───────────┤            │          │
 │               │ showAuthDialog()         │            │          │
 │               ├─────────────────────────────────────────────────▶│
 │  Enter MRZ    │             │            │            │          │
 │  data         │             │            │            │          │
 │◀──────────────────────────────────────────────────────────────────┤
 │───────────────────────────────────────────────────────────────────▶│
 │               │ onAuthProvided(MRZ)      │            │          │
 │               │◀──────────────────────────────────────────────────┤
 │               ├────────────▶│            │            │          │
 │               │             │ readCardWithAuth(tag, MRZ)         │
 │               │             ├───────────▶│            │          │
 │               │             │            │ readCardWithAuth()    │
 │               │             │            ├───────────▶│
 │               │             │            │  BAC Auth  │          │
 │               │             │            │  Read DGs  │          │
 │               │             │            │ CardData   │          │
 │               │             │            │◀───────────┤          │
 │               │             │ Success    │            │          │
 │               │             │◀───────────┤            │          │
 │               │ display     │            │            │          │
 │               │◀────────────┤            │            │          │
 │  See data     │             │            │            │          │
 │◀──────────────┤             │            │            │          │
```

## 💡 Design Decisions

### Why Sealed Classes for CardData?

**Benefits:**
- Type-safe polymorphism
- Exhaustive when expressions
- Compile-time guarantees
- Easy to add new types

```kotlin
when (cardData) {
    is TurkishEidData -> displayEidInfo(cardData)
    is IstanbulkartData -> displayTransportInfo(cardData)
    is StudentCardData -> displayStudentInfo(cardData)
    is GenericCardData -> displayGenericInfo(cardData)
    // Compiler ensures all cases covered
}
```

### Why Suspend Functions?

NFC operations are I/O bound and may take time:
- Avoid blocking UI thread
- Easy cancellation
- Structured concurrency
- Clean error handling with coroutines

```kotlin
suspend fun readCard(tag: Tag): Result<CardData> = withContext(Dispatchers.IO) {
    // I/O operations here
    // Cancellable
    // Exception safe
}
```

### Why Factory Pattern?

- **Single Responsibility**: Factory only creates readers
- **Open/Closed**: Add new readers without modifying existing code
- **Dependency Management**: Centralized reader construction
- **Testability**: Easy to mock factory in tests

### Why Result Wrapper?

Instead of throwing exceptions:

```kotlin
// Bad: Exceptions for control flow
try {
    val data = readCard(tag)
} catch (e: AuthException) {
    // Handle
} catch (e: IOException) {
    // Handle
}

// Good: Explicit result types
when (val result = readCard(tag)) {
    is Result.Success -> use(result.data)
    is Result.Error -> handle(result.error)
}
```

**Benefits:**
- Explicit error handling
- Type-safe
- No hidden control flow
- Functional programming style

---

**This architecture is production-ready, scalable, and maintainable.** 🚀
