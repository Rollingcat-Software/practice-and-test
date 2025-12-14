# Passport NFC Reader Implementation Roadmap

## Executive Summary

This document provides a comprehensive roadmap for implementing passport NFC reading functionality, building upon the existing Turkish eID NFC Reader architecture. The implementation follows **strict compliance** with the Security & Engineering Checklist (se-checklist.md).

**Target Project**: UniversalNfcReader (primary) or TurkishEidNfcReader (extended)
**Document Standard**: ICAO Doc 9303 (Machine Readable Travel Documents)
**Authentication Protocol**: BAC (Basic Access Control) - same as Turkish eID
**MRZ Format**: TD3 (2 lines, 44 characters each) vs TD1 (3 lines, 30 chars)

---

## Part 1: Strict Compliance Audit Against SE-Checklist

### 1.1 Security Requirements Compliance Matrix

| Requirement | Current Status | Passport Implementation | Priority |
|-------------|----------------|------------------------|----------|
| **PIN Handling** ||||
| Never log/store PIN in plain text | N/A (MRZ-based) | MRZ data same treatment | P0 |
| Clear sensitive bytes after use | PARTIAL | Must implement SecureByteArray | P0 |
| Use SecureRandom | YES | Reuse BacAuthentication | P0 |
| **Data Protection** ||||
| Never persist sensitive data unencrypted | YES (in-memory only) | Maintain same approach | P0 |
| Use Android Keystore for encryption keys | NOT NEEDED | Only if adding export feature | P1 |
| No analytics on sensitive data | YES | Maintain | P0 |
| **NFC Security** ||||
| Verify AID matches expected | YES | Add passport AID verification | P0 |
| Validate APDU responses | YES | Reuse ApduHelper | P0 |
| Implement timeout handling | YES (30s) | Reuse existing | P0 |
| **Cryptographic Operations** ||||
| SOD validation with CSCA | PARTIAL (60%) | Complete for passports | P1 |
| Validate certificate chain | PARTIAL | Implement full chain | P1 |
| Check certificate expiration | YES | Reuse SodValidator | P0 |
| Hash verification of DGs | PARTIAL | Implement DG hash verification | P1 |
| **Code Security** ||||
| Input validation | YES | Extend for TD3 MRZ | P0 |
| No PII in logs | YES | Maintain Timber redaction | P0 |
| Fail securely | YES | Maintain Result pattern | P0 |

### 1.2 Engineering Requirements Compliance Matrix

| Requirement | Current Status | Passport Implementation | Priority |
|-------------|----------------|------------------------|----------|
| **Architecture** ||||
| Clean Architecture | YES | Extend existing layers | P0 |
| MVVM with Compose | YES | Reuse ViewModel pattern | P0 |
| Dependency Injection | YES (Hilt) | Add new modules | P0 |
| **Code Quality** ||||
| Coroutines for async | YES | Maintain | P0 |
| Sealed classes for state | YES | Add PassportState | P0 |
| Data classes for models | YES | Create PassportData | P0 |
| Null safety | YES | Maintain | P0 |
| **Documentation** ||||
| APDU sequence comments | YES | Document passport APDUs | P0 |
| File ID documentation | YES | Add passport file IDs | P0 |
| Cryptographic explanations | YES | Extend for passport specifics | P1 |
| **Testing** ||||
| Unit tests for APDU | YES (partial) | Add passport APDU tests | P1 |
| Unit tests for DG parsing | YES | Add TD3 parser tests | P0 |
| Integration tests | NO | Implement mock passport | P2 |
| **Performance** ||||
| Dispatchers.IO for NFC | YES | Maintain | P0 |
| Proper cancellation | YES | Maintain | P0 |
| Memory management | PARTIAL | Improve bitmap handling | P1 |

### 1.3 Compliance Gaps Requiring Immediate Action

#### CRITICAL (Must fix before passport implementation)

1. **SecureByteArray Implementation**
   - Current: Byte arrays not explicitly zeroed after use
   - Required: Implement `SecureByteArray` wrapper that zeros memory on close
   - Files affected: `BacAuthentication.kt`, `SecureMessaging.kt`, `NfcCardReader.kt`

2. **Complete SOD Validation**
   - Current: 60% complete, missing CSCA certificate store
   - Required: Implement certificate chain validation
   - Approach: Bundle known CSCA certificates or allow user import

3. **Data Group Hash Verification**
   - Current: SOD signature verified but DG hashes not compared
   - Required: Extract hashes from SOD, compare with computed DG hashes
   - Files affected: `SodValidator.kt`

#### HIGH PRIORITY (Should fix during implementation)

4. **Bitmap Memory Management**
   - Current: Large bitmaps may not be recycled properly
   - Required: Implement proper lifecycle management for photo bitmaps
   - Add `bitmap.recycle()` calls when data is cleared

5. **Integration Test Infrastructure**
   - Current: 0% integration tests
   - Required: Mock IsoDep interface for passport flow testing

---

## Part 2: Technical Differences - Turkish eID vs Passport

### 2.1 Document Structure Comparison

| Aspect | Turkish eID (TD1) | Passport (TD3) |
|--------|-------------------|----------------|
| **Physical Format** | ID Card | Booklet |
| **MRZ Lines** | 3 lines x 30 chars | 2 lines x 44 chars |
| **Document Type** | I (ID Card) | P (Passport) |
| **Country Code Position** | Line 1, chars 3-5 | Line 1, chars 3-5 |
| **Document Number** | Line 1, chars 6-14 | Line 1, chars 1-9 (after type) |
| **DOB Position** | Line 2, chars 1-6 | Line 2, chars 14-19 |
| **DOE Position** | Line 2, chars 9-14 | Line 2, chars 22-27 |
| **Name Position** | Line 3 | Line 1, chars 6-44 |
| **Optional Data** | Line 1/2 optional fields | Line 2 optional field |

### 2.2 MRZ Format Details

#### Turkish eID (TD1 - 3x30)
```
Line 1: I<TUR[DocNum9][CD][Optional15]
Line 2: [DOB6][CD][Sex1][DOE6][CD][Nat3][Optional11][CD]
Line 3: [Surname<<GivenNames<<<<<<<<<<<<<<<]

Example:
I<TURA12345678<<<<<<<<<<<<<
9001151M3012319TUR<<<<<<<<0
YILMAZ<<MEHMET<<<<<<<<<<<<<
```

#### Passport (TD3 - 2x44)
```
Line 1: P<[Country3][Surname<<GivenNames<<<<<<<<<<<<<<<<<<<<<<<]
Line 2: [DocNum9][CD][Nat3][DOB6][CD][Sex1][DOE6][CD][Optional14][CD]

Example:
P<TURYILMAZ<<MEHMET<<<<<<<<<<<<<<<<<<<<<<<<
A1234567890TUR9001151M3012319<<<<<<<<<<<<<<0
```

### 2.3 Application Identifier (AID) Differences

| Document | AID | Description |
|----------|-----|-------------|
| Turkish eID | `A0 00 00 01 67 45 53 49 44` | TR-specific AID |
| ICAO Passport | `A0 00 00 02 47 10 01` | Standard eMRTD AID |

**Note**: Your current implementation already uses the ICAO standard AID (`A0 00 00 02 47 10 01`), which is correct for passports.

### 2.4 Data Group Differences

| Data Group | Turkish eID | Passport | Content |
|------------|-------------|----------|---------|
| DG1 | Required | Required | MRZ Data |
| DG2 | Required | Required | Facial Image |
| DG3 | Available (restricted) | Optional | Fingerprints |
| DG4 | N/A | Optional | Iris |
| DG5-DG16 | N/A | Various | Additional optional data |
| EF.COM | Available | Required | Data group presence list |
| EF.SOD | Available | Required | Security Object |
| EF.CardAccess | Available | Optional | PACE info |

### 2.5 Authentication Protocol Comparison

| Aspect | Turkish eID | Passport |
|--------|-------------|----------|
| **BAC** | Supported | Supported (mandatory for most) |
| **PACE** | Not implemented | Optional (newer passports) |
| **Key Derivation** | Same (SHA-1 from MRZ) | Same |
| **Session Keys** | 3DES | 3DES |
| **Secure Messaging** | ISO 7816-4 SM | ISO 7816-4 SM |

### 2.6 Image Format Differences

| Aspect | Turkish eID | Passport |
|--------|-------------|----------|
| **Format** | JPEG2000 (JP2) | JPEG2000 or JPEG |
| **Resolution** | Typically 300-400px | 240-640px (varies) |
| **Color** | Color | Color or Grayscale |
| **Size** | ~20-50KB | ~15-50KB |

---

## Part 3: Architecture Design

### 3.1 Proposed Module Structure

```
app/src/main/java/com/rollingcatsoftware/universalnfcreader/
├── data/
│   ├── nfc/
│   │   ├── readers/
│   │   │   ├── PassportNfcReader.kt          [NEW]
│   │   │   ├── TurkishEidNfcReader.kt        [EXISTING - refactor]
│   │   │   └── base/
│   │   │       └── BaseMrtdReader.kt         [NEW - shared logic]
│   │   ├── protocol/
│   │   │   ├── BacAuthentication.kt          [EXISTING - reuse]
│   │   │   ├── SecureMessaging.kt            [EXISTING - reuse]
│   │   │   └── ApduHelper.kt                 [EXISTING - extend]
│   │   ├── parser/
│   │   │   ├── MrzParser.kt                  [NEW - unified]
│   │   │   ├── Td1Parser.kt                  [REFACTOR from Dg1Parser]
│   │   │   ├── Td3Parser.kt                  [NEW]
│   │   │   ├── Dg2ImageParser.kt             [EXISTING - reuse]
│   │   │   └── SodValidator.kt               [EXISTING - extend]
│   │   └── security/
│   │       └── SecureByteArray.kt            [NEW - compliance]
│   ├── repository/
│   │   ├── MrtdRepository.kt                 [NEW - unified interface]
│   │   ├── PassportRepository.kt             [NEW]
│   │   └── TurkishEidRepository.kt           [EXISTING - refactor]
│   └── scanner/
│       └── MrzScanner.kt                     [EXISTING - extend for TD3]
├── domain/
│   ├── model/
│   │   ├── MrtdData.kt                       [NEW - base sealed class]
│   │   ├── PassportData.kt                   [NEW]
│   │   ├── TurkishEidData.kt                 [REFACTOR from EidData]
│   │   ├── MrzData.kt                        [NEW - unified MRZ]
│   │   └── DocumentType.kt                   [NEW - enum]
│   └── usecase/
│       ├── ReadMrtdUseCase.kt                [NEW - unified]
│       ├── ReadPassportUseCase.kt            [NEW]
│       ├── ValidateMrzUseCase.kt             [NEW - unified]
│       └── DetectDocumentTypeUseCase.kt      [NEW]
├── ui/
│   ├── screens/
│   │   ├── DocumentSelectionScreen.kt        [NEW]
│   │   ├── PassportReadScreen.kt             [NEW]
│   │   ├── TurkishEidReadScreen.kt           [EXISTING - refactor]
│   │   └── MrzScannerScreen.kt               [EXISTING - extend]
│   └── viewmodel/
│       ├── PassportViewModel.kt              [NEW]
│       └── MrtdViewModel.kt                  [NEW - base class]
└── di/
    ├── MrtdModule.kt                         [NEW]
    └── PassportModule.kt                     [NEW]
```

### 3.2 Class Hierarchy Design

```kotlin
// Base sealed class for all MRTD documents
sealed class MrtdData {
    abstract val personalInfo: PersonalInfo
    abstract val photo: Bitmap?
    abstract val documentType: DocumentType
    abstract val isAuthenticated: Boolean
}

// Passport-specific data
data class PassportData(
    override val personalInfo: PersonalInfo,
    override val photo: Bitmap?,
    override val documentType: DocumentType = DocumentType.PASSPORT,
    override val isAuthenticated: Boolean,
    val passportNumber: String,
    val issuingCountry: String,
    val nationality: String,
    val optionalData: String? = null
) : MrtdData()

// Turkish eID-specific data (refactored)
data class TurkishEidData(
    override val personalInfo: PersonalInfo,
    override val photo: Bitmap?,
    override val documentType: DocumentType = DocumentType.TURKISH_EID,
    override val isAuthenticated: Boolean,
    val tckn: String,
    val serialNumber: String
) : MrtdData()

// Shared personal information
data class PersonalInfo(
    val surname: String,
    val givenNames: String,
    val dateOfBirth: LocalDate,
    val gender: Gender,
    val dateOfExpiry: LocalDate,
    val documentNumber: String
)

enum class DocumentType {
    PASSPORT,
    TURKISH_EID,
    UNKNOWN
}

enum class Gender { MALE, FEMALE, UNSPECIFIED }
```

### 3.3 Reader Interface Design

```kotlin
// Base interface for all MRTD readers
interface MrtdReader<T : MrtdData> {
    suspend fun readDocument(
        tag: Tag,
        mrzData: MrzData
    ): NfcResult<T>

    fun getSupportedDocumentTypes(): List<DocumentType>
}

// Factory for creating appropriate reader
class MrtdReaderFactory @Inject constructor(
    private val passportReader: PassportNfcReader,
    private val turkishEidReader: TurkishEidNfcReader
) {
    fun getReader(documentType: DocumentType): MrtdReader<*> {
        return when (documentType) {
            DocumentType.PASSPORT -> passportReader
            DocumentType.TURKISH_EID -> turkishEidReader
            else -> throw IllegalArgumentException("Unsupported document type")
        }
    }
}
```

### 3.4 Secure Byte Array Implementation (Compliance)

```kotlin
/**
 * Secure byte array wrapper that zeros memory on close.
 * COMPLIANCE: se-checklist.md - PIN Handling - Clear PIN bytes from memory
 */
class SecureByteArray private constructor(
    private val data: ByteArray
) : Closeable {

    val size: Int get() = data.size

    operator fun get(index: Int): Byte = data[index]

    fun copyOf(): ByteArray = data.copyOf()

    fun toByteArray(): ByteArray = data.copyOf()

    override fun close() {
        // Zero out the memory
        SecureRandom().nextBytes(data)
        data.fill(0)
    }

    companion object {
        fun wrap(data: ByteArray): SecureByteArray = SecureByteArray(data)

        fun allocate(size: Int): SecureByteArray = SecureByteArray(ByteArray(size))
    }
}

// Usage with use() for automatic cleanup
fun example() {
    SecureByteArray.wrap(sensitiveData).use { secure ->
        // Use secure.data
    } // Automatically zeroed here
}
```

---

## Part 4: Implementation Phases

### Phase 0: Pre-Implementation Compliance Fixes (CRITICAL)
**Duration Estimate**: Foundation work
**Priority**: P0 - Must complete before passport implementation

#### 0.1 Implement SecureByteArray
- [ ] Create `SecureByteArray.kt` in `data/nfc/security/`
- [ ] Implement `Closeable` interface with memory zeroing
- [ ] Add factory methods: `wrap()`, `allocate()`
- [ ] Write unit tests for memory clearing behavior

#### 0.2 Refactor Existing Code for Secure Memory
- [ ] Update `BacAuthentication.kt`:
  - Wrap `kSeed`, `kEnc`, `kMac` in SecureByteArray
  - Use `use {}` blocks for automatic cleanup
- [ ] Update `SecureMessaging.kt`:
  - Wrap session keys in SecureByteArray
  - Clear SSC after session ends
- [ ] Update `NfcCardReader.kt`:
  - Clear MRZ-derived data after use

#### 0.3 Complete SOD Validation
- [ ] Implement `CscaCertificateStore.kt`:
  - Bundle known CSCA certificates (at minimum: Turkey)
  - Add passport-issuing country certificates
  - Implement certificate lookup by country code
- [ ] Enhance `SodValidator.kt`:
  - Add certificate chain validation
  - Implement DG hash extraction from SOD
  - Add hash comparison for DG1 and DG2
- [ ] Create `HashVerifier.kt`:
  - Compute SHA-256/SHA-1 of raw DG data
  - Compare with SOD-embedded hashes

#### 0.4 Add Logging Safeguards
- [ ] Create `SecureLogger.kt` wrapper:
  - Automatically redact patterns (TCKN, passport numbers)
  - Truncate potentially sensitive data
  - Different log levels for debug/release
- [ ] Update all Timber calls to use SecureLogger

### Phase 1: Core Passport Reading Infrastructure
**Priority**: P0

#### 1.1 MRZ Parser for TD3 Format
- [ ] Create `Td3Parser.kt`:
  ```kotlin
  class Td3Parser {
      fun parse(rawMrz: ByteArray): Td3MrzData
      fun parseLine1(line: String): Td3Line1Data  // Names
      fun parseLine2(line: String): Td3Line2Data  // Numbers, dates
      fun validateCheckDigits(data: Td3MrzData): Boolean
  }
  ```
- [ ] Implement check digit validation (ICAO weights: 7,3,1)
- [ ] Handle character substitutions for passport-specific OCR errors
- [ ] Write comprehensive unit tests with real passport MRZ samples

#### 1.2 Unified MRZ Data Model
- [ ] Create `MrzData.kt`:
  ```kotlin
  sealed class MrzData {
      abstract val documentNumber: String
      abstract val dateOfBirth: String  // YYMMDD
      abstract val dateOfExpiry: String // YYMMDD

      data class Td1(/* TD1 specific */) : MrzData()
      data class Td3(/* TD3 specific */) : MrzData()
  }
  ```
- [ ] Add factory methods for creating from parsed MRZ
- [ ] Implement `toBacInput()` method for key derivation

#### 1.3 Passport NFC Reader
- [ ] Create `PassportNfcReader.kt`:
  - Implement `MrtdReader<PassportData>` interface
  - Reuse `BacAuthentication` for key derivation
  - Reuse `SecureMessaging` for encrypted communication
  - Add passport-specific AID selection (same as current: `A0 00 00 02 47 10 01`)
- [ ] Implement `readEfCom()` to get data group list
- [ ] Implement `readDg1()` with TD3 parsing
- [ ] Implement `readDg2()` reusing existing image parser

#### 1.4 Passport Repository
- [ ] Create `PassportRepository.kt`:
  ```kotlin
  interface PassportRepository {
      suspend fun readPassport(
          tag: Tag,
          mrzData: MrzData.Td3
      ): Result<PassportData>
  }
  ```
- [ ] Implement error mapping for passport-specific errors
- [ ] Add input validation for TD3 MRZ format

### Phase 2: MRZ Scanner Enhancement
**Priority**: P0

#### 2.1 Document Type Detection
- [ ] Enhance `MrzScanner.kt`:
  - Detect TD1 vs TD3 format automatically
  - Return document type with scanned data
- [ ] Create `DocumentTypeDetector.kt`:
  - Analyze line count and length
  - Check document type indicator (I vs P)
  - Return confidence score

#### 2.2 TD3 OCR Optimization
- [ ] Add passport-specific OCR corrections:
  - Line 1: More letters, fewer digits
  - Line 2: More digits, specific positions
- [ ] Implement multi-line correlation:
  - Cross-validate data between lines
  - Use surname from line 1 in confidence scoring
- [ ] Add MRZ region detection:
  - Passport MRZ is at bottom of data page
  - Guide user to correct positioning

#### 2.3 Scanner UI Updates
- [ ] Update `MrzScannerScreen.kt`:
  - Add document type selector (auto-detect / passport / ID)
  - Show different overlay for TD3 (2 lines vs 3)
  - Display detected document type

### Phase 3: UI Implementation
**Priority**: P1

#### 3.1 Document Selection Screen
- [ ] Create `DocumentSelectionScreen.kt`:
  - Card-based selection: Passport / Turkish eID
  - Show supported features per document type
  - Navigate to appropriate read screen
- [ ] Add document type icons and descriptions

#### 3.2 Passport Read Screen
- [ ] Create `PassportReadScreen.kt`:
  - MRZ input for TD3 format
  - Camera scan integration
  - NFC reading flow
  - Result display
- [ ] Implement `PassportViewModel.kt`:
  - State management for passport reading
  - Error handling with passport-specific messages
  - Integration with `ReadPassportUseCase`

#### 3.3 Unified Result Display
- [ ] Create shared components for data display:
  - `PersonalInfoCard.kt` - Shared between passport/eID
  - `PhotoDisplay.kt` - With loading states
  - `DocumentInfoCard.kt` - Document-specific details
- [ ] Add copy-to-clipboard functionality
- [ ] Implement data clearing (memory cleanup on navigate away)

### Phase 4: Testing & Validation
**Priority**: P1

#### 4.1 Unit Tests
- [ ] TD3 Parser tests:
  - Valid passport MRZ samples (multiple countries)
  - Invalid MRZ handling
  - Check digit validation
  - Edge cases (special characters, fillers)
- [ ] PassportNfcReader tests:
  - Mock APDU responses
  - Error scenario handling
  - Secure messaging verification
- [ ] SecureByteArray tests:
  - Memory zeroing verification
  - Use block behavior

#### 4.2 Integration Tests
- [ ] Create mock passport interface:
  - Simulate full BAC handshake
  - Return test DG1/DG2 data
  - Simulate various error conditions
- [ ] End-to-end flow tests:
  - MRZ scan -> validation -> read -> display

#### 4.3 Manual Testing Checklist
- [ ] Test with real passport (if available):
  - [ ] MRZ scan accuracy
  - [ ] BAC authentication success
  - [ ] DG1 reading and parsing
  - [ ] DG2 image extraction
  - [ ] Error handling (wrong MRZ, connection lost)
- [ ] Test error scenarios:
  - [ ] Invalid MRZ data
  - [ ] NFC connection interruption
  - [ ] Timeout handling
  - [ ] Corrupted data handling

### Phase 5: Documentation & Compliance Verification
**Priority**: P1

#### 5.1 Code Documentation
- [ ] Document all new APDU sequences
- [ ] Add KDoc comments to public APIs
- [ ] Create architecture decision records (ADRs)
- [ ] Update README with passport support

#### 5.2 Security Documentation
- [ ] Document cryptographic operations
- [ ] Create security considerations guide
- [ ] Add data flow diagrams showing sensitive data handling

#### 5.3 Final Compliance Audit
- [ ] Complete se-checklist.md verification:
  - [ ] All security requirements checked
  - [ ] All engineering requirements checked
  - [ ] All items marked complete or documented as N/A
- [ ] Security review of new code
- [ ] Code quality review (KtLint, Detekt)

---

## Part 5: Detailed Technical Specifications

### 5.1 Passport APDU Command Sequences

#### Select MRTD Application
```
Command:  00 A4 04 0C 07 A0 00 00 02 47 10 01
Response: 90 00 (Success)
```

#### Read EF.COM (Data Group List)
```
Select:   00 A4 02 0C 02 01 1E
Read:     00 B0 00 00 00
Response: [ASN.1 TLV with DG list] 90 00
```

#### BAC Authentication
```
1. GET CHALLENGE:    00 84 00 00 08
   Response:         [RND.ICC 8 bytes] 90 00

2. EXTERNAL AUTH:    00 82 00 00 28 [E.IFD || M.IFD]
   Response:         [E.ICC || M.ICC] 90 00
```

#### Read DG1 (with Secure Messaging)
```
Select:   0C A4 02 0C [SM wrapped: 01 01]
Read:     0C B0 00 00 [SM wrapped: Le=00]
Response: [SM wrapped DG1 data] 90 00
```

### 5.2 TD3 MRZ Parsing Algorithm

```kotlin
fun parseTd3Mrz(line1: String, line2: String): PassportMrzData {
    require(line1.length == 44) { "Line 1 must be 44 characters" }
    require(line2.length == 44) { "Line 2 must be 44 characters" }

    // Line 1: P<COUNTRY_NAME<<<<<<<<<<<<<<<<<<<<<<<<<<
    val documentType = line1.substring(0, 1)      // "P"
    val issuingCountry = line1.substring(2, 5)    // Country code
    val names = line1.substring(5, 44)            // Surname<<GivenNames

    val nameParts = names.split("<<")
    val surname = nameParts[0].replace("<", " ").trim()
    val givenNames = nameParts.getOrElse(1) { "" }.replace("<", " ").trim()

    // Line 2: DOCNUM___CDNATDOBCDSDOECDOPTIONAL________CD
    val documentNumber = line2.substring(0, 9).replace("<", "")
    val docNumCheckDigit = line2[9]
    val nationality = line2.substring(10, 13)
    val dateOfBirth = line2.substring(13, 19)
    val dobCheckDigit = line2[19]
    val sex = line2[20]
    val dateOfExpiry = line2.substring(21, 27)
    val doeCheckDigit = line2[27]
    val optionalData = line2.substring(28, 42).replace("<", "")
    val optionalCheckDigit = line2[42]
    val compositeCheckDigit = line2[43]

    // Validate check digits
    validateCheckDigit(documentNumber, docNumCheckDigit)
    validateCheckDigit(dateOfBirth, dobCheckDigit)
    validateCheckDigit(dateOfExpiry, doeCheckDigit)

    // Composite check digit covers: docNum+CD + DOB+CD + DOE+CD + optional+CD
    val compositeData = line2.substring(0, 10) + line2.substring(13, 20) +
                        line2.substring(21, 43)
    validateCheckDigit(compositeData, compositeCheckDigit)

    return PassportMrzData(
        documentType = documentType,
        issuingCountry = issuingCountry,
        surname = surname,
        givenNames = givenNames,
        documentNumber = documentNumber,
        nationality = nationality,
        dateOfBirth = dateOfBirth,
        sex = sex,
        dateOfExpiry = dateOfExpiry,
        optionalData = optionalData.ifEmpty { null }
    )
}
```

### 5.3 BAC Key Derivation for Passports

The BAC key derivation is **identical** to Turkish eID:

```kotlin
fun deriveBacKeys(mrzData: MrzData.Td3): BacKeys {
    // MRZ_information = DocNum + CD + DOB + CD + DOE + CD
    val mrzInfo = buildString {
        append(mrzData.documentNumber.padEnd(9, '<'))
        append(calculateCheckDigit(mrzData.documentNumber))
        append(mrzData.dateOfBirth)
        append(calculateCheckDigit(mrzData.dateOfBirth))
        append(mrzData.dateOfExpiry)
        append(calculateCheckDigit(mrzData.dateOfExpiry))
    }

    // Kseed = SHA-1(MRZ_information)[0:16]
    val kSeed = sha1(mrzInfo.toByteArray(Charsets.UTF_8)).take(16)

    // Derive encryption and MAC keys
    val kEnc = deriveKey(kSeed, counter = 1)  // For encryption
    val kMac = deriveKey(kSeed, counter = 2)  // For MAC

    return BacKeys(kEnc, kMac)
}
```

### 5.4 Data Group Hash Verification

```kotlin
fun verifyDataGroupHashes(sod: ByteArray, dg1: ByteArray, dg2: ByteArray): Boolean {
    // 1. Parse SOD to extract LDSSecurityObject
    val sodParser = SodParser()
    val ldsSecurityObject = sodParser.parseLdsSecurityObject(sod)

    // 2. Get hash algorithm from SOD
    val hashAlgorithm = ldsSecurityObject.hashAlgorithm // Usually SHA-256

    // 3. Get expected hashes from SOD
    val expectedDg1Hash = ldsSecurityObject.dataGroupHashes[1]
    val expectedDg2Hash = ldsSecurityObject.dataGroupHashes[2]

    // 4. Compute actual hashes
    val actualDg1Hash = computeHash(dg1, hashAlgorithm)
    val actualDg2Hash = computeHash(dg2, hashAlgorithm)

    // 5. Compare hashes (constant-time comparison)
    return MessageDigest.isEqual(expectedDg1Hash, actualDg1Hash) &&
           MessageDigest.isEqual(expectedDg2Hash, actualDg2Hash)
}
```

---

## Part 6: Risk Assessment & Mitigation

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| BAC incompatibility | Low | High | BAC is standardized; test with multiple passports |
| Image format variation | Medium | Medium | Support multiple formats (JPEG, JPEG2000, JP2) |
| CSCA certificate availability | High | Medium | Bundle known certificates; allow user import |
| Memory issues with large photos | Medium | Low | Implement proper bitmap lifecycle management |
| OCR accuracy for TD3 | Medium | Medium | Multi-pass parsing; manual input fallback |

### 6.2 Security Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Sensitive data exposure | Low | Critical | SecureByteArray; memory zeroing; no logging |
| Man-in-middle attack | Very Low | High | BAC provides session encryption |
| Cloned/fake passport | Medium | High | SOD validation; CSCA chain verification |
| Side-channel attacks | Low | Medium | Constant-time comparisons; no timing leaks |

### 6.3 Compliance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| KVKK violation | Low | Critical | In-memory only; user consent; no persistence |
| Export control | Low | High | App for personal use only; terms of service |
| Unauthorized use | Medium | Medium | Disclaimer; require own document ownership |

---

## Part 7: Appendices

### Appendix A: ICAO Doc 9303 References

- **Part 3**: Specifications Common to all MRTDs (MRZ formats)
- **Part 4**: Specifications for MRPs (Passports)
- **Part 5**: Specifications for TD1/TD2 (ID Cards)
- **Part 10**: Logical Data Structure (LDS)
- **Part 11**: Security Mechanisms (BAC, PACE, AA, PA)

### Appendix B: File Identifiers

| File | FID | Description |
|------|-----|-------------|
| EF.COM | 011E | Data group presence list |
| EF.SOD | 011D | Security Object Document |
| EF.CardAccess | 011C | PACE info (if supported) |
| DG1 | 0101 | MRZ Data |
| DG2 | 0102 | Facial Image |
| DG3 | 0103 | Fingerprints (EAC required) |
| DG7 | 0107 | Displayed Signature |
| DG11 | 010B | Additional Personal Details |
| DG12 | 010C | Additional Document Details |
| DG14 | 010E | Security Options |
| DG15 | 010F | Active Authentication Public Key |

### Appendix C: Check Digit Algorithm

```kotlin
fun calculateCheckDigit(data: String): Int {
    val weights = intArrayOf(7, 3, 1)
    var sum = 0

    data.forEachIndexed { index, char ->
        val value = when {
            char.isDigit() -> char.digitToInt()
            char.isLetter() -> char.uppercaseChar() - 'A' + 10
            char == '<' -> 0
            else -> 0
        }
        sum += value * weights[index % 3]
    }

    return sum % 10
}
```

### Appendix D: Compliance Checklist Sign-off Template

```markdown
## Passport NFC Reader - Compliance Sign-off

### Security Requirements
- [ ] PIN/MRZ Handling: ___________ (Initials/Date)
- [ ] Data Protection: ___________ (Initials/Date)
- [ ] NFC Security: ___________ (Initials/Date)
- [ ] Cryptographic Operations: ___________ (Initials/Date)
- [ ] Code Security: ___________ (Initials/Date)

### Engineering Requirements
- [ ] Architecture: ___________ (Initials/Date)
- [ ] Code Quality: ___________ (Initials/Date)
- [ ] Documentation: ___________ (Initials/Date)
- [ ] Testing: ___________ (Initials/Date)
- [ ] Performance: ___________ (Initials/Date)

### Final Approval
- [ ] Security Review Complete: ___________ (Initials/Date)
- [ ] Code Review Complete: ___________ (Initials/Date)
- [ ] Ready for Release: ___________ (Initials/Date)
```

---

## Summary

This roadmap provides a comprehensive path from the current Turkish eID implementation to full passport support. Key priorities:

1. **Phase 0 (Compliance)**: Fix critical security gaps before adding features
2. **Phase 1 (Core)**: Implement passport reading with maximum code reuse
3. **Phase 2 (Scanner)**: Enhance MRZ scanning for TD3 format
4. **Phase 3 (UI)**: Create unified user experience
5. **Phase 4 (Testing)**: Ensure reliability and security
6. **Phase 5 (Documentation)**: Complete compliance verification

The architecture leverages your existing clean architecture and BAC implementation, minimizing new code while ensuring strict compliance with security requirements.

---

**Document Version**: 1.0
**Created**: 2025-12-14
**Author**: Claude Code
**Project**: UniversalNfcReader / TurkishEidNfcReader
**Standard**: ICAO Doc 9303
