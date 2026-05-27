# Practical Improvements - Turkish eID NFC Reader
## Code Quality, Architecture & Best Practices

**Purpose**: Improve code quality, architecture, and adherence to best practices for a high-quality reference implementation.

**Note**: This is NOT for enterprise deployment, but for learning, demonstration, and maintaining high code standards.

---

## Table of Contents

1. [Immediate Quick Wins](#1-immediate-quick-wins)
2. [Architecture Improvements](#2-architecture-improvements)
3. [Code Quality Enhancements](#3-code-quality-enhancements)
4. [Testing Implementation](#4-testing-implementation)
5. [Performance Optimizations](#5-performance-optimizations)
6. [UX Improvements](#6-ux-improvements)
7. [Security Best Practices](#7-security-best-practices)
8. [Implementation Priority](#8-implementation-priority)

---

## 1. Immediate Quick Wins

### 1.1 Add Dependency Injection (Hilt) - 2-3 hours

**Why**: Improves testability, reduces coupling, follows SOLID principles

**Current Problem**: Hard-coded dependencies make testing difficult

```kotlin
// Current (MainActivity.kt)
class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()  // Hard-coded
}

// Current (MainViewModel.kt)
class MainViewModel : ViewModel() {
    private val cardReader = NfcCardReader()  // Hard-coded dependency
}
```

**Improved with Hilt**:

```kotlin
// 1. Add dependencies to app/build.gradle
dependencies {
    implementation("com.google.dagger:hilt-android:2.48")
    kapt("com.google.dagger:hilt-compiler:2.48")
}

// 2. Create Application class
@HiltAndroidApp
class EidReaderApplication : Application()

// 3. Update AndroidManifest.xml
<application
    android:name=".EidReaderApplication"
    ...>

// 4. Create modules
@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideNfcCardReader(): NfcCardReader {
        return NfcCardReader()
    }

    @Provides
    fun provideDg1Parser(): Dg1Parser = Dg1Parser

    @Provides
    fun provideDg2Parser(): Dg2Parser = Dg2Parser

    @Provides
    fun provideSodValidator(): SodValidator = SodValidator
}

// 5. Update ViewModel
@HiltViewModel
class MainViewModel @Inject constructor(
    private val cardReader: NfcCardReader
) : ViewModel() {
    // Now testable!
}

// 6. Update MainActivity
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    // Hilt will inject automatically
}
```

**Benefits**:
- ✅ Easy to mock for testing
- ✅ Follows Dependency Inversion Principle
- ✅ Single source of truth for dependencies
- ✅ Better code organization

---

### 1.2 Add Result Wrapper for Better Error Handling - 1 hour

**Why**: Makes error handling explicit and type-safe

**Current Problem**: Using sealed classes inconsistently

```kotlin
// Current approach is actually good, but let's make it more consistent

// Create a common Result wrapper
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val exception: Exception) : Result<Nothing>()
    data object Loading : Result<Nothing>()
}

// Extension functions for better usability
inline fun <T> Result<T>.onSuccess(action: (T) -> Unit): Result<T> {
    if (this is Result.Success) action(data)
    return this
}

inline fun <T> Result<T>.onError(action: (Exception) -> Unit): Result<T> {
    if (this is Result.Error) action(exception)
    return this
}

// Usage in ViewModel
viewModelScope.launch {
    cardReader.readCard(tag, pin)
        .onSuccess { cardData ->
            _uiState.value = UiState.Success(cardData)
        }
        .onError { exception ->
            _uiState.value = UiState.Error(exception.message ?: "Unknown error")
        }
}
```

---

### 1.3 Extract String Resources - 30 minutes

**Why**: Internationalization readiness, better maintenance

**Current Problem**: Hardcoded strings in composables

```xml
<!-- Add to app/src/main/res/values/strings.xml -->
<resources>
    <!-- App -->
    <string name="app_name">Turkish eID NFC Reader</string>

    <!-- Idle Screen -->
    <string name="title_eid_reader">Turkish eID Card Reader</string>
    <string name="instruction_enter_pin">Enter your 6-digit PIN and hold your ID card near the device</string>
    <string name="label_pin">PIN (6 digits)</string>
    <string name="placeholder_pin">Enter your PIN</string>
    <string name="info_enable_nfc">Make sure NFC is enabled in your device settings</string>
    <string name="pin_length_format">%1$d/6</string>

    <!-- Reading Screen -->
    <string name="status_reading">Reading card…</string>
    <string name="instruction_keep_steady">Keep your card near the device</string>

    <!-- Success Screen -->
    <string name="title_success">Card Read Successfully</string>
    <string name="label_photo">Photo</string>
    <string name="label_personal_info">Personal Information</string>
    <string name="action_read_another">Read Another Card</string>

    <!-- Personal Data Labels -->
    <string name="label_name">Name</string>
    <string name="label_tckn">TCKN</string>
    <string name="label_birth_date">Birth Date</string>
    <string name="label_gender">Gender</string>
    <string name="label_nationality">Nationality</string>
    <string name="label_document_number">Document Number</string>
    <string name="label_issue_date">Issue Date</string>
    <string name="label_expiry_date">Expiry Date</string>
    <string name="label_place_of_birth">Place of Birth</string>

    <!-- Error Screen -->
    <string name="title_error">Error Reading Card</string>
    <string name="action_try_again">Try Again</string>

    <!-- Error Messages -->
    <string name="error_wrong_pin">Wrong PIN. %1$d attempt(s) remaining.</string>
    <string name="error_card_locked">Card is locked. Contact authorities to unlock.</string>
    <string name="error_security_not_satisfied">Security condition not satisfied. Please verify PIN.</string>
    <string name="error_file_not_found">Required file not found on card.</string>
    <string name="error_invalid_card">This is not a valid Turkish eID card.</string>
    <string name="error_nfc_not_available">NFC is not available on this device.</string>
    <string name="error_nfc_disabled">Please enable NFC in device settings.</string>
    <string name="error_connection_lost">Connection to card lost. Please try again.</string>
    <string name="error_timeout">Operation timed out. Please try again.</string>
    <string name="error_parse">Failed to parse card data.</string>
</resources>
```

```kotlin
// Update MainScreen.kt to use string resources
@Composable
fun IdleScreen(pin: String, onPinChanged: (String) -> Unit) {
    Column(/* ... */) {
        Text(
            text = stringResource(R.string.title_eid_reader),
            style = MaterialTheme.typography.headlineMedium
        )

        Text(
            text = stringResource(R.string.instruction_enter_pin),
            style = MaterialTheme.typography.bodyLarge
        )

        OutlinedTextField(
            label = { Text(stringResource(R.string.label_pin)) },
            placeholder = { Text(stringResource(R.string.placeholder_pin)) },
            supportingText = {
                Text(stringResource(R.string.pin_length_format, pin.length))
            }
        )
    }
}

// Update EidData.kt for error messages
fun NfcError.toUserMessage(context: Context): String {
    return when (this) {
        is NfcError.WrongPin ->
            context.getString(R.string.error_wrong_pin, attemptsRemaining)
        NfcError.CardLocked ->
            context.getString(R.string.error_card_locked)
        // ... etc
    }
}
```

---

### 1.4 Add Timber Initialization - 15 minutes

**Why**: Proper logging setup, prevents crashes in release

**Current Problem**: Timber is used but not properly initialized

```kotlin
// Create TimberInitializer.kt
object TimberInitializer {

    fun initialize(isDebug: Boolean) {
        if (isDebug) {
            Timber.plant(Timber.DebugTree())
        } else {
            Timber.plant(ReleaseTree())
        }
    }

    private class ReleaseTree : Timber.Tree() {
        override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
            if (priority == Log.ERROR || priority == Log.WARN) {
                // Log to crash reporting service (Firebase Crashlytics)
                // For now, just use Android Log
                if (t != null) {
                    Log.e(tag, message, t)
                } else {
                    Log.e(tag, message)
                }
            }
        }
    }
}

// Update EidReaderApplication.kt
@HiltAndroidApp
class EidReaderApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        TimberInitializer.initialize(BuildConfig.DEBUG)
    }
}
```

---

## 2. Architecture Improvements

### 2.1 Add Repository Layer - 2-3 hours

**Why**: Separates data sources from business logic, easier to test

**Create data/repository package**:

```kotlin
// EidRepository.kt
interface EidRepository {
    suspend fun readCard(tag: Tag, pin: String): Result<CardData>
}

class EidRepositoryImpl @Inject constructor(
    private val nfcCardReader: NfcCardReader
) : EidRepository {

    override suspend fun readCard(tag: Tag, pin: String): Result<CardData> {
        return try {
            when (val result = nfcCardReader.readCard(tag, pin)) {
                is NfcResult.Success -> Result.Success(result.data)
                is NfcResult.Error -> Result.Error(
                    Exception(result.error.toUserMessage())
                )
                NfcResult.Loading -> Result.Loading
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to read card")
            Result.Error(e)
        }
    }
}

// Update Hilt module
@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindEidRepository(
        impl: EidRepositoryImpl
    ): EidRepository
}

// Update ViewModel to use repository
@HiltViewModel
class MainViewModel @Inject constructor(
    private val repository: EidRepository
) : ViewModel() {

    fun onTagDetected(tag: Tag) {
        val currentPin = _pin.value

        if (currentPin.length != 6) {
            _uiState.value = UiState.Error("Please enter a 6-digit PIN")
            return
        }

        viewModelScope.launch {
            _uiState.value = UiState.Reading

            repository.readCard(tag, currentPin)
                .onSuccess { cardData ->
                    _uiState.value = UiState.Success(cardData)
                    clearPin()
                }
                .onError { exception ->
                    _uiState.value = UiState.Error(exception.message ?: "Unknown error")
                }
        }
    }
}
```

**Benefits**:
- ✅ Clear separation of concerns
- ✅ Easy to add caching later
- ✅ Easy to mock for testing
- ✅ Single source of truth for data operations

---

### 2.2 Add Use Case Layer (Optional but Recommended) - 2 hours

**Why**: Encapsulates business logic, reusable across ViewModels

```kotlin
// Create domain/usecase package

// ReadCardUseCase.kt
class ReadCardUseCase @Inject constructor(
    private val repository: EidRepository
) {
    suspend operator fun invoke(tag: Tag, pin: String): Result<CardData> {
        // Pre-validation
        if (!isPinValid(pin)) {
            return Result.Error(IllegalArgumentException("Invalid PIN format"))
        }

        // Business logic
        return repository.readCard(tag, pin)
    }

    private fun isPinValid(pin: String): Boolean {
        return pin.length == 6 && pin.all { it.isDigit() }
    }
}

// ValidateCardDataUseCase.kt
class ValidateCardDataUseCase @Inject constructor() {

    operator fun invoke(cardData: CardData): ValidationResult {
        val errors = mutableListOf<String>()

        // Validate TCKN
        cardData.personalData?.let { data ->
            if (!isValidTckn(data.tckn)) {
                errors.add("Invalid TCKN")
            }

            if (data.firstName.isBlank() || data.lastName.isBlank()) {
                errors.add("Name fields cannot be empty")
            }
        }

        return ValidationResult(
            isValid = errors.isEmpty(),
            errors = errors
        )
    }

    private fun isValidTckn(tckn: String): Boolean {
        if (tckn.length != 11) return false
        if (!tckn.all { it.isDigit() }) return false

        // TCKN validation algorithm
        val digits = tckn.map { it.toString().toInt() }

        if (digits[0] == 0) return false

        val sum1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8]) * 7
        val sum2 = digits[1] + digits[3] + digits[5] + digits[7]
        val check10 = (sum1 - sum2) % 10

        if (check10 != digits[9]) return false

        val sumAll = digits.take(10).sum()
        val check11 = sumAll % 10

        return check11 == digits[10]
    }
}

data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>
)

// Update ViewModel
@HiltViewModel
class MainViewModel @Inject constructor(
    private val readCardUseCase: ReadCardUseCase,
    private val validateCardDataUseCase: ValidateCardDataUseCase
) : ViewModel() {

    fun onTagDetected(tag: Tag) {
        viewModelScope.launch {
            _uiState.value = UiState.Reading

            readCardUseCase(tag, _pin.value)
                .onSuccess { cardData ->
                    // Validate the data
                    val validation = validateCardDataUseCase(cardData)

                    if (validation.isValid) {
                        _uiState.value = UiState.Success(cardData)
                    } else {
                        _uiState.value = UiState.Error(
                            "Data validation failed: ${validation.errors.joinToString()}"
                        )
                    }
                    clearPin()
                }
                .onError { exception ->
                    _uiState.value = UiState.Error(exception.message ?: "Unknown error")
                }
        }
    }
}
```

---

### 2.3 Improve State Management - 1 hour

**Why**: More robust state handling, prevent invalid states

```kotlin
// Create better sealed state classes

sealed class CardReadState {
    data object Idle : CardReadState()
    data object Reading : CardReadState()
    data class Success(val cardData: CardData) : CardReadState()

    sealed class Error : CardReadState() {
        abstract val message: String

        data class PinError(
            override val message: String,
            val attemptsRemaining: Int
        ) : Error()

        data class CardError(override val message: String) : Error()
        data class ConnectionError(override val message: String) : Error()
        data class ValidationError(override val message: String) : Error()
        data class UnknownError(override val message: String) : Error()
    }
}

// Update ViewModel with better state management
@HiltViewModel
class MainViewModel @Inject constructor(
    private val readCardUseCase: ReadCardUseCase
) : ViewModel() {

    private val _state = MutableStateFlow<CardReadState>(CardReadState.Idle)
    val state: StateFlow<CardReadState> = _state.asStateFlow()

    private val _pin = MutableStateFlow("")
    val pin: StateFlow<String> = _pin.asStateFlow()

    val isPinValid: StateFlow<Boolean> = _pin
        .map { it.length == 6 && it.all { c -> c.isDigit() } }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    fun onPinChanged(newPin: String) {
        if (newPin.length <= 6 && newPin.all { it.isDigit() }) {
            _pin.value = newPin
        }
    }

    fun onTagDetected(tag: Tag) {
        if (!isPinValid.value) {
            _state.value = CardReadState.Error.ValidationError(
                "Please enter a valid 6-digit PIN"
            )
            return
        }

        viewModelScope.launch {
            _state.value = CardReadState.Reading

            readCardUseCase(tag, _pin.value)
                .onSuccess { cardData ->
                    _state.value = CardReadState.Success(cardData)
                    clearPin()
                }
                .onError { exception ->
                    _state.value = mapExceptionToErrorState(exception)
                }
        }
    }

    private fun mapExceptionToErrorState(exception: Exception): CardReadState.Error {
        return when {
            exception.message?.contains("Wrong PIN") == true -> {
                // Extract attempts remaining
                val attempts = extractAttempts(exception.message)
                CardReadState.Error.PinError(
                    exception.message ?: "Wrong PIN",
                    attempts
                )
            }
            exception.message?.contains("locked") == true ->
                CardReadState.Error.CardError(exception.message ?: "Card locked")
            exception.message?.contains("Connection") == true ->
                CardReadState.Error.ConnectionError(exception.message ?: "Connection lost")
            else ->
                CardReadState.Error.UnknownError(exception.message ?: "Unknown error")
        }
    }

    fun resetState() {
        _state.value = CardReadState.Idle
        clearPin()
    }

    private fun clearPin() {
        _pin.value = ""
    }
}
```

---

## 3. Code Quality Enhancements

### 3.1 Add KtLint for Code Style - 30 minutes

```kotlin
// Add to build.gradle (project level)
plugins {
    id("org.jlleitschuh.gradle.ktlint") version "11.6.1"
}

// Add to build.gradle (app level)
plugins {
    id("org.jlleitschuh.gradle.ktlint")
}

ktlint {
    android.set(true)
    ignoreFailures.set(false)
    reporters {
        reporter(org.jlleitschuh.gradle.ktlint.reporter.ReporterType.PLAIN)
        reporter(org.jlleitschuh.gradle.ktlint.reporter.ReporterType.CHECKSTYLE)
    }
}

// Run lint check
// ./gradlew ktlintCheck

// Auto-format code
// ./gradlew ktlintFormat
```

---

### 3.2 Add Detekt for Static Analysis - 30 minutes

```kotlin
// Add to build.gradle (project level)
plugins {
    id("io.gitlab.arturbosch.detekt") version "1.23.3"
}

// Add to build.gradle (app level)
plugins {
    id("io.gitlab.arturbosch.detekt")
}

detekt {
    config = files("$projectDir/config/detekt/detekt.yml")
    buildUponDefaultConfig = true
    allRules = false
}

// Create config/detekt/detekt.yml
build:
  maxIssues: 0

complexity:
  LongMethod:
    threshold: 60
  LongParameterList:
    threshold: 6
  ComplexMethod:
    threshold: 15

style:
  MagicNumber:
    ignoreNumbers: ['-1', '0', '1', '2', '6', '8', '11', '12', '16']

// Run detekt
// ./gradlew detekt
```

---

### 3.3 Improve Constants Management - 30 minutes

**Current Problem**: Magic numbers and strings scattered in code

```kotlin
// Create Constants.kt
object NfcConstants {
    // Timeouts
    const val CONNECTION_TIMEOUT_MS = 5000
    const val OPERATION_TIMEOUT_MS = 30000

    // PIN
    const val PIN_LENGTH = 6
    const val MIN_PIN_ATTEMPTS_WARNING = 3

    // APDU
    const val APDU_SUCCESS = 0x9000
    const val APDU_WRONG_PIN_MASK = 0x63C0
    const val APDU_SECURITY_NOT_SATISFIED = 0x6982
    const val APDU_AUTH_BLOCKED = 0x6983

    // File sizes
    const val MAX_DG1_SIZE = 10240  // 10KB
    const val MAX_DG2_SIZE = 102400 // 100KB

    // Image
    const val JPEG2000_MAGIC_BYTES_SIZE = 8
    const val PHOTO_MAX_DIMENSION = 800
}

object UiConstants {
    const val DEBOUNCE_DELAY_MS = 300L
    const val SNACKBAR_DURATION_MS = 3000L
    const val ANIMATION_DURATION_MS = 300
}

// Usage
class NfcCardReader {
    companion object {
        private const val TIMEOUT_MS = NfcConstants.OPERATION_TIMEOUT_MS
    }
}
```

---

### 3.4 Add Extension Functions for Cleaner Code - 1 hour

```kotlin
// Create Extensions.kt

// ByteArray extensions
fun ByteArray.toHexString(): String {
    return joinToString(" ") { String.format("%02X", it) }
}

fun ByteArray.toInt(): Int {
    require(size <= 4) { "ByteArray too large to convert to Int" }
    var result = 0
    for (byte in this) {
        result = (result shl 8) or (byte.toInt() and 0xFF)
    }
    return result
}

// String extensions
fun String.isValidPin(): Boolean {
    return length == NfcConstants.PIN_LENGTH && all { it.isDigit() }
}

fun String.isValidTckn(): Boolean {
    if (length != 11 || !all { it.isDigit() }) return false

    val digits = map { it.toString().toInt() }
    if (digits[0] == 0) return false

    val sum1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8]) * 7
    val sum2 = digits[1] + digits[3] + digits[5] + digits[7]
    val check10 = (sum1 - sum2) % 10

    if (check10 != digits[9]) return false

    val sumAll = digits.take(10).sum()
    val check11 = sumAll % 10

    return check11 == digits[10]
}

// Flow extensions
fun <T> Flow<T>.throttleFirst(periodMillis: Long): Flow<T> {
    return flow {
        var lastEmissionTime = 0L
        collect { value ->
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastEmissionTime >= periodMillis) {
                lastEmissionTime = currentTime
                emit(value)
            }
        }
    }
}

// Context extensions
fun Context.isNfcAvailable(): Boolean {
    return NfcAdapter.getDefaultAdapter(this) != null
}

fun Context.isNfcEnabled(): Boolean {
    return NfcAdapter.getDefaultAdapter(this)?.isEnabled == true
}

// Composable extensions
@Composable
fun Modifier.shimmerEffect(): Modifier = composed {
    var size by remember { mutableStateOf(IntSize.Zero) }
    val transition = rememberInfiniteTransition()
    val startOffsetX by transition.animateFloat(
        initialValue = -2 * size.width.toFloat(),
        targetValue = 2 * size.width.toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(1000)
        )
    )

    background(
        brush = Brush.linearGradient(
            colors = listOf(
                Color(0xFFB8B5B5),
                Color(0xFF8F8B8B),
                Color(0xFFB8B5B5),
            ),
            start = Offset(startOffsetX, 0f),
            end = Offset(startOffsetX + size.width.toFloat(), size.height.toFloat())
        )
    )
        .onGloballyPositioned {
            size = it.size
        }
}
```

---

## 4. Testing Implementation

### 4.1 Add Unit Tests for Core Logic - 4-6 hours

**Goal**: 60-70% coverage on business logic

```kotlin
// Add to build.gradle
dependencies {
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.mockito.kotlin:mockito-kotlin:5.1.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    testImplementation("app.cash.turbine:turbine:1.0.0")
    testImplementation("com.google.truth:truth:1.1.5")
}

// Test ApduHelper
class ApduHelperTest {

    @Test
    fun `selectEidAid returns correct APDU command`() {
        val command = ApduHelper.selectEidAid()

        Truth.assertThat(command).hasLength(16)
        Truth.assertThat(command[0]).isEqualTo(0x00)
        Truth.assertThat(command[1]).isEqualTo(0xA4.toByte())
        Truth.assertThat(command[2]).isEqualTo(0x04)
    }

    @Test
    fun `verifyPinCommand with valid PIN returns command`() {
        val pin = "123456"
        val command = ApduHelper.verifyPinCommand(pin)

        Truth.assertThat(command).isNotNull()
        Truth.assertThat(command).hasLength(11)
        Truth.assertThat(command!![4]).isEqualTo(0x06)
    }

    @Test
    fun `verifyPinCommand with invalid PIN returns null`() {
        val invalidPins = listOf("12345", "1234567", "abc123", "")

        invalidPins.forEach { pin ->
            val command = ApduHelper.verifyPinCommand(pin)
            Truth.assertThat(command).isNull()
        }
    }

    @Test
    fun `isValidPin validates correctly`() {
        Truth.assertThat(ApduHelper.isValidPin("123456")).isTrue()
        Truth.assertThat(ApduHelper.isValidPin("000000")).isTrue()

        Truth.assertThat(ApduHelper.isValidPin("12345")).isFalse()
        Truth.assertThat(ApduHelper.isValidPin("1234567")).isFalse()
        Truth.assertThat(ApduHelper.isValidPin("abc123")).isFalse()
    }

    @Test
    fun `parseResponse extracts status word correctly`() {
        val response = byteArrayOf(0x01, 0x02, 0x03, 0x90.toByte(), 0x00)
        val (data, statusWord) = ApduHelper.parseResponse(response)

        Truth.assertThat(statusWord).isEqualTo(0x9000)
        Truth.assertThat(data).hasLength(3)
    }

    @Test
    fun `getRemainingPinAttempts parses 63CX correctly`() {
        Truth.assertThat(ApduHelper.getRemainingPinAttempts(0x63C2)).isEqualTo(2)
        Truth.assertThat(ApduHelper.getRemainingPinAttempts(0x63C1)).isEqualTo(1)
        Truth.assertThat(ApduHelper.getRemainingPinAttempts(0x63C0)).isEqualTo(0)
        Truth.assertThat(ApduHelper.getRemainingPinAttempts(0x9000)).isEqualTo(-1)
    }
}

// Test TCKN validation
class ExtensionsTest {

    @Test
    fun `isValidTckn validates correct TCKNs`() {
        val validTckns = listOf(
            "12345678901",  // Would need real valid TCKNs for proper testing
        )

        validTckns.forEach { tckn ->
            // Test with actual validation logic
        }
    }

    @Test
    fun `isValidTckn rejects invalid TCKNs`() {
        val invalidTckns = listOf(
            "1234567890",   // Too short
            "123456789012", // Too long
            "0123456789",   // Starts with 0
            "abcdefghijk",  // Contains letters
        )

        invalidTckns.forEach { tckn ->
            Truth.assertThat(tckn.isValidTckn()).isFalse()
        }
    }

    @Test
    fun `isValidPin validates correctly`() {
        Truth.assertThat("123456".isValidPin()).isTrue()
        Truth.assertThat("000000".isValidPin()).isTrue()

        Truth.assertThat("12345".isValidPin()).isFalse()
        Truth.assertThat("abc123".isValidPin()).isFalse()
    }
}

// Test ViewModel
@ExperimentalCoroutinesTest
class MainViewModelTest {

    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()

    private lateinit var viewModel: MainViewModel
    private val mockRepository: EidRepository = mock()

    @Before
    fun setup() {
        viewModel = MainViewModel(mockRepository)
    }

    @Test
    fun `onPinChanged updates pin value`() = runTest {
        viewModel.onPinChanged("123456")

        Truth.assertThat(viewModel.pin.value).isEqualTo("123456")
    }

    @Test
    fun `onPinChanged rejects non-digits`() = runTest {
        viewModel.onPinChanged("abc123")

        Truth.assertThat(viewModel.pin.value).isEmpty()
    }

    @Test
    fun `onPinChanged limits to 6 digits`() = runTest {
        viewModel.onPinChanged("1234567")

        Truth.assertThat(viewModel.pin.value).hasLength(6)
    }

    @Test
    fun `resetState clears pin and returns to idle`() = runTest {
        viewModel.onPinChanged("123456")
        viewModel.resetState()

        Truth.assertThat(viewModel.pin.value).isEmpty()
        Truth.assertThat(viewModel.state.value).isEqualTo(CardReadState.Idle)
    }
}
```

---

### 4.2 Add UI Tests (Basic) - 2 hours

```kotlin
// Add to build.gradle
androidTestImplementation("androidx.compose.ui:ui-test-junit4")
debugImplementation("androidx.compose.ui:ui-test-manifest")

// Create MainScreenTest.kt
class MainScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun pinInputField_isDisplayed() {
        composeTestRule.setContent {
            MainScreen(
                uiState = MainViewModel.UiState.Idle,
                pin = "",
                onPinChanged = {},
                onResetClick = {}
            )
        }

        composeTestRule
            .onNodeWithText("PIN (6 digits)")
            .assertIsDisplayed()
    }

    @Test
    fun pinInput_acceptsDigits() {
        var pin = ""

        composeTestRule.setContent {
            IdleScreen(
                pin = pin,
                onPinChanged = { pin = it }
            )
        }

        composeTestRule
            .onNodeWithText("PIN (6 digits)")
            .performTextInput("123456")

        Truth.assertThat(pin).isEqualTo("123456")
    }
}
```

---

## 5. Performance Optimizations

### 5.1 Add Remember and Derivedstateof - 1 hour

```kotlin
// Optimize recompositions in MainScreen.kt

@Composable
fun MainScreen(
    uiState: MainViewModel.UiState,
    pin: String,
    onPinChanged: (String) -> Unit,
    onResetClick: () -> Unit
) {
    // Avoid unnecessary recompositions
    val isPinValid = remember(pin) {
        pin.length == 6 && pin.all { it.isDigit() }
    }

    Scaffold(/* ... */) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            when (uiState) {
                MainViewModel.UiState.Idle -> IdleScreen(pin, onPinChanged, isPinValid)
                // ... other states
            }
        }
    }
}

@Composable
fun IdleScreen(
    pin: String,
    onPinChanged: (String) -> Unit,
    isPinValid: Boolean = remember(pin) { pin.length == 6 }
) {
    // Memoize expensive operations
    val pinMask = remember(pin) {
        "•".repeat(pin.length)
    }

    val pinColor by remember(isPinValid) {
        derivedStateOf {
            if (isPinValid) Color.Green else MaterialTheme.colorScheme.primary
        }
    }

    // ... UI code
}
```

---

### 5.2 Optimize Bitmap Handling - 1 hour

```kotlin
// Add BitmapUtils.kt
object BitmapUtils {

    fun optimizeBitmap(
        original: Bitmap,
        maxWidth: Int = 800,
        maxHeight: Int = 800
    ): Bitmap {
        val width = original.width
        val height = original.height

        if (width <= maxWidth && height <= maxHeight) {
            return original
        }

        val ratio = min(
            maxWidth.toFloat() / width,
            maxHeight.toFloat() / height
        )

        val newWidth = (width * ratio).toInt()
        val newHeight = (height * ratio).toInt()

        val scaled = Bitmap.createScaledBitmap(original, newWidth, newHeight, true)

        if (scaled != original) {
            original.recycle()
        }

        return scaled
    }

    fun recycleBitmap(bitmap: Bitmap?) {
        bitmap?.let {
            if (!it.isRecycled) {
                it.recycle()
            }
        }
    }
}

// Update Dg2Parser to use optimization
fun parse(dg2Data: ByteArray): Bitmap? {
    return try {
        val bitmap = decodeJpeg2000(imageData)
        bitmap?.let { BitmapUtils.optimizeBitmap(it) }
    } catch (e: Exception) {
        Timber.e(e, "Failed to parse DG2")
        null
    }
}
```

---

### 5.3 Add Lazy Loading for Photo - 30 minutes

```kotlin
// Update PhotoCard composable
@Composable
fun PhotoCard(photo: Bitmap) {
    var isLoading by remember { mutableStateOf(true) }
    val imageBitmap = remember(photo) {
        photo.asImageBitmap()
    }

    LaunchedEffect(photo) {
        isLoading = false
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = stringResource(R.string.label_photo),
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.align(Alignment.Start)
            )
            Spacer(modifier = Modifier.height(8.dp))

            Box(
                modifier = Modifier.size(200.dp),
                contentAlignment = Alignment.Center
            ) {
                if (isLoading) {
                    CircularProgressIndicator()
                } else {
                    Image(
                        bitmap = imageBitmap,
                        contentDescription = stringResource(R.string.label_photo),
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(8.dp))
                    )
                }
            }
        }
    }
}
```

---

## 6. UX Improvements

### 6.1 Add Loading Skeleton - 1 hour

```kotlin
// Add shimmer effect for loading state
@Composable
fun LoadingPersonalDataCard() {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Box(
                modifier = Modifier
                    .height(24.dp)
                    .fillMaxWidth(0.5f)
                    .shimmerEffect()
            )
            Spacer(modifier = Modifier.height(16.dp))

            repeat(9) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(20.dp)
                            .shimmerEffect()
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Box(
                        modifier = Modifier
                            .weight(1.5f)
                            .height(20.dp)
                            .shimmerEffect()
                    )
                }
            }
        }
    }
}
```

---

### 6.2 Add Animations - 1 hour

```kotlin
// Add smooth transitions between states
@Composable
fun MainScreen(/* ... */) {
    Scaffold(/* ... */) { paddingValues ->
        AnimatedContent(
            targetState = uiState,
            transitionSpec = {
                fadeIn(animationSpec = tween(300)) with
                    fadeOut(animationSpec = tween(300))
            }
        ) { state ->
            when (state) {
                MainViewModel.UiState.Idle -> IdleScreen(pin, onPinChanged)
                MainViewModel.UiState.Reading -> ReadingScreen()
                is MainViewModel.UiState.Success -> SuccessScreen(state.cardData, onResetClick)
                is MainViewModel.UiState.Error -> ErrorScreen(state.message, onResetClick)
            }
        }
    }
}

// Add scale animation for success icon
@Composable
fun SuccessScreen(/* ... */) {
    val scale by remember { Animatable(0f) }

    LaunchedEffect(Unit) {
        scale.animateTo(
            targetValue = 1f,
            animationSpec = spring(
                dampingRatio = Spring.DampingRatioMediumBouncy,
                stiffness = Spring.StiffnessLow
            )
        )
    }

    Icon(
        imageVector = Icons.Default.CheckCircle,
        contentDescription = "Success",
        modifier = Modifier
            .size(32.dp)
            .scale(scale.value)
    )
}
```

---

### 6.3 Add Haptic Feedback - 30 minutes

```kotlin
// Add haptic feedback for better UX
@Composable
fun MainScreen(/* ... */) {
    val view = LocalView.current

    LaunchedEffect(uiState) {
        when (uiState) {
            is MainViewModel.UiState.Success -> {
                view.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
            }
            is MainViewModel.UiState.Error -> {
                view.performHapticFeedback(HapticFeedbackConstants.REJECT)
            }
            else -> {}
        }
    }

    // ... rest of UI
}
```

---

### 6.4 Improve Error Messages - 30 minutes

```kotlin
// Make error messages more actionable
fun NfcError.toDetailedMessage(context: Context): String {
    return when (this) {
        is NfcError.WrongPin -> buildString {
            append(context.getString(R.string.error_wrong_pin, attemptsRemaining))
            append("\n\n")
            if (attemptsRemaining <= 1) {
                append("⚠️ Warning: Your card will be locked after the next failed attempt.")
            } else {
                append("💡 Tip: Make sure you're entering your 6-digit PIN1, not PIN2.")
            }
        }

        NfcError.NfcDisabled -> buildString {
            append(context.getString(R.string.error_nfc_disabled))
            append("\n\n")
            append("📱 To enable NFC:\n")
            append("1. Open Settings\n")
            append("2. Go to Connected devices\n")
            append("3. Toggle NFC ON")
        }

        NfcError.ConnectionLost -> buildString {
            append(context.getString(R.string.error_connection_lost))
            append("\n\n")
            append("💡 Tips:\n")
            append("• Hold the card steady against your phone\n")
            append("• Try different positions\n")
            append("• Remove thick phone cases")
        }

        else -> toUserMessage(context)
    }
}
```

---

## 7. Security Best Practices

### 7.1 Add Secure PIN Memory Clearing - 30 minutes

```kotlin
// Create SecurePinManager.kt
class SecurePinManager {

    private var pinChars: CharArray? = null

    fun setPin(pin: String) {
        clearPin()
        pinChars = pin.toCharArray()
    }

    fun getPin(): String? {
        return pinChars?.let { String(it) }
    }

    fun clearPin() {
        pinChars?.let { chars ->
            // Overwrite memory
            for (i in chars.indices) {
                chars[i] = '0'
            }
        }
        pinChars = null
    }

    fun getPinBytes(): ByteArray? {
        return pinChars?.let { chars ->
            ByteArray(chars.size) { i ->
                chars[i].code.toByte()
            }
        }
    }

    override fun finalize() {
        clearPin()
    }
}

// Update ViewModel to use SecurePinManager
@HiltViewModel
class MainViewModel @Inject constructor(
    private val repository: EidRepository
) : ViewModel() {

    private val securePinManager = SecurePinManager()

    private val _pinDisplay = MutableStateFlow("")
    val pinDisplay: StateFlow<String> = _pinDisplay.asStateFlow()

    fun onPinChanged(newPin: String) {
        if (newPin.length <= 6 && newPin.all { it.isDigit() }) {
            securePinManager.setPin(newPin)
            _pinDisplay.value = newPin
        }
    }

    fun onTagDetected(tag: Tag) {
        viewModelScope.launch {
            val pin = securePinManager.getPin()
            if (pin == null || pin.length != 6) {
                _state.value = CardReadState.Error.ValidationError("Invalid PIN")
                return@launch
            }

            // Use pin...

            // Clear immediately after use
            securePinManager.clearPin()
        }
    }

    override fun onCleared() {
        super.onCleared()
        securePinManager.clearPin()
    }
}
```

---

### 7.2 Add ProGuard Optimization - 30 minutes

```kotlin
// Update proguard-rules.pro with better rules

# Optimization
-optimizationpasses 5
-dontusemixedcaseclassnames
-dontskipnonpubliclibraryclasses
-verbose

# Keep line numbers for debugging
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Keep models
-keep class com.turkey.eidnfc.domain.model.** { *; }

# Keep Parcelable
-keepclassmembers class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

# Bouncy Castle
-keep class org.bouncycastle.** { *; }
-dontwarn org.bouncycastle.**

# JAI ImageIO
-keep class com.github.jaiimageio.** { *; }
-dontwarn com.github.jaiimageio.**
-dontwarn javax.imageio.**

# Hilt
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keep class * extends dagger.hilt.android.lifecycle.HiltViewModel

# Compose
-keep class androidx.compose.** { *; }

# Remove logging in release
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}

-assumenosideeffects class timber.log.Timber {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}
```

---

## 8. Implementation Priority

### 🔴 High Priority (Do First - Week 1)

1. **Add Hilt DI** (2-3 hours)
   - Immediate testability improvement
   - Foundation for other improvements

2. **Extract String Resources** (30 min)
   - Easy win, big impact

3. **Add Timber Initialization** (15 min)
   - Prevents crashes

4. **Improve Constants** (30 min)
   - Code clarity

5. **Add Extension Functions** (1 hour)
   - Cleaner code

### 🟡 Medium Priority (Week 2)

6. **Add Repository Layer** (2-3 hours)
   - Better architecture

7. **Improve State Management** (1 hour)
   - More robust

8. **Add KtLint** (30 min)
   - Code consistency

9. **Add Unit Tests** (4-6 hours)
   - Core functionality coverage

10. **Optimize Bitmap Handling** (1 hour)
    - Performance

### 🟢 Low Priority (Week 3+)

11. **Add Use Cases** (2 hours)
    - Optional but recommended

12. **Add Detekt** (30 min)
    - Static analysis

13. **Add Animations** (1 hour)
    - Polish

14. **Add UI Tests** (2 hours)
    - Coverage

15. **Add Loading Skeleton** (1 hour)
    - UX polish

---

## Summary

### Total Estimated Time: ~30-40 hours

**Week 1 (High Priority)**: 5-6 hours
- Hilt, strings, timber, constants, extensions

**Week 2 (Medium Priority)**: 10-15 hours
- Repository, state, tests, ktlint, performance

**Week 3 (Low Priority)**: 7-10 hours
- Use cases, detekt, animations, UI tests

**Ongoing**: Code review, refactoring, polish

---

## Key Improvements Achieved

✅ **Architecture**: Repository pattern, DI, better separation
✅ **Quality**: KtLint, Detekt, unit tests (60-70% coverage)
✅ **Performance**: Bitmap optimization, lazy loading, remember
✅ **UX**: Animations, loading states, better errors
✅ **Security**: Secure PIN handling, ProGuard
✅ **Maintainability**: Constants, extensions, string resources

---

## What This Gives You

- 🎓 **Learning**: Best practices, modern Android development
- 💼 **Portfolio**: High-quality reference project
- 🏗️ **Foundation**: Solid base for future enhancements
- 📚 **Knowledge**: Testing, architecture patterns, optimization
- ✨ **Polish**: Production-quality code (without the production overhead)

---

**This is practical, achievable, and focused on code quality rather than deployment infrastructure.**
