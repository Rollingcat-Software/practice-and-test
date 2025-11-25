package com.turkey.eidnfc.data.repository

import android.nfc.Tag
import com.turkey.eidnfc.data.nfc.NfcCardReader
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.NfcError
import com.turkey.eidnfc.domain.model.NfcResult
import com.turkey.eidnfc.domain.model.PersonalData
import com.turkey.eidnfc.domain.model.Result
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

// Type aliases for clearer any() usage
private typealias AnyTag = Tag
private typealias AnyString = String

/**
 * Unit tests for EidRepositoryImpl.
 *
 * Tests the repository layer with mocked NfcCardReader dependency.
 */
class EidRepositoryImplTest {

    private lateinit var cardReader: NfcCardReader
    private lateinit var repository: EidRepositoryImpl
    private lateinit var mockTag: Tag

    @Before
    fun setup() {
        cardReader = mockk()
        repository = EidRepositoryImpl(cardReader)
        mockTag = mockk(relaxed = true)
    }

    // ============================================================================
    // PIN Validation Tests
    // ============================================================================

    @Test
    fun `validatePin returns true for valid PIN`() {
        assertTrue(repository.validatePin("123456"))
        assertTrue(repository.validatePin("000000"))
        assertTrue(repository.validatePin("999999"))
    }

    @Test
    fun `validatePin returns false for invalid PIN`() {
        assertFalse(repository.validatePin("12345"))   // Too short
        assertFalse(repository.validatePin("1234567")) // Too long
        assertFalse(repository.validatePin("12345a"))  // Contains letter
        assertFalse(repository.validatePin(""))        // Empty
    }

    // ============================================================================
    // Read Card Success Tests
    // ============================================================================

    @Test
    fun `readCard returns Success when card reader succeeds`() = runTest {
        // Given
        val validPin = "123456"
        val personalData = PersonalData(
            tckn = "12345678901",
            firstName = "Ahmet",
            lastName = "Yılmaz",
            birthDate = "1990-01-01",
            gender = "M",
            nationality = "TUR",
            documentNumber = "ABC123",
            expiryDate = "2030-01-01",
            issueDate = "2020-01-01",
            placeOfBirth = "İstanbul",
            serialNumber = "123456"
        )
        val cardData = CardData(
            personalData = personalData,
            photo = null,
            cardAccessData = null,
            sodData = null,
            isAuthenticated = true
        )
        val nfcSuccess = NfcResult.Success(cardData)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcSuccess

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Success)
        assertEquals(cardData, (result as Result.Success).data)
        coVerify { cardReader.readCard(mockTag, validPin) }
    }

    // ============================================================================
    // Read Card Error Tests
    // ============================================================================

    @Test
    fun `readCard returns Error when PIN is invalid`() = runTest {
        // Given
        val invalidPin = "12345" // Too short

        // When
        val result = repository.readCard(mockTag, invalidPin)

        // Then
        assertTrue(result is Result.Error)
        val exception = (result as Result.Error).exception
        assertTrue(exception is IllegalArgumentException)
        assertEquals("PIN must be exactly 6 digits", exception.message)

        // Card reader should not be called
        coVerify(exactly = 0) { cardReader.readCard(any<AnyTag>(), any<AnyString>()) }
    }

    @Test
    fun `readCard returns Error for wrong PIN from card reader`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.WrongPin(attemptsRemaining = 2))

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        val exception = (result as Result.Error).exception
        assertEquals("Wrong PIN. 2 attempt(s) remaining.", exception.message)
    }

    @Test
    fun `readCard returns Error for card locked`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.CardLocked)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Card is locked. Contact authorities to unlock.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for security not satisfied`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.SecurityNotSatisfied)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Security condition not satisfied. Please verify PIN.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for file not found`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.FileNotFound)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Required file not found on card.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for invalid card`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.InvalidCard)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("This is not a valid Turkish eID card.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for NFC not available`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.NfcNotAvailable)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("NFC is not available on this device.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for NFC disabled`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.NfcDisabled)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Please enable NFC in device settings.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for connection lost`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.ConnectionLost)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Connection to card lost. Please try again.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for timeout`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.Timeout)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Operation timed out. Please try again.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for parse error`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.ParseError)

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Failed to parse card data.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard returns Error for unknown error`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.UnknownError("Custom error message"))

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Error: Custom error message", (result as Result.Error).exception.message)
    }

    // ============================================================================
    // Loading State Tests
    // ============================================================================

    @Test
    fun `readCard returns Loading when card reader returns Loading`() = runTest {
        // Given
        val validPin = "123456"
        coEvery { cardReader.readCard(mockTag, validPin) } returns NfcResult.Loading

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Loading)
    }

    // ============================================================================
    // Exception Handling Tests
    // ============================================================================

    @Test
    fun `readCard catches and wraps exceptions from card reader`() = runTest {
        // Given
        val validPin = "123456"
        val exception = RuntimeException("Unexpected error")

        coEvery { cardReader.readCard(mockTag, validPin) } throws exception

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals(exception, (result as Result.Error).exception)
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    @Test
    fun `readCard with valid PIN calls card reader exactly once`() = runTest {
        // Given
        val validPin = "123456"
        val cardData = CardData(
            personalData = null,
            photo = null,
            cardAccessData = null,
            sodData = null
        )
        coEvery { cardReader.readCard(mockTag, validPin) } returns NfcResult.Success(cardData)

        // When
        repository.readCard(mockTag, validPin)

        // Then
        coVerify(exactly = 1) { cardReader.readCard(mockTag, validPin) }
    }

    @Test
    fun `readCard with multiple invalid PINs never calls card reader`() = runTest {
        // Test multiple invalid PINs
        val invalidPins = listOf("12345", "1234567", "abcdef", "")

        for (invalidPin in invalidPins) {
            // When
            val result = repository.readCard(mockTag, invalidPin)

            // Then
            assertTrue(result is Result.Error)
        }

        // Card reader should never be called
        coVerify(exactly = 0) { cardReader.readCard(any<AnyTag>(), any<AnyString>()) }
    }

    @Test
    fun `readCard handles zero remaining PIN attempts`() = runTest {
        // Given
        val validPin = "123456"
        val nfcError = NfcResult.Error(NfcError.WrongPin(attemptsRemaining = 0))

        coEvery { cardReader.readCard(mockTag, validPin) } returns nfcError

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals("Wrong PIN. 0 attempt(s) remaining.", (result as Result.Error).exception.message)
    }

    @Test
    fun `readCard handles success with minimal card data`() = runTest {
        // Given
        val validPin = "123456"
        val minimalCardData = CardData(
            personalData = null,
            photo = null,
            cardAccessData = null,
            sodData = null,
            isAuthenticated = false
        )
        coEvery { cardReader.readCard(mockTag, validPin) } returns NfcResult.Success(minimalCardData)

        // When
        val result = repository.readCard(mockTag, validPin)

        // Then
        assertTrue(result is Result.Success)
        assertEquals(minimalCardData, (result as Result.Success).data)
        assertNull(result.data.personalData)
        assertNull(result.data.photo)
        assertFalse(result.data.isAuthenticated)
    }
}
