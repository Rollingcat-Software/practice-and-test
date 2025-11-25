package com.turkey.eidnfc.domain.usecase

import android.nfc.Tag
import com.turkey.eidnfc.data.repository.EidRepository
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.PersonalData
import com.turkey.eidnfc.domain.model.Result
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for ReadEidCardUseCase.
 *
 * Tests the orchestration of PIN validation and card reading.
 */
class ReadEidCardUseCaseTest {

    private lateinit var repository: EidRepository
    private lateinit var validatePinUseCase: ValidatePinUseCase
    private lateinit var useCase: ReadEidCardUseCase
    private lateinit var mockTag: Tag

    @Before
    fun setup() {
        repository = mockk()
        validatePinUseCase = ValidatePinUseCase()
        useCase = ReadEidCardUseCase(repository, validatePinUseCase)
        mockTag = mockk(relaxed = true)
    }

    // ============================================================================
    // Success Path Tests
    // ============================================================================

    @Test
    fun `invoke with valid PIN and successful read returns Success`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val cardData = createValidCardData()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Success)
        assertEquals(cardData, (result as Result.Success).data)
        coVerify { repository.readCard(mockTag, validPin) }
    }

    // ============================================================================
    // PIN Validation Tests
    // ============================================================================

    @Test
    fun `invoke with invalid PIN returns Error without calling repository`() = runTest {
        // Given
        val invalidPin = "12345" // Too short
        val params = ReadEidCardUseCase.Params(mockTag, invalidPin)

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Error)
        assertEquals(
            "PIN must be exactly 6 digits",
            (result as Result.Error).exception.message
        )

        // Repository should not be called
        coVerify(exactly = 0) { repository.readCard(any(), any()) }
    }

    @Test
    fun `invoke with empty PIN returns Error`() = runTest {
        val params = ReadEidCardUseCase.Params(mockTag, "")
        val result = useCase(params)

        assertTrue(result is Result.Error)
        coVerify(exactly = 0) { repository.readCard(any(), any()) }
    }

    @Test
    fun `invoke with non-digit PIN returns Error`() = runTest {
        val params = ReadEidCardUseCase.Params(mockTag, "abc123")
        val result = useCase(params)

        assertTrue(result is Result.Error)
        coVerify(exactly = 0) { repository.readCard(any(), any()) }
    }

    // ============================================================================
    // Repository Error Tests
    // ============================================================================

    @Test
    fun `invoke returns Error when repository fails`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val errorMessage = "Card reading failed"

        coEvery { repository.readCard(mockTag, validPin) } returns
            Result.Error(Exception(errorMessage))

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Error)
        assertEquals(errorMessage, (result as Result.Error).exception.message)
    }

    @Test
    fun `invoke returns Loading when repository returns Loading`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Loading

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Loading)
    }

    // ============================================================================
    // Card Data Validation Tests
    // ============================================================================

    @Test
    fun `invoke returns Error when card has no personal data`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val cardDataWithoutPersonal = CardData(
            personalData = null,
            photo = null,
            cardAccessData = null,
            sodData = null
        )

        coEvery { repository.readCard(mockTag, validPin) } returns
            Result.Success(cardDataWithoutPersonal)

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Error)
        assertEquals(
            "No personal data found on card",
            (result as Result.Error).exception.message
        )
    }

    @Test
    fun `invoke returns Error when TCKN is empty`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val invalidCardData = createCardDataWithEmptyTckn()

        coEvery { repository.readCard(mockTag, validPin) } returns
            Result.Success(invalidCardData)

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Error)
        val errorMessage = (result as Result.Error).exception.message
        assertTrue(errorMessage?.contains("TCKN") == true)
    }

    @Test
    fun `invoke returns Error when name fields are empty`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val invalidCardData = createCardDataWithEmptyNames()

        coEvery { repository.readCard(mockTag, validPin) } returns
            Result.Success(invalidCardData)

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Error)
        val errorMessage = (result as Result.Error).exception.message
        assertTrue(errorMessage?.contains("name") == true)
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    @Test
    fun `invoke performs PIN validation before repository call`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val cardData = createValidCardData()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Success)
        coVerify { repository.readCard(mockTag, validPin) }
    }

    @Test
    fun `invoke with exception from repository returns Error`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val exception = RuntimeException("Unexpected error")

        coEvery { repository.readCard(mockTag, validPin) } throws exception

        // When
        val result = useCase(params)

        // Then
        assertTrue(result is Result.Error)
    }

    @Test
    fun `multiple invocations with same params produce consistent results`() = runTest {
        // Given
        val validPin = "123456"
        val params = ReadEidCardUseCase.Params(mockTag, validPin)
        val cardData = createValidCardData()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        val result1 = useCase(params)
        val result2 = useCase(params)

        // Then
        assertTrue(result1 is Result.Success)
        assertTrue(result2 is Result.Success)
        assertEquals((result1 as Result.Success).data, (result2 as Result.Success).data)
    }

    // ============================================================================
    // Helper Functions
    // ============================================================================

    private fun createValidCardData(): CardData {
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

        return CardData(
            personalData = personalData,
            photo = null,
            cardAccessData = null,
            sodData = null,
            isAuthenticated = true
        )
    }

    private fun createCardDataWithEmptyTckn(): CardData {
        val personalData = PersonalData(
            tckn = "", // Empty TCKN
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

        return CardData(
            personalData = personalData,
            photo = null,
            cardAccessData = null,
            sodData = null
        )
    }

    private fun createCardDataWithEmptyNames(): CardData {
        val personalData = PersonalData(
            tckn = "12345678901",
            firstName = "", // Empty
            lastName = "", // Empty
            birthDate = "1990-01-01",
            gender = "M",
            nationality = "TUR",
            documentNumber = "ABC123",
            expiryDate = "2030-01-01",
            issueDate = "2020-01-01",
            placeOfBirth = "İstanbul",
            serialNumber = "123456"
        )

        return CardData(
            personalData = personalData,
            photo = null,
            cardAccessData = null,
            sodData = null
        )
    }
}
