package com.turkey.eidnfc.ui

import android.nfc.Tag
import app.cash.turbine.test
import com.turkey.eidnfc.data.repository.EidRepository
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.PersonalData
import com.turkey.eidnfc.domain.model.Result
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for MainViewModel.
 *
 * Tests ViewModel logic with mocked Repository and Tag dependencies.
 * Uses Turbine for testing StateFlow emissions.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModelTest {

    private lateinit var repository: EidRepository
    private lateinit var viewModel: MainViewModel
    private lateinit var mockTag: Tag

    private val testDispatcher = StandardTestDispatcher()

    @Before
    fun setup() {
        Dispatchers.setMain(testDispatcher)
        repository = mockk()
        viewModel = MainViewModel(repository)
        mockTag = mockk(relaxed = true)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    // ============================================================================
    // Initial State Tests
    // ============================================================================

    @Test
    fun `initial UI state is Idle`() = runTest {
        viewModel.uiState.test {
            assertEquals(MainViewModel.UiState.Idle, awaitItem())
        }
    }

    @Test
    fun `initial PIN is empty`() = runTest {
        viewModel.pin.test {
            assertEquals("", awaitItem())
        }
    }

    // ============================================================================
    // PIN Change Tests
    // ============================================================================

    @Test
    fun `onPinChanged updates PIN for valid digits`() = runTest {
        viewModel.pin.test {
            assertEquals("", awaitItem()) // Initial value

            viewModel.onPinChanged("1")
            assertEquals("1", awaitItem())

            viewModel.onPinChanged("12")
            assertEquals("12", awaitItem())

            viewModel.onPinChanged("123456")
            assertEquals("123456", awaitItem())
        }
    }

    @Test
    fun `onPinChanged rejects non-digit characters`() = runTest {
        viewModel.pin.test {
            assertEquals("", awaitItem())

            viewModel.onPinChanged("123a56")
            expectNoEvents() // Should not emit anything

            viewModel.onPinChanged("12 456")
            expectNoEvents() // Should not emit anything
        }
    }

    @Test
    fun `onPinChanged enforces maximum length of 6`() = runTest {
        viewModel.pin.test {
            assertEquals("", awaitItem())

            viewModel.onPinChanged("123456")
            assertEquals("123456", awaitItem())

            // Try to add more digits
            viewModel.onPinChanged("1234567")
            expectNoEvents() // Should not emit beyond 6 digits
        }
    }

    @Test
    fun `onPinChanged allows all digits 0-9`() = runTest {
        viewModel.pin.test {
            skipItems(1) // Skip initial empty

            viewModel.onPinChanged("012345")
            assertEquals("012345", awaitItem())

            viewModel.onPinChanged("678901")
            assertEquals("678901", awaitItem())
        }
    }

    @Test
    fun `onPinChanged handles backspace by accepting shorter strings`() = runTest {
        viewModel.pin.test {
            skipItems(1) // Skip initial

            viewModel.onPinChanged("123")
            assertEquals("123", awaitItem())

            viewModel.onPinChanged("12")
            assertEquals("12", awaitItem())

            viewModel.onPinChanged("1")
            assertEquals("1", awaitItem())

            viewModel.onPinChanged("")
            assertEquals("", awaitItem())
        }
    }

    // ============================================================================
    // Tag Detection - Success Tests
    // ============================================================================

    @Test
    fun `onTagDetected with valid PIN and successful read updates state to Success`() = runTest {
        // Given
        val validPin = "123456"
        val cardData = createMockCardData()

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip Idle

            viewModel.onTagDetected(mockTag)
            assertEquals(MainViewModel.UiState.Reading, awaitItem())
            assertEquals(MainViewModel.UiState.Success(cardData), awaitItem())
        }
    }

    @Test
    fun `onTagDetected clears PIN after successful read`() = runTest {
        // Given
        val validPin = "123456"
        val cardData = createMockCardData()

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // Then
        viewModel.pin.test {
            assertEquals("", awaitItem()) // PIN should be cleared
        }
    }

    // ============================================================================
    // Tag Detection - Error Tests
    // ============================================================================

    @Test
    fun `onTagDetected with invalid PIN shows error without calling repository`() = runTest {
        // Given
        val invalidPin = "12345" // Too short

        viewModel.onPinChanged(invalidPin)
        advanceUntilIdle()

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip Idle

            viewModel.onTagDetected(mockTag)
            val state = awaitItem()
            assertTrue(state is MainViewModel.UiState.Error)
            assertEquals("Please enter a valid 6-digit PIN", (state as MainViewModel.UiState.Error).message)
        }

        // Repository should not be called
        coVerify(exactly = 0) { repository.readCard(any(), any()) }
    }

    @Test
    fun `onTagDetected with empty PIN shows error`() = runTest {
        // Given - PIN is empty by default

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip Idle

            viewModel.onTagDetected(mockTag)
            val state = awaitItem()
            assertTrue(state is MainViewModel.UiState.Error)
            assertEquals("Please enter a valid 6-digit PIN", (state as MainViewModel.UiState.Error).message)
        }
    }

    @Test
    fun `onTagDetected with repository error updates state to Error`() = runTest {
        // Given
        val validPin = "123456"
        val errorMessage = "Card reading failed"

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Error(Exception(errorMessage))

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip Idle

            viewModel.onTagDetected(mockTag)
            assertEquals(MainViewModel.UiState.Reading, awaitItem())

            val errorState = awaitItem()
            assertTrue(errorState is MainViewModel.UiState.Error)
            assertEquals(errorMessage, (errorState as MainViewModel.UiState.Error).message)
        }
    }

    @Test
    fun `onTagDetected handles exception with unknown error message`() = runTest {
        // Given
        val validPin = "123456"

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Error(Exception())

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip Idle

            viewModel.onTagDetected(mockTag)
            assertEquals(MainViewModel.UiState.Reading, awaitItem())

            val errorState = awaitItem()
            assertTrue(errorState is MainViewModel.UiState.Error)
            assertEquals("Unknown error occurred", (errorState as MainViewModel.UiState.Error).message)
        }
    }

    // ============================================================================
    // Reset State Tests
    // ============================================================================

    @Test
    fun `resetState sets UI state to Idle`() = runTest {
        // Given - set state to Error first
        val invalidPin = "12345"
        viewModel.onPinChanged(invalidPin)
        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip current state

            viewModel.resetState()
            assertEquals(MainViewModel.UiState.Idle, awaitItem())
        }
    }

    @Test
    fun `resetState clears PIN`() = runTest {
        // Given - set PIN first
        viewModel.onPinChanged("123456")
        advanceUntilIdle()

        // When
        viewModel.resetState()
        advanceUntilIdle()

        // Then
        viewModel.pin.test {
            assertEquals("", awaitItem())
        }
    }

    @Test
    fun `resetState from Success state returns to Idle`() = runTest {
        // Given - successful card read
        val validPin = "123456"
        val cardData = createMockCardData()

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // When
        viewModel.uiState.test {
            skipItems(1) // Skip Success state

            viewModel.resetState()
            assertEquals(MainViewModel.UiState.Idle, awaitItem())
        }
    }

    // ============================================================================
    // State Transition Tests
    // ============================================================================

    @Test
    fun `complete flow from Idle to Reading to Success`() = runTest {
        // Given
        val validPin = "123456"
        val cardData = createMockCardData()

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When/Then
        viewModel.uiState.test {
            assertEquals(MainViewModel.UiState.Idle, awaitItem())

            viewModel.onTagDetected(mockTag)
            assertEquals(MainViewModel.UiState.Reading, awaitItem())
            assertEquals(MainViewModel.UiState.Success(cardData), awaitItem())
        }
    }

    @Test
    fun `complete flow from Idle to Reading to Error`() = runTest {
        // Given
        val validPin = "123456"
        val errorMessage = "Connection lost"

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Error(Exception(errorMessage))

        // When/Then
        viewModel.uiState.test {
            assertEquals(MainViewModel.UiState.Idle, awaitItem())

            viewModel.onTagDetected(mockTag)
            assertEquals(MainViewModel.UiState.Reading, awaitItem())

            val errorState = awaitItem()
            assertTrue(errorState is MainViewModel.UiState.Error)
            assertEquals(errorMessage, (errorState as MainViewModel.UiState.Error).message)
        }
    }

    @Test
    fun `multiple tag detection attempts work correctly`() = runTest {
        // Given
        val validPin = "123456"
        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        // First attempt - success
        val cardData = createMockCardData()
        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // Reset
        viewModel.resetState()
        advanceUntilIdle()

        // Second attempt - error
        viewModel.onPinChanged(validPin)
        advanceUntilIdle()
        coEvery { repository.readCard(mockTag, validPin) } returns Result.Error(Exception("Error"))

        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // Verify repository was called twice
        coVerify(exactly = 2) { repository.readCard(mockTag, validPin) }
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    @Test
    fun `onTagDetected with PIN containing only zeros works`() = runTest {
        // Given
        val validPin = "000000"
        val cardData = createMockCardData()

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // Then
        coVerify { repository.readCard(mockTag, validPin) }
    }

    @Test
    fun `onPinChanged with exactly 6 digits is accepted`() = runTest {
        viewModel.pin.test {
            skipItems(1)

            viewModel.onPinChanged("999999")
            assertEquals("999999", awaitItem())
        }
    }

    @Test
    fun `repository is called with correct parameters`() = runTest {
        // Given
        val validPin = "123456"
        val cardData = createMockCardData()

        viewModel.onPinChanged(validPin)
        advanceUntilIdle()

        coEvery { repository.readCard(mockTag, validPin) } returns Result.Success(cardData)

        // When
        viewModel.onTagDetected(mockTag)
        advanceUntilIdle()

        // Then
        coVerify(exactly = 1) { repository.readCard(mockTag, validPin) }
    }

    // ============================================================================
    // Helper Functions
    // ============================================================================

    private fun createMockCardData(): CardData {
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
}
