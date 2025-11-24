package com.turkey.eidnfc.domain.usecase

import com.turkey.eidnfc.domain.model.Result
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for ValidatePinUseCase.
 *
 * Tests PIN validation business logic.
 */
class ValidatePinUseCaseTest {

    private lateinit var useCase: ValidatePinUseCase

    @Before
    fun setup() {
        useCase = ValidatePinUseCase()
    }

    // ============================================================================
    // Valid PIN Tests
    // ============================================================================

    @Test
    fun `invoke with valid 6-digit PIN returns Success`() = runTest {
        // Given
        val validPin = "123456"

        // When
        val result = useCase(validPin)

        // Then
        assertTrue(result is Result.Success)
        assertEquals(true, (result as Result.Success).data)
    }

    @Test
    fun `invoke with all zeros returns Success`() = runTest {
        // Given - even though it's weak, it's still valid format
        val pin = "000000"

        // When
        val result = useCase(pin)

        // Then
        assertTrue(result is Result.Success)
    }

    @Test
    fun `invoke with all nines returns Success`() = runTest {
        val result = useCase("999999")
        assertTrue(result is Result.Success)
    }

    // ============================================================================
    // Invalid PIN Tests
    // ============================================================================

    @Test
    fun `invoke with too short PIN returns Error`() = runTest {
        // Given
        val shortPin = "12345"

        // When
        val result = useCase(shortPin)

        // Then
        assertTrue(result is Result.Error)
        assertEquals(
            "PIN must be exactly 6 digits",
            (result as Result.Error).exception.message
        )
    }

    @Test
    fun `invoke with too long PIN returns Error`() = runTest {
        val longPin = "1234567"
        val result = useCase(longPin)

        assertTrue(result is Result.Error)
    }

    @Test
    fun `invoke with non-digit characters returns Error`() = runTest {
        val invalidPin = "12345a"
        val result = useCase(invalidPin)

        assertTrue(result is Result.Error)
    }

    @Test
    fun `invoke with empty PIN returns Error`() = runTest {
        val result = useCase("")

        assertTrue(result is Result.Error)
    }

    @Test
    fun `invoke with spaces returns Error`() = runTest {
        val result = useCase("12 456")

        assertTrue(result is Result.Error)
    }

    @Test
    fun `invoke with special characters returns Error`() = runTest {
        val result = useCase("123-56")

        assertTrue(result is Result.Error)
    }

    // ============================================================================
    // Quick Validation Tests
    // ============================================================================

    @Test
    fun `validateQuick returns true for valid PIN`() {
        assertTrue(ValidatePinUseCase.validateQuick("123456"))
        assertTrue(ValidatePinUseCase.validateQuick("000000"))
        assertTrue(ValidatePinUseCase.validateQuick("999999"))
    }

    @Test
    fun `validateQuick returns false for invalid PIN`() {
        assertFalse(ValidatePinUseCase.validateQuick("12345"))
        assertFalse(ValidatePinUseCase.validateQuick("1234567"))
        assertFalse(ValidatePinUseCase.validateQuick("abc123"))
        assertFalse(ValidatePinUseCase.validateQuick(""))
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    @Test
    fun `invoke handles common PINs correctly`() = runTest {
        // Common PINs should still be valid (we just log a warning)
        val commonPins = listOf(
            "123456", "000000", "111111", "222222",
            "333333", "444444", "555555", "666666",
            "777777", "888888", "999999", "654321"
        )

        commonPins.forEach { pin ->
            val result = useCase(pin)
            assertTrue("$pin should be valid", result is Result.Success)
        }
    }

    @Test
    fun `invoke with unicode digits returns Error`() = runTest {
        // Unicode digits (e.g., Arabic numerals) should not be accepted
        val unicodePin = "١٢٣٤٥٦" // Arabic numerals 1-6
        val result = useCase(unicodePin)

        assertTrue(result is Result.Error)
    }

    @Test
    fun `validateQuick is consistent with full validation`() = runTest {
        val testPins = listOf(
            "123456" to true,
            "000000" to true,
            "12345" to false,
            "1234567" to false,
            "abc123" to false,
            "" to false
        )

        testPins.forEach { (pin, expectedValid) ->
            val quickResult = ValidatePinUseCase.validateQuick(pin)
            val fullResult = useCase(pin)

            assertEquals("Quick validation mismatch for: $pin", expectedValid, quickResult)
            assertEquals(
                "Full validation mismatch for: $pin",
                expectedValid,
                fullResult is Result.Success
            )
        }
    }
}
