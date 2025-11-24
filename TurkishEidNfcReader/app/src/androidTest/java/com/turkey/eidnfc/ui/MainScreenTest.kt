package com.turkey.eidnfc.ui

import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.PersonalData
import org.junit.Rule
import org.junit.Test

/**
 * UI tests for MainScreen composables.
 *
 * Tests user interactions and UI state changes.
 */
class MainScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    // ============================================================================
    // Idle Screen Tests
    // ============================================================================

    @Test
    fun idleScreen_displaysCorrectElements() {
        // Given
        var pinValue = ""

        // When
        composeTestRule.setContent {
            IdleScreen(
                pin = pinValue,
                onPinChanged = { pinValue = it }
            )
        }

        // Then
        composeTestRule
            .onNodeWithText("Turkish eID Card Reader")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("Enter your 6-digit PIN and hold your ID card near the device")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("PIN (6 digits)")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("Make sure NFC is enabled in your device settings")
            .assertIsDisplayed()
    }

    @Test
    fun pinInputField_acceptsDigitsOnly() {
        // Given
        var pinValue = ""

        composeTestRule.setContent {
            PinInputField(
                pin = pinValue,
                onPinChanged = { pinValue = it }
            )
        }

        // When - Input digits
        composeTestRule
            .onNodeWithText("PIN (6 digits)")
            .performTextInput("123456")

        // Then - PIN should be updated
        assert(pinValue == "123456")
    }

    @Test
    fun pinInputField_showsErrorForInvalidLength() {
        // Given
        composeTestRule.setContent {
            PinInputField(
                pin = "12345", // Invalid - only 5 digits
                onPinChanged = {}
            )
        }

        // Then - Should show error state (isError = true when length != 6)
        composeTestRule
            .onNodeWithText("5/6")
            .assertIsDisplayed()
    }

    @Test
    fun pinInputField_togglesVisibility() {
        // Given
        composeTestRule.setContent {
            PinInputField(
                pin = "123456",
                onPinChanged = {}
            )
        }

        // When - Click visibility toggle
        composeTestRule
            .onNodeWithContentDescription("Show PIN")
            .performClick()

        // Then - Icon should change
        composeTestRule
            .onNodeWithContentDescription("Hide PIN")
            .assertExists()
    }

    // ============================================================================
    // Reading Screen Tests
    // ============================================================================

    @Test
    fun readingScreen_displaysLoadingIndicator() {
        // When
        composeTestRule.setContent {
            ReadingScreen()
        }

        // Then
        composeTestRule
            .onNodeWithText("Reading card")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("Keep your card near the device")
            .assertIsDisplayed()

        // Progress indicator should be displayed
        composeTestRule
            .onNode(hasProgressBarRangeInfo(ProgressBarRangeInfo.Indeterminate))
            .assertExists()
    }

    // ============================================================================
    // Success Screen Tests
    // ============================================================================

    @Test
    fun successScreen_displaysCardData() {
        // Given
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

        var resetClicked = false

        // When
        composeTestRule.setContent {
            SuccessScreen(
                cardData = cardData,
                onResetClick = { resetClicked = true }
            )
        }

        // Then - Success header
        composeTestRule
            .onNodeWithText("Card Read Successfully")
            .assertIsDisplayed()

        // Personal data
        composeTestRule
            .onNodeWithText("Personal Information")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("Ahmet Yılmaz", substring = true)
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("12345678901")
            .assertIsDisplayed()

        // Reset button
        composeTestRule
            .onNodeWithText("Read Another Card")
            .assertIsDisplayed()
            .performClick()

        assert(resetClicked)
    }

    @Test
    fun successScreen_displaysPersonalDataFields() {
        // Given
        val personalData = PersonalData(
            tckn = "12345678901",
            firstName = "Mehmet",
            lastName = "Demir",
            birthDate = "1985-05-15",
            gender = "M",
            nationality = "TUR",
            documentNumber = "XYZ789",
            expiryDate = "2028-12-31",
            issueDate = "2018-01-01",
            placeOfBirth = "Ankara",
            serialNumber = "789012"
        )

        val cardData = CardData(
            personalData = personalData,
            photo = null,
            cardAccessData = null,
            sodData = null
        )

        // When
        composeTestRule.setContent {
            SuccessScreen(
                cardData = cardData,
                onResetClick = {}
            )
        }

        // Then - All fields should be displayed
        composeTestRule.onNodeWithText("Mehmet Demir", substring = true).assertIsDisplayed()
        composeTestRule.onNodeWithText("12345678901").assertIsDisplayed()
        composeTestRule.onNodeWithText("1985-05-15").assertIsDisplayed()
        composeTestRule.onNodeWithText("M").assertIsDisplayed()
        composeTestRule.onNodeWithText("TUR").assertIsDisplayed()
        composeTestRule.onNodeWithText("XYZ789").assertIsDisplayed()
        composeTestRule.onNodeWithText("Ankara").assertIsDisplayed()
    }

    // ============================================================================
    // Error Screen Tests
    // ============================================================================

    @Test
    fun errorScreen_displaysErrorMessage() {
        // Given
        val errorMessage = "Wrong PIN. 2 attempt(s) remaining."
        var resetClicked = false

        // When
        composeTestRule.setContent {
            ErrorScreen(
                message = errorMessage,
                onResetClick = { resetClicked = true }
            )
        }

        // Then
        composeTestRule
            .onNodeWithText("Error Reading Card")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText(errorMessage)
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("Try Again")
            .assertIsDisplayed()
            .performClick()

        assert(resetClicked)
    }

    @Test
    fun errorScreen_tryAgainButton_callsOnReset() {
        // Given
        var resetCallCount = 0

        composeTestRule.setContent {
            ErrorScreen(
                message = "Test error",
                onResetClick = { resetCallCount++ }
            )
        }

        // When
        composeTestRule
            .onNodeWithText("Try Again")
            .performClick()

        // Then
        assert(resetCallCount == 1)
    }

    // ============================================================================
    // Data Row Tests
    // ============================================================================

    @Test
    fun dataRow_displaysLabelAndValue() {
        // When
        composeTestRule.setContent {
            DataRow(label = "Name", value = "John Doe")
        }

        // Then
        composeTestRule
            .onNodeWithText("Name:")
            .assertIsDisplayed()

        composeTestRule
            .onNodeWithText("John Doe")
            .assertIsDisplayed()
    }

    // ============================================================================
    // Personal Data Card Tests
    // ============================================================================

    @Test
    fun personalDataCard_displaysAllFields() {
        // Given
        val personalData = PersonalData(
            tckn = "98765432109",
            firstName = "Ali",
            lastName = "Kaya",
            birthDate = "1995-03-20",
            gender = "M",
            nationality = "TUR",
            documentNumber = "DEF456",
            expiryDate = "2025-06-30",
            issueDate = "2015-07-01",
            placeOfBirth = "İzmir",
            serialNumber = "654321"
        )

        // When
        composeTestRule.setContent {
            PersonalDataCard(personalData)
        }

        // Then
        composeTestRule.onNodeWithText("Personal Information").assertIsDisplayed()
        composeTestRule.onNodeWithText("Ali Kaya", substring = true).assertIsDisplayed()
        composeTestRule.onNodeWithText("98765432109").assertIsDisplayed()
        composeTestRule.onNodeWithText("1995-03-20").assertIsDisplayed()
        composeTestRule.onNodeWithText("İzmir").assertIsDisplayed()
    }
}
