package com.rollingcatsoftware.universalnfcreader.domain.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Unit tests for [CardError] and its factory methods.
 */
class CardErrorTest {

    @Test
    fun `ConnectionLost error is recoverable`() {
        val error = CardError.ConnectionLost()

        assertThat(error.isRecoverable).isTrue()
        assertThat(error.code).isEqualTo("ERR_CONNECTION_LOST")
    }

    @Test
    fun `CardBlocked error is not recoverable`() {
        val error = CardError.CardBlocked()

        assertThat(error.isRecoverable).isFalse()
        assertThat(error.code).isEqualTo("ERR_CARD_BLOCKED")
    }

    @Test
    fun `UnsupportedCard error contains detected technologies`() {
        val techs = listOf("IsoDep", "NfcA")
        val error = CardError.UnsupportedCard(detectedTechnologies = techs)

        assertThat(error.detectedTechnologies).isEqualTo(techs)
        assertThat(error.isRecoverable).isFalse()
    }

    @Test
    fun `AuthenticationFailed error contains remaining attempts`() {
        val error = CardError.AuthenticationFailed(attemptsRemaining = 2)

        assertThat(error.attemptsRemaining).isEqualTo(2)
        assertThat(error.isRecoverable).isTrue()
    }

    @Test
    fun `fromApduStatusWord returns success error for 63C3`() {
        val error = CardError.fromApduStatusWord(0x63C3)

        assertThat(error).isInstanceOf(CardError.AuthenticationFailed::class.java)
        assertThat((error as CardError.AuthenticationFailed).attemptsRemaining).isEqualTo(3)
    }

    @Test
    fun `fromApduStatusWord returns blocked for 63C0`() {
        val error = CardError.fromApduStatusWord(0x63C0)

        assertThat(error).isInstanceOf(CardError.CardBlocked::class.java)
    }

    @Test
    fun `fromApduStatusWord returns auth required for 6982`() {
        val error = CardError.fromApduStatusWord(0x6982)

        assertThat(error).isInstanceOf(CardError.AuthenticationRequired::class.java)
    }

    @Test
    fun `fromApduStatusWord returns card blocked for 6983`() {
        val error = CardError.fromApduStatusWord(0x6983)

        assertThat(error).isInstanceOf(CardError.CardBlocked::class.java)
    }

    @Test
    fun `fromApduStatusWord returns invalid response for file not found`() {
        val error = CardError.fromApduStatusWord(0x6A82)

        assertThat(error).isInstanceOf(CardError.InvalidResponse::class.java)
        assertThat((error as CardError.InvalidResponse).statusWord).isEqualTo(0x6A82)
    }

    @Test
    fun `fromApduStatusWord throws for success status`() {
        try {
            CardError.fromApduStatusWord(0x9000)
            assertThat(false).isTrue() // Should not reach here
        } catch (e: IllegalArgumentException) {
            assertThat(e.message).contains("9000")
        }
    }

    @Test
    fun `fromApduStatusWord returns generic invalid response for unknown code`() {
        val error = CardError.fromApduStatusWord(0x6999)

        assertThat(error).isInstanceOf(CardError.InvalidResponse::class.java)
        assertThat((error as CardError.InvalidResponse).statusWord).isEqualTo(0x6999)
    }

    @Test
    fun `Timeout error includes timeout value`() {
        val error = CardError.Timeout(timeoutMs = 5000)

        assertThat(error.timeoutMs).isEqualTo(5000)
        assertThat(error.code).isEqualTo("ERR_TIMEOUT")
    }

    @Test
    fun `ParseError includes field name`() {
        val error = CardError.ParseError(field = "dateOfBirth")

        assertThat(error.field).isEqualTo("dateOfBirth")
        assertThat(error.isRecoverable).isFalse()
    }

    @Test
    fun `SecurityValidationFailed includes validation type`() {
        val error = CardError.SecurityValidationFailed(validationType = "SOD_SIGNATURE")

        assertThat(error.validationType).isEqualTo("SOD_SIGNATURE")
        assertThat(error.isRecoverable).isFalse()
    }

    @Test
    fun `IoError is recoverable`() {
        val error = CardError.IoError()

        assertThat(error.isRecoverable).isTrue()
        assertThat(error.code).isEqualTo("ERR_IO")
    }
}
