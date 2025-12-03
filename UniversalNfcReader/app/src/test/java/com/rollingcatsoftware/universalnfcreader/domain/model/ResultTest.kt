package com.rollingcatsoftware.universalnfcreader.domain.model

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Unit tests for [Result] wrapper class.
 */
class ResultTest {

    @Test
    fun `Success result returns correct data`() {
        val data = "test data"
        val result: Result<String> = Result.Success(data)

        assertThat(result.isSuccess).isTrue()
        assertThat(result.isError).isFalse()
        assertThat(result.getOrNull()).isEqualTo(data)
        assertThat(result.errorOrNull()).isNull()
    }

    @Test
    fun `Error result returns correct error`() {
        val error = CardError.Timeout()
        val result: Result<String> = Result.Error(error)

        assertThat(result.isSuccess).isFalse()
        assertThat(result.isError).isTrue()
        assertThat(result.getOrNull()).isNull()
        assertThat(result.errorOrNull()).isEqualTo(error)
    }

    @Test
    fun `getOrDefault returns default for error`() {
        val result: Result<Int> = Result.Error(CardError.Unknown())
        val default = 42

        assertThat(result.getOrDefault(default)).isEqualTo(default)
    }

    @Test
    fun `getOrDefault returns data for success`() {
        val result: Result<Int> = Result.Success(100)

        assertThat(result.getOrDefault(42)).isEqualTo(100)
    }

    @Test
    fun `map transforms success value`() {
        val result: Result<Int> = Result.Success(5)
        val mapped = result.map { it * 2 }

        assertThat(mapped.getOrNull()).isEqualTo(10)
    }

    @Test
    fun `map preserves error`() {
        val error = CardError.Unknown()
        val result: Result<Int> = Result.Error(error)
        val mapped = result.map { it * 2 }

        assertThat(mapped.isError).isTrue()
        assertThat(mapped.errorOrNull()).isEqualTo(error)
    }

    @Test
    fun `flatMap chains successful results`() {
        val result: Result<Int> = Result.Success(5)
        val flatMapped = result.flatMap { value ->
            if (value > 0) Result.Success(value.toString())
            else Result.Error(CardError.Unknown())
        }

        assertThat(flatMapped.getOrNull()).isEqualTo("5")
    }

    @Test
    fun `flatMap short-circuits on error`() {
        val error = CardError.Timeout()
        val result: Result<Int> = Result.Error(error)
        var wasCalled = false

        val flatMapped = result.flatMap { value ->
            wasCalled = true
            Result.Success(value.toString())
        }

        assertThat(wasCalled).isFalse()
        assertThat(flatMapped.errorOrNull()).isEqualTo(error)
    }

    @Test
    fun `onSuccess executes block for success`() {
        var captured: String? = null
        val result: Result<String> = Result.Success("test")

        result.onSuccess { captured = it }

        assertThat(captured).isEqualTo("test")
    }

    @Test
    fun `onSuccess does not execute for error`() {
        var captured: String? = null
        val result: Result<String> = Result.Error(CardError.Unknown())

        result.onSuccess { captured = it }

        assertThat(captured).isNull()
    }

    @Test
    fun `onError executes block for error`() {
        var captured: CardError? = null
        val error = CardError.ConnectionLost()
        val result: Result<String> = Result.Error(error)

        result.onError { captured = it }

        assertThat(captured).isEqualTo(error)
    }

    @Test
    fun `fold handles success case`() {
        val result: Result<Int> = Result.Success(10)

        val folded = result.fold(
            onSuccess = { "success: $it" },
            onError = { "error: ${it.message}" }
        )

        assertThat(folded).isEqualTo("success: 10")
    }

    @Test
    fun `fold handles error case`() {
        val result: Result<Int> = Result.Error(CardError.Timeout(message = "test error"))

        val folded = result.fold(
            onSuccess = { "success: $it" },
            onError = { "error: ${it.code}" }
        )

        assertThat(folded).isEqualTo("error: ERR_TIMEOUT")
    }

    @Test
    fun `zip combines two success results`() {
        val result1: Result<Int> = Result.Success(1)
        val result2: Result<String> = Result.Success("a")

        val zipped = result1.zip(result2)

        assertThat(zipped.getOrNull()).isEqualTo(1 to "a")
    }

    @Test
    fun `zip returns first error`() {
        val error = CardError.Timeout()
        val result1: Result<Int> = Result.Error(error)
        val result2: Result<String> = Result.Success("a")

        val zipped = result1.zip(result2)

        assertThat(zipped.errorOrNull()).isEqualTo(error)
    }

    @Test
    fun `toResult converts non-null to success`() {
        val value: String? = "test"
        val result = value.toResult()

        assertThat(result.isSuccess).isTrue()
        assertThat(result.getOrNull()).isEqualTo("test")
    }

    @Test
    fun `toResult converts null to error`() {
        val value: String? = null
        val result = value.toResult()

        assertThat(result.isError).isTrue()
    }

    @Test
    fun `runCatching returns success for normal execution`() {
        val result = Result.runCatching { 42 }

        assertThat(result.isSuccess).isTrue()
        assertThat(result.getOrNull()).isEqualTo(42)
    }

    @Test
    fun `runCatching returns error for CardException`() {
        val error = CardError.ConnectionLost()
        val result = Result.runCatching<Int> {
            throw CardException(error)
        }

        assertThat(result.isError).isTrue()
        assertThat(result.errorOrNull()).isEqualTo(error)
    }
}
