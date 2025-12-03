package com.rollingcatsoftware.universalnfcreader.domain.model

/**
 * A discriminated union that encapsulates a successful outcome with a value of type [T]
 * or a failure with an error of type [CardError].
 *
 * This follows the Railway-Oriented Programming pattern for error handling,
 * avoiding exceptions for expected error cases.
 *
 * Usage:
 * ```kotlin
 * when (val result = readCard(tag)) {
 *     is Result.Success -> handleData(result.data)
 *     is Result.Error -> handleError(result.error)
 * }
 * ```
 */
sealed class Result<out T> {
    /**
     * Represents a successful operation with the resulting data.
     */
    data class Success<T>(val data: T) : Result<T>()

    /**
     * Represents a failed operation with the error details.
     */
    data class Error(val error: CardError) : Result<Nothing>()

    /**
     * Returns true if this is a Success.
     */
    val isSuccess: Boolean get() = this is Success

    /**
     * Returns true if this is an Error.
     */
    val isError: Boolean get() = this is Error

    /**
     * Returns the data if Success, or null if Error.
     */
    fun getOrNull(): T? = when (this) {
        is Success -> data
        is Error -> null
    }

    /**
     * Returns the error if Error, or null if Success.
     */
    fun errorOrNull(): CardError? = when (this) {
        is Success -> null
        is Error -> error
    }

    /**
     * Returns the data if Success, or the default value if Error.
     */
    fun getOrDefault(default: @UnsafeVariance T): T = when (this) {
        is Success -> data
        is Error -> default
    }

    /**
     * Returns the data if Success, or throws the error as an exception.
     */
    fun getOrThrow(): T = when (this) {
        is Success -> data
        is Error -> throw CardException(error)
    }

    /**
     * Transform the success value with the given function.
     */
    inline fun <R> map(transform: (T) -> R): Result<R> = when (this) {
        is Success -> Success(transform(data))
        is Error -> Error(error)
    }

    /**
     * Transform the success value with a function that returns a Result.
     */
    inline fun <R> flatMap(transform: (T) -> Result<R>): Result<R> = when (this) {
        is Success -> transform(data)
        is Error -> Error(error)
    }

    /**
     * Transform the error with the given function.
     */
    inline fun mapError(transform: (CardError) -> CardError): Result<T> = when (this) {
        is Success -> this
        is Error -> Error(transform(error))
    }

    /**
     * Execute the given block if this is a Success.
     */
    inline fun onSuccess(block: (T) -> Unit): Result<T> {
        if (this is Success) block(data)
        return this
    }

    /**
     * Execute the given block if this is an Error.
     */
    inline fun onError(block: (CardError) -> Unit): Result<T> {
        if (this is Error) block(error)
        return this
    }

    /**
     * Fold the result into a single value.
     */
    inline fun <R> fold(
        onSuccess: (T) -> R,
        onError: (CardError) -> R
    ): R = when (this) {
        is Success -> onSuccess(data)
        is Error -> onError(error)
    }

    companion object {
        /**
         * Create a Success result.
         */
        fun <T> success(data: T): Result<T> = Success(data)

        /**
         * Create an Error result.
         */
        fun <T> error(error: CardError): Result<T> = Error(error)

        /**
         * Wrap a suspending block in a Result, catching exceptions.
         */
        suspend fun <T> catching(block: suspend () -> T): Result<T> = try {
            Success(block())
        } catch (e: CardException) {
            Error(e.error)
        } catch (e: Exception) {
            Error(CardError.Unknown(message = e.message ?: "Unknown error"))
        }

        /**
         * Wrap a block in a Result, catching exceptions.
         */
        inline fun <T> runCatching(block: () -> T): Result<T> = try {
            Success(block())
        } catch (e: CardException) {
            Error(e.error)
        } catch (e: Exception) {
            Error(CardError.Unknown(message = e.message ?: "Unknown error"))
        }
    }
}

/**
 * Exception wrapper for CardError to allow throwing when necessary.
 */
class CardException(val error: CardError) : Exception(error.message)

/**
 * Combine two results into a pair if both are successful.
 */
fun <A, B> Result<A>.zip(other: Result<B>): Result<Pair<A, B>> = when {
    this is Result.Success && other is Result.Success -> Result.Success(data to other.data)
    this is Result.Error -> Result.Error(error)
    other is Result.Error -> Result.Error(other.error)
    else -> Result.Error(CardError.Unknown())
}

/**
 * Extension to convert a nullable value to a Result.
 */
fun <T> T?.toResult(errorIfNull: CardError = CardError.Unknown()): Result<T> =
    if (this != null) Result.Success(this) else Result.Error(errorIfNull)
