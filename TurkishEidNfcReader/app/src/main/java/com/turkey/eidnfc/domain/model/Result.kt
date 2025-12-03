package com.turkey.eidnfc.domain.model

/**
 * A generic wrapper for operation results that can be either Success or Error.
 * This provides type-safe error handling and eliminates the need for exceptions
 * in expected error cases.
 */
sealed class Result<out T> {
    /**
     * Represents a successful operation with data.
     */
    data class Success<T>(val data: T) : Result<T>()

    /**
     * Represents a failed operation with an error.
     */
    data class Error(val exception: Exception) : Result<Nothing>()

    /**
     * Represents an ongoing operation (loading state).
     */
    data object Loading : Result<Nothing>()

    /**
     * Returns true if this is a Success result.
     */
    val isSuccess: Boolean
        get() = this is Success

    /**
     * Returns true if this is an Error result.
     */
    val isError: Boolean
        get() = this is Error

    /**
     * Returns true if this is a Loading result.
     */
    val isLoading: Boolean
        get() = this is Loading

    /**
     * Returns the data if Success, null otherwise.
     */
    fun getOrNull(): T? = when (this) {
        is Success -> data
        else -> null
    }

    /**
     * Returns the exception if Error, null otherwise.
     */
    fun exceptionOrNull(): Exception? = when (this) {
        is Error -> exception
        else -> null
    }
}

/**
 * Executes [action] if this is a Success result.
 * Returns the original Result for chaining.
 */
inline fun <T> Result<T>.onSuccess(action: (T) -> Unit): Result<T> {
    if (this is Result.Success) {
        action(data)
    }
    return this
}

/**
 * Executes [action] if this is an Error result.
 * Returns the original Result for chaining.
 */
inline fun <T> Result<T>.onError(action: (Exception) -> Unit): Result<T> {
    if (this is Result.Error) {
        action(exception)
    }
    return this
}

/**
 * Executes [action] if this is a Loading result.
 * Returns the original Result for chaining.
 */
inline fun <T> Result<T>.onLoading(action: () -> Unit): Result<T> {
    if (this is Result.Loading) {
        action()
    }
    return this
}

/**
 * Maps the success value to a new value using [transform].
 * Error and Loading states pass through unchanged.
 */
inline fun <T, R> Result<T>.map(transform: (T) -> R): Result<R> {
    return when (this) {
        is Result.Success -> Result.Success(transform(data))
        is Result.Error -> Result.Error(exception)
        Result.Loading -> Result.Loading
    }
}

/**
 * Flat-maps the success value to a new Result using [transform].
 * Error and Loading states pass through unchanged.
 */
inline fun <T, R> Result<T>.flatMap(transform: (T) -> Result<R>): Result<R> {
    return when (this) {
        is Result.Success -> transform(data)
        is Result.Error -> Result.Error(exception)
        Result.Loading -> Result.Loading
    }
}
