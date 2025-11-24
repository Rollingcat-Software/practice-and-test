package com.turkey.eidnfc.domain.usecase.base

import com.turkey.eidnfc.domain.model.Result
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Base class for all use cases in the application.
 *
 * Use cases represent business logic operations and encapsulate single responsibilities.
 * They sit between the presentation layer (ViewModel) and data layer (Repository).
 *
 * Benefits:
 * - Single Responsibility Principle
 * - Easier to test business logic in isolation
 * - Reusable across different ViewModels
 * - Clear separation of concerns
 *
 * @param P Input parameters type
 * @param R Return type (wrapped in Result)
 */
abstract class UseCase<in P, R>(
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    /**
     * Executes the use case with given parameters.
     *
     * Automatically switches to the appropriate dispatcher (IO by default)
     * and handles exceptions.
     *
     * @param parameters Input parameters
     * @return Result containing the operation outcome
     */
    suspend operator fun invoke(parameters: P): Result<R> {
        return try {
            withContext(dispatcher) {
                execute(parameters)
            }
        } catch (e: Exception) {
            Result.Error(e)
        }
    }

    /**
     * Contains the actual business logic.
     * Implement this method in concrete use cases.
     *
     * @param parameters Input parameters
     * @return Result containing the operation outcome
     */
    protected abstract suspend fun execute(parameters: P): Result<R>
}

/**
 * Use case that doesn't require parameters.
 */
abstract class NoParamsUseCase<R>(
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    suspend operator fun invoke(): Result<R> {
        return try {
            withContext(dispatcher) {
                execute()
            }
        } catch (e: Exception) {
            Result.Error(e)
        }
    }

    protected abstract suspend fun execute(): Result<R>
}
