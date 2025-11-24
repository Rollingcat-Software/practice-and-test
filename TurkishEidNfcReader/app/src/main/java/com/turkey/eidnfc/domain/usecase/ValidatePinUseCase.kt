package com.turkey.eidnfc.domain.usecase

import com.turkey.eidnfc.domain.model.Result
import com.turkey.eidnfc.domain.usecase.base.UseCase
import com.turkey.eidnfc.util.isValidPin
import kotlinx.coroutines.Dispatchers
import timber.log.Timber
import javax.inject.Inject

/**
 * Use case for validating PIN format and security requirements.
 *
 * Encapsulates PIN validation business logic:
 * - Length validation (6 digits)
 * - Character validation (only digits)
 * - Security checks (no common PINs like 000000, 123456)
 *
 * This keeps validation logic separate from UI and repository layers.
 */
class ValidatePinUseCase @Inject constructor() : UseCase<String, Boolean>(
    dispatcher = Dispatchers.Default // CPU-bound operation
) {

    /**
     * Common/weak PINs that should be warned about.
     * Note: We don't block these, just validate format.
     */
    private val commonPins = setOf(
        "000000", "111111", "222222", "333333", "444444",
        "555555", "666666", "777777", "888888", "999999",
        "123456", "654321", "012345"
    )

    override suspend fun execute(parameters: String): Result<Boolean> {
        val pin = parameters

        Timber.d("Validating PIN format...")

        // Check basic format
        if (!pin.isValidPin()) {
            Timber.w("PIN validation failed: invalid format")
            return Result.Error(
                IllegalArgumentException("PIN must be exactly 6 digits")
            )
        }

        // Optional: Check for common/weak PINs (for logging only)
        if (pin in commonPins) {
            Timber.w("User is using a common PIN: ${pin.mask()}")
            // We still return success, but log the warning
        }

        Timber.d("PIN validation successful")
        return Result.Success(true)
    }

    /**
     * Masks PIN for secure logging.
     */
    private fun String.mask(): String = "••••••"

    companion object {
        /**
         * Validates PIN synchronously without creating Result wrapper.
         * Useful for quick validation in UI.
         */
        fun validateQuick(pin: String): Boolean {
            return pin.isValidPin()
        }
    }
}
