package com.turkey.eidnfc.domain.usecase

import com.turkey.eidnfc.domain.model.Result
import com.turkey.eidnfc.domain.usecase.base.UseCase
import kotlinx.coroutines.Dispatchers
import timber.log.Timber
import javax.inject.Inject

/**
 * Use case for validating MRZ data format and security requirements.
 *
 * Encapsulates MRZ validation business logic:
 * - Document number validation (1-9 alphanumeric characters)
 * - Date of birth validation (6 digits, YYMMDD format)
 * - Date of expiry validation (6 digits, YYMMDD format)
 *
 * For backward compatibility, also supports legacy 6-digit PIN format,
 * but the primary format is now MRZ data: "documentNumber|dateOfBirth|dateOfExpiry"
 *
 * This keeps validation logic separate from UI and repository layers.
 */
class ValidatePinUseCase @Inject constructor() : UseCase<String, Boolean>(
    dispatcher = Dispatchers.Default // CPU-bound operation
) {

    override suspend fun execute(parameters: String): Result<Boolean> {
        val input = parameters

        Timber.d("Validating PIN format...")

        // Try to parse as MRZ data (format: docNo|dob|doe)
        val parts = input.split("|")

        if (parts.size == 3) {
            return validateMrzData(parts[0], parts[1], parts[2])
        }

        // Reject old 6-digit PIN format
        Timber.w("PIN validation failed: MRZ format required")
        return Result.Error(
            IllegalArgumentException(
                "Please enter MRZ data in format: documentNumber|dateOfBirth|dateOfExpiry"
            )
        )
    }

    /**
     * Validates MRZ data fields.
     */
    private fun validateMrzData(
        documentNumber: String,
        dateOfBirth: String,
        dateOfExpiry: String
    ): Result<Boolean> {
        // Validate document number
        if (documentNumber.isEmpty() || documentNumber.length > 9) {
            Timber.w("MRZ validation failed: invalid document number length")
            return Result.Error(
                IllegalArgumentException("Document number must be 1-9 characters")
            )
        }

        if (!documentNumber.all { it.isLetterOrDigit() }) {
            Timber.w("MRZ validation failed: document number contains invalid characters")
            return Result.Error(
                IllegalArgumentException("Document number must contain only letters and digits")
            )
        }

        // Validate date of birth
        if (dateOfBirth.length != 6 || !dateOfBirth.all { it.isDigit() }) {
            Timber.w("MRZ validation failed: invalid date of birth format")
            return Result.Error(
                IllegalArgumentException("Date of birth must be 6 digits (YYMMDD)")
            )
        }

        if (!isValidDate(dateOfBirth)) {
            Timber.w("MRZ validation failed: invalid date of birth value")
            return Result.Error(
                IllegalArgumentException("Date of birth is not a valid date")
            )
        }

        // Validate date of expiry
        if (dateOfExpiry.length != 6 || !dateOfExpiry.all { it.isDigit() }) {
            Timber.w("MRZ validation failed: invalid date of expiry format")
            return Result.Error(
                IllegalArgumentException("Date of expiry must be 6 digits (YYMMDD)")
            )
        }

        if (!isValidDate(dateOfExpiry)) {
            Timber.w("MRZ validation failed: invalid date of expiry value")
            return Result.Error(
                IllegalArgumentException("Date of expiry is not a valid date")
            )
        }

        Timber.d("PIN validation successful")
        return Result.Success(true)
    }

    /**
     * Validates YYMMDD date format.
     */
    private fun isValidDate(date: String): Boolean {
        if (date.length != 6) return false

        return try {
            val month = date.substring(2, 4).toInt()
            val day = date.substring(4, 6).toInt()

            month in 1..12 && day in 1..31
        } catch (e: Exception) {
            false
        }
    }

    companion object {
        /**
         * Validates MRZ data synchronously without creating Result wrapper.
         * Useful for quick validation in UI.
         */
        fun validateQuick(input: String): Boolean {
            val parts = input.split("|")
            if (parts.size != 3) return false

            val (docNo, dob, doe) = parts

            return docNo.isNotEmpty() &&
                    docNo.length <= 9 &&
                    docNo.all { it.isLetterOrDigit() } &&
                    dob.length == 6 &&
                    dob.all { it.isDigit() } &&
                    doe.length == 6 &&
                    doe.all { it.isDigit() }
        }
    }
}
