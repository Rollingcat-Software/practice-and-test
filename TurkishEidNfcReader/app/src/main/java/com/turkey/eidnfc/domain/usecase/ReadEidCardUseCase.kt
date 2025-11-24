package com.turkey.eidnfc.domain.usecase

import android.nfc.Tag
import com.turkey.eidnfc.data.repository.EidRepository
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.Result
import com.turkey.eidnfc.domain.usecase.base.UseCase
import kotlinx.coroutines.Dispatchers
import timber.log.Timber
import javax.inject.Inject

/**
 * Use case for reading Turkish eID card via NFC.
 *
 * Orchestrates the complete card reading flow:
 * 1. Validates PIN format
 * 2. Calls repository to read card
 * 3. Performs business logic validation on card data
 * 4. Returns structured result
 *
 * This encapsulates the business logic and keeps it separate from:
 * - UI layer (ViewModel)
 * - Data layer (Repository, NFC Reader)
 */
class ReadEidCardUseCase @Inject constructor(
    private val repository: EidRepository,
    private val validatePinUseCase: ValidatePinUseCase
) : UseCase<ReadEidCardUseCase.Params, CardData>(
    dispatcher = Dispatchers.IO // NFC operations are I/O bound
) {

    override suspend fun execute(parameters: Params): Result<CardData> {
        val (tag, pin) = parameters

        Timber.d("Starting eID card read operation...")

        // Step 1: Validate PIN
        when (val validationResult = validatePinUseCase(pin)) {
            is Result.Error -> {
                Timber.e("PIN validation failed: ${validationResult.exception.message}")
                return Result.Error(validationResult.exception)
            }
            is Result.Success -> {
                Timber.d("PIN validation successful")
            }
            Result.Loading -> {
                // Should not happen, but handle it
                return Result.Error(IllegalStateException("Unexpected loading state"))
            }
        }

        // Step 2: Read card from repository
        Timber.d("Reading card from NFC tag...")
        when (val readResult = repository.readCard(tag, pin)) {
            is Result.Success -> {
                val cardData = readResult.data
                Timber.d("Card read successful")

                // Step 3: Perform business logic validation
                return validateCardData(cardData)
            }

            is Result.Error -> {
                Timber.e("Card read failed: ${readResult.exception.message}")
                return Result.Error(readResult.exception)
            }

            Result.Loading -> {
                return Result.Loading
            }
        }
    }

    /**
     * Validates card data business rules.
     *
     * This is where you'd add domain-specific validation:
     * - Check if card is expired
     * - Validate TCKN checksum
     * - Verify mandatory fields are present
     */
    private fun validateCardData(cardData: CardData): Result<CardData> {
        // Check if we got any data
        if (cardData.personalData == null) {
            Timber.w("Card read successful but no personal data found")
            return Result.Error(
                IllegalStateException("No personal data found on card")
            )
        }

        val personalData = cardData.personalData

        // Validate TCKN is present
        if (personalData.tckn.isEmpty()) {
            Timber.e("Invalid card data: TCKN is empty")
            return Result.Error(
                IllegalStateException("Invalid card: Missing TCKN")
            )
        }

        // Validate basic fields are present
        if (personalData.firstName.isEmpty() || personalData.lastName.isEmpty()) {
            Timber.e("Invalid card data: Missing name fields")
            return Result.Error(
                IllegalStateException("Invalid card: Missing name information")
            )
        }

        // Optional: Check expiry date
        // This would require parsing the expiryDate and comparing with current date
        // For now, we just log it
        Timber.d("Card data validation successful for TCKN: ${personalData.tckn.take(3)}***")

        return Result.Success(cardData)
    }

    /**
     * Parameters for reading eID card.
     *
     * @param tag NFC tag detected by the system
     * @param pin 6-digit PIN for authentication
     */
    data class Params(
        val tag: Tag,
        val pin: String
    )
}
