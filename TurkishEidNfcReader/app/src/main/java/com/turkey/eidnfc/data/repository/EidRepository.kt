package com.turkey.eidnfc.data.repository

import android.nfc.Tag
import com.turkey.eidnfc.domain.model.CardData
import com.turkey.eidnfc.domain.model.Result

/**
 * Repository interface for Turkish eID card operations.
 *
 * This interface defines the contract for data operations related to eID cards.
 * It separates the data layer from the business logic layer (ViewModel),
 * making the code more testable and maintainable.
 */
interface EidRepository {

    /**
     * Reads card data from the NFC tag with PIN authentication.
     *
     * @param tag The NFC tag detected by the system
     * @param pin The 6-digit PIN for authentication
     * @return Result<CardData> containing the card data on success, or an error
     */
    suspend fun readCard(tag: Tag, pin: String): Result<CardData>

    /**
     * Validates if the provided PIN format is correct.
     *
     * @param pin The PIN to validate
     * @return True if PIN is valid (6 digits), false otherwise
     */
    fun validatePin(pin: String): Boolean
}
