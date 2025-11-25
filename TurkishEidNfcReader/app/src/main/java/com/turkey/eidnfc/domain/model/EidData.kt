package com.turkey.eidnfc.domain.model

import android.graphics.Bitmap

/**
 * Represents personal data from DG1 (Data Group 1) of Turkish eID.
 *
 * Contains the basic identity information stored on the card.
 */
data class PersonalData(
    val tckn: String,              // Turkish Citizenship Number (11 digits)
    val firstName: String,          // Given name(s)
    val lastName: String,           // Surname
    val birthDate: String,          // Date of birth
    val gender: String,             // Gender (M/F)
    val nationality: String,        // Nationality (typically "TUR")
    val documentNumber: String,     // Document number
    val expiryDate: String,         // Card expiry date
    val issueDate: String,          // Card issue date
    val placeOfBirth: String,       // Place of birth
    val serialNumber: String        // Serial number of the card
)

/**
 * Represents the complete eID card data including photo.
 */
data class CardData(
    val personalData: PersonalData?,
    val photo: Bitmap?,
    val cardAccessData: ByteArray?,
    val sodData: ByteArray?,
    val isAuthenticated: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as CardData

        if (personalData != other.personalData) return false
        if (photo != other.photo) return false
        if (cardAccessData != null) {
            if (other.cardAccessData == null) return false
            if (!cardAccessData.contentEquals(other.cardAccessData)) return false
        } else if (other.cardAccessData != null) return false
        if (sodData != null) {
            if (other.sodData == null) return false
            if (!sodData.contentEquals(other.sodData)) return false
        } else if (other.sodData != null) return false
        if (isAuthenticated != other.isAuthenticated) return false

        return true
    }

    override fun hashCode(): Int {
        var result = personalData?.hashCode() ?: 0
        result = 31 * result + (photo?.hashCode() ?: 0)
        result = 31 * result + (cardAccessData?.contentHashCode() ?: 0)
        result = 31 * result + (sodData?.contentHashCode() ?: 0)
        result = 31 * result + isAuthenticated.hashCode()
        return result
    }
}

/**
 * Sealed class representing the result of NFC operations.
 */
sealed class NfcResult<out T> {
    data class Success<T>(val data: T) : NfcResult<T>()
    data class Error(val error: NfcError) : NfcResult<Nothing>()
    data object Loading : NfcResult<Nothing>()
}

/**
 * Enumeration of possible NFC operation errors.
 */
sealed class NfcError {
    data class WrongPin(val attemptsRemaining: Int) : NfcError()
    data object CardLocked : NfcError()
    data object SecurityNotSatisfied : NfcError()
    data object FileNotFound : NfcError()
    data object InvalidCard : NfcError()
    data object NfcNotAvailable : NfcError()
    data object NfcDisabled : NfcError()
    data object ConnectionLost : NfcError()
    data object Timeout : NfcError()
    data object ParseError : NfcError()
    data class UnknownError(val message: String, val statusCode: Int? = null) : NfcError()
}

/**
 * Extension function to get user-friendly error message.
 */
fun NfcError.toUserMessage(): String {
    return when (this) {
        is NfcError.WrongPin ->
            "Wrong PIN. $attemptsRemaining attempt(s) remaining."

        NfcError.CardLocked ->
            "Card is locked. Contact authorities to unlock."

        NfcError.SecurityNotSatisfied ->
            "Security condition not satisfied. Please verify PIN."

        NfcError.FileNotFound ->
            "Required file not found on card."

        NfcError.InvalidCard ->
            "This is not a valid Turkish eID card."

        NfcError.NfcNotAvailable ->
            "NFC is not available on this device."

        NfcError.NfcDisabled ->
            "Please enable NFC in device settings."

        NfcError.ConnectionLost ->
            "Connection to card lost. Please try again."

        NfcError.Timeout ->
            "Operation timed out. Please try again."

        NfcError.ParseError ->
            "Failed to parse card data."

        is NfcError.UnknownError ->
            "Error: $message ${statusCode?.let { "(0x${String.format("%04X", it)})" } ?: ""}"
    }
}
