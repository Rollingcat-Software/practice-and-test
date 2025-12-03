package com.rollingcatsoftware.universalnfcreader.data.nfc.reader

import android.nfc.Tag
import com.rollingcatsoftware.universalnfcreader.domain.model.AuthenticationData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.Result

/**
 * Base interface for all card readers.
 *
 * Each implementation handles reading a specific card type or family.
 * Readers should be stateless and thread-safe.
 */
interface CardReader {
    /**
     * Card types supported by this reader.
     */
    val supportedCardTypes: List<CardType>

    /**
     * Whether this reader requires authentication to read data.
     * If true, [readCardWithAuth] should be used instead of [readCard].
     */
    fun requiresAuthentication(): Boolean

    /**
     * Read card data without authentication.
     *
     * For cards that require authentication, this will return either:
     * - Partial data (UID, basic info)
     * - AuthenticationRequired error
     *
     * @param tag The NFC tag to read
     * @return Result containing CardData or CardError
     */
    suspend fun readCard(tag: Tag): Result<CardData>

    /**
     * Read card data with authentication.
     *
     * @param tag The NFC tag to read
     * @param authData Authentication credentials
     * @return Result containing CardData or CardError
     */
    suspend fun readCardWithAuth(tag: Tag, authData: AuthenticationData): Result<CardData>

    /**
     * Check if this reader can handle the given card type.
     */
    fun canRead(cardType: CardType): Boolean = cardType in supportedCardTypes
}

/**
 * Base class providing common functionality for card readers.
 */
abstract class BaseCardReader : CardReader {

    /**
     * Read basic card information (UID, technologies).
     * Always succeeds if tag is valid.
     */
    protected fun readBasicInfo(tag: Tag): BasicCardInfo {
        return BasicCardInfo(
            uid = tag.id,
            technologies = tag.techList.toList()
        )
    }

    /**
     * Default implementation throws authentication required error.
     * Override in readers that support authentication.
     */
    override suspend fun readCardWithAuth(
        tag: Tag,
        authData: AuthenticationData
    ): Result<CardData> {
        // Default: delegate to readCard, ignore auth
        return readCard(tag)
    }

    /**
     * Basic card info container.
     */
    protected data class BasicCardInfo(
        val uid: ByteArray,
        val technologies: List<String>
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is BasicCardInfo) return false
            return uid.contentEquals(other.uid)
        }

        override fun hashCode(): Int = uid.contentHashCode()
    }
}
