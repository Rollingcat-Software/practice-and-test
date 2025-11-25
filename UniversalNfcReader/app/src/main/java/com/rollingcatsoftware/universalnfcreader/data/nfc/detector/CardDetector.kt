package com.rollingcatsoftware.universalnfcreader.data.nfc.detector

import android.nfc.Tag
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType

/**
 * Interface for card type detection.
 *
 * Implementations are responsible for identifying specific card types
 * by examining tag properties, attempting AID selection, or other methods.
 */
interface CardDetector {
    /**
     * Detect the card type from the given NFC tag.
     *
     * @param tag The NFC tag to analyze
     * @return The detected [CardType]
     */
    suspend fun detectCardType(tag: Tag): CardType

    /**
     * Get the list of NFC technologies supported by the tag.
     *
     * @param tag The NFC tag to analyze
     * @return List of technology class names (e.g., "android.nfc.tech.IsoDep")
     */
    fun getSupportedTechnologies(tag: Tag): List<String>

    /**
     * Get a simplified list of technology names.
     *
     * @param tag The NFC tag to analyze
     * @return List of simple technology names (e.g., "IsoDep", "MifareClassic")
     */
    fun getTechnologyNames(tag: Tag): List<String> {
        return getSupportedTechnologies(tag).map { tech ->
            tech.substringAfterLast('.')
        }
    }
}

/**
 * Interface for specific card type detectors.
 *
 * These are used by [UniversalCardDetector] to check for specific card types.
 */
interface SpecificCardDetector {
    /**
     * The card type this detector is responsible for.
     */
    val targetCardType: CardType

    /**
     * Priority for detection order. Lower numbers are checked first.
     * More specific detectors should have lower priority values.
     */
    val priority: Int

    /**
     * Check if this detector can identify the card type from the tag.
     *
     * @param tag The NFC tag to check
     * @return true if this detector can handle the card, false otherwise
     */
    suspend fun canDetect(tag: Tag): Boolean

    /**
     * Perform detailed detection to confirm the card type.
     *
     * Called only if [canDetect] returns true.
     *
     * @param tag The NFC tag to analyze
     * @return The confirmed [CardType] or null if detection fails
     */
    suspend fun detect(tag: Tag): CardType?
}
