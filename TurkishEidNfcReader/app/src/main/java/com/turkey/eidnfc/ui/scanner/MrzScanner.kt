package com.turkey.eidnfc.ui.scanner

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import timber.log.Timber

/**
 * Parsed MRZ data from OCR scan.
 *
 * @param documentNumber The document/card number (up to 9 alphanumeric characters)
 * @param dateOfBirth Date of birth in YYMMDD format
 * @param dateOfExpiry Date of expiry in YYMMDD format
 * @param confidence OCR confidence (0.0 to 1.0)
 * @param checksumValid True if all MRZ check digits validated correctly
 */
data class ScannedMrzData(
    val documentNumber: String,
    val dateOfBirth: String,
    val dateOfExpiry: String,
    val confidence: Float = 0f,
    val checksumValid: Boolean = false
)

/**
 * MRZ Scanner using ML Kit Text Recognition.
 *
 * Scans TD1 format (Turkish ID cards) which has 3 lines of 30 characters each.
 *
 * TD1 Format (ID Cards) - ICAO Doc 9303:
 * Line 1 (30 chars): Type (2) + Country (3) + Document Number (9) + Check (1) + Optional (15)
 *   - Positions 0-1: Document type (I< for ID card)
 *   - Positions 2-4: Country code (TUR for Turkey)
 *   - Positions 5-13: Document number (9 alphanumeric)
 *   - Position 14: Check digit
 *   - Positions 15-29: Optional data (may contain TCKN)
 *
 * Line 2 (30 chars): DOB (6) + Check (1) + Sex (1) + DOE (6) + Check (1) + Nationality (3) + Optional (11) + Check (1)
 *   - Positions 0-5: Date of birth (YYMMDD)
 *   - Position 6: Check digit
 *   - Position 7: Sex (M/F/<)
 *   - Positions 8-13: Date of expiry (YYMMDD)
 *   - Position 14: Check digit
 *   - Positions 15-17: Nationality (TUR)
 *   - Positions 18-28: Optional data
 *   - Position 29: Overall check digit
 *
 * Line 3 (30 chars): Name (Surname<<GivenNames)
 */
class MrzAnalyzer(
    private val onMrzDetected: (ScannedMrzData) -> Unit,
    private val onError: (String) -> Unit,
    private val onTextDetected: ((String) -> Unit)? = null
) : ImageAnalysis.Analyzer {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var isProcessing = false
    private var lastSuccessTime = 0L

    // Debounce successful scans to prevent flickering
    private val scanDebounceMs = 1500L

    // MRZ character set (uppercase letters, digits, and < filler)
    private val mrzCharRegex = Regex("^[A-Z0-9<]+$")

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Skip if still processing or recently succeeded
        if (isProcessing || (currentTime - lastSuccessTime < scanDebounceMs)) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        isProcessing = true

        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                val fullText = visionText.text
                Timber.d("OCR Text: $fullText")

                // Send detected text for live preview (filtered for MRZ-like content)
                val mrzLikeText = extractMrzLikeText(fullText)
                if (mrzLikeText.isNotEmpty()) {
                    onTextDetected?.invoke(mrzLikeText)
                }

                // Try to parse MRZ from the recognized text
                val mrzData = parseMrz(fullText)
                if (mrzData != null) {
                    lastSuccessTime = System.currentTimeMillis()
                    onMrzDetected(mrzData)
                }
            }
            .addOnFailureListener { e ->
                Timber.e(e, "MRZ OCR failed")
                onError("OCR failed: ${e.message}")
            }
            .addOnCompleteListener {
                isProcessing = false
                imageProxy.close()
            }
    }

    /**
     * Extract MRZ-like text for live display.
     * Filters and formats text that looks like it could be MRZ data.
     */
    private fun extractMrzLikeText(text: String): String {
        val lines = text.uppercase()
            .split("\n")
            .map { line -> line.replace(" ", "").trim() }
            .filter { line ->
                // MRZ lines typically have alphanumeric chars and < fillers
                line.length >= 15 &&
                    (line.contains("<") || line.any { it.isDigit() }) &&
                    line.count { it.isLetterOrDigit() || it == '<' } > line.length * 0.7
            }
            .take(3) // TD1 has 3 lines max

        return lines.joinToString("")
    }

    /**
     * Parse MRZ from OCR text.
     * Supports TD1 (ID cards) and TD3 (passports) formats.
     */
    private fun parseMrz(text: String): ScannedMrzData? {
        // Normalize text but preserve letters in document numbers
        val lines = text.uppercase()
            .split("\n")
            .map { line ->
                // Replace spaces but keep < as filler markers
                line.replace(" ", "").trim()
            }
            .filter { it.length >= 20 }

        Timber.d("MRZ lines found: $lines")

        // Try TD1 format (3 lines, 30 chars each - ID cards)
        val td1Result = parseTd1(lines)
        if (td1Result != null) return td1Result

        // Try TD3 format (2 lines, 44 chars each - passports)
        val td3Result = parseTd3(lines)
        if (td3Result != null) return td3Result

        // Try to find MRZ patterns anywhere in text
        return findMrzPatterns(text)
    }

    /**
     * Parse TD1 format (ID cards - 3 lines of 30 characters).
     *
     * Turkish ID Card TD1 Format (ICAO Doc 9303):
     * Line 1: I<TUR[DocNumber:9][Check:1][Optional:15]  (30 chars)
     * Line 2: [DOB:6][Check:1][Sex:1][DOE:6][Check:1][Nationality:3][Optional:11][Check:1] (30 chars)
     * Line 3: [Surname<<GivenNames] (30 chars)
     */
    private fun parseTd1(lines: List<String>): ScannedMrzData? {
        if (lines.size < 2) return null

        // Apply strict MRZ line validation
        val mrzLines = lines.mapNotNull { line ->
            normalizeMrzLine(line)
        }.filter { line ->
            // MRZ lines must be at least 28 chars and contain valid MRZ characters
            line.length >= 28 && mrzCharRegex.matches(line)
        }

        Timber.d("TD1 candidate lines after strict filter: $mrzLines")

        if (mrzLines.size < 2) return null

        // Try to find valid TD1 Line 1 (starts with document type + country code)
        val line1Candidates = mrzLines.filter { isTd1Line1(it) }
        val line2Candidates = mrzLines.filter { isTd1Line2(it) }

        Timber.d("TD1 Line1 candidates: $line1Candidates")
        Timber.d("TD1 Line2 candidates: $line2Candidates")

        // First pass: Try to find a pair with valid checksums
        for (line1 in line1Candidates) {
            for (line2 in line2Candidates) {
                Timber.d("Trying TD1 pair - Line1: $line1, Line2: $line2")

                val docNum = extractTd1DocumentNumber(line1)
                val dob = extractTd1DateOfBirth(line2)
                val doe = extractTd1DateOfExpiry(line2)

                Timber.d("Extracted - DocNum: $docNum, DOB: $dob, DOE: $doe")

                if (docNum != null && dob != null && doe != null) {
                    // Validate check digits for data integrity
                    val checksumValid = validateTd1CheckDigits(line1, line2)
                    Timber.d("Checksum validation result: $checksumValid")

                    // If checksum is valid, we have high confidence - return immediately
                    if (checksumValid) {
                        return ScannedMrzData(
                            documentNumber = docNum,
                            dateOfBirth = dob,
                            dateOfExpiry = doe,
                            confidence = 0.98f,
                            checksumValid = true
                        )
                    }
                }
            }
        }

        // Second pass: If no checksum-valid result, return best match without checksum
        // (OCR errors might have corrupted the check digit itself)
        for (line1 in line1Candidates) {
            for (line2 in line2Candidates) {
                val docNum = extractTd1DocumentNumber(line1)
                val dob = extractTd1DateOfBirth(line2)
                val doe = extractTd1DateOfExpiry(line2)

                if (docNum != null && dob != null && doe != null) {
                    Timber.d("Returning result without valid checksum - data may have OCR errors")
                    return ScannedMrzData(
                        documentNumber = docNum,
                        dateOfBirth = dob,
                        dateOfExpiry = doe,
                        confidence = 0.7f,
                        checksumValid = false
                    )
                }
            }
        }

        return null
    }

    /**
     * Normalize an OCR line to MRZ format.
     * Converts common OCR mistakes and removes invalid characters.
     */
    private fun normalizeMrzLine(line: String): String? {
        var normalized = line.uppercase()
            .replace(" ", "")
            .replace("«", "<")  // Common OCR mistake
            .replace("‹", "<")
            .replace("›", "<")
            .replace("|", "I")  // Pipe often mistaken for I
            .trim()

        // MRZ lines should be mostly alphanumeric + <
        val validChars = normalized.count { it.isLetterOrDigit() || it == '<' }
        if (validChars < normalized.length * 0.85) {
            return null
        }

        // Remove any remaining invalid characters
        normalized = normalized.filter { it.isLetterOrDigit() || it == '<' }

        return if (normalized.length >= 25) normalized else null
    }

    /**
     * Check if a line looks like TD1 Line 1 (document type + country + document number).
     * Line 1 must start with document type (I, A, C) and contain country code.
     */
    private fun isTd1Line1(line: String): Boolean {
        if (line.length < 15) return false

        // TD1 Line 1 patterns for ID cards:
        // - I<TUR... (Turkish ID)
        // - IDTUR... (alternative)
        // - I<XXX... (other countries)
        // - A<XXX... (administrative docs)
        // - C<XXX... (crew member certs)
        val startsWithType = line.matches(Regex("^[IAC][<A-Z][A-Z]{3}.*"))

        // Should have alphanumeric document number after position 5
        val hasDocNumber = if (line.length >= 14) {
            val docPart = line.substring(5, minOf(14, line.length))
            docPart.any { it.isLetterOrDigit() && it != '<' }
        } else false

        // Should NOT have consecutive dates at the start (that would be Line 2)
        val startsWithDate = line.take(6).all { it.isDigit() }

        return startsWithType && hasDocNumber && !startsWithDate
    }

    /**
     * Check if a line looks like TD1 Line 2 (dates + sex + nationality).
     * Line 2 starts with DOB (6 digits) and has DOE at position 8-13.
     */
    private fun isTd1Line2(line: String): Boolean {
        if (line.length < 18) return false

        // Extract potential dates with OCR correction
        val potentialDob = extractDateWithOcrCorrection(line, 0, 6)
        val potentialDoe = extractDateWithOcrCorrection(line, 8, 14)

        // Position 7 should be sex indicator (M, F, <, or X)
        val sexChar = if (line.length > 7) line[7] else ' '
        val validSex = sexChar in listOf('M', 'F', '<', 'X', 'H', 'W')  // H/W are OCR mistakes for M/F

        // Should have nationality code around position 15-17
        val hasNationality = if (line.length >= 18) {
            val natPart = line.substring(15, 18)
            natPart.all { it.isLetter() || it == '<' }
        } else false

        Timber.d("Line2 check - DOB: $potentialDob, DOE: $potentialDoe, Sex: $sexChar, Nat: $hasNationality")

        return potentialDob != null && potentialDoe != null && (validSex || hasNationality)
    }

    /**
     * Extract document number from TD1 line 1.
     * Format: [Type:2][Country:3][DocNumber:9][Check:1][Optional:15]
     * Turkish ID starts with "I<TUR" or "IDTUR"
     *
     * The document number is at positions 5-13 (9 characters).
     */
    private fun extractTd1DocumentNumber(line: String): String? {
        if (line.length < 14) return null

        // Verify this is a valid Line 1
        if (!isTd1Line1(line)) return null

        // Document number starts at position 5, length 9
        val docNumRaw = line.substring(5, minOf(14, line.length))

        // Clean the document number - remove < fillers from the end
        val docNum = docNumRaw.trimEnd('<').replace("<", "")

        if (docNum.isEmpty()) return null

        // Turkish document numbers are alphanumeric (like A12B34567)
        // Validate format: typically starts with letter(s) followed by digits
        // Do NOT apply aggressive OCR corrections that destroy letters

        Timber.d("TD1 DocNum raw: '$docNumRaw' -> cleaned: '$docNum'")

        // Document numbers should be 1-9 alphanumeric characters
        return if (docNum.length in 1..9 && docNum.all { it.isLetterOrDigit() }) {
            docNum
        } else {
            null
        }
    }

    /**
     * Extract date of birth from TD1 Line 2.
     * DOB is at positions 0-5 (6 digits in YYMMDD format).
     */
    private fun extractTd1DateOfBirth(line: String): String? {
        return extractDateWithOcrCorrection(line, 0, 6)
    }

    /**
     * Extract date of expiry from TD1 Line 2.
     * DOE is at positions 8-13 (6 digits in YYMMDD format).
     */
    private fun extractTd1DateOfExpiry(line: String): String? {
        return extractDateWithOcrCorrection(line, 8, 14)
    }

    /**
     * Parse TD3 format (passports - 2 lines of 44 characters).
     * TD3 Line 2: [DocNumber:9][Check:1][Nationality:3][DOB:6][Check:1][Sex:1][DOE:6][Check:1][Optional:14][Check:1]
     */
    private fun parseTd3(lines: List<String>): ScannedMrzData? {
        val mrzLines = lines.filter { it.length >= 40 }

        for (line in mrzLines) {
            if (line.length >= 28) {
                // Extract document number (first 9 chars, remove trailing <)
                val docNumRaw = line.take(9).trimEnd('<').replace("<", "")
                val docNum = if (docNumRaw.isNotEmpty() && docNumRaw.length <= 9) docNumRaw else null

                // DOB at position 13-18, DOE at position 21-26
                val dob = extractDateWithOcrCorrection(line, 13, 19)
                val doe = extractDateWithOcrCorrection(line, 21, 27)

                if (docNum != null && dob != null && doe != null) {
                    return ScannedMrzData(
                        documentNumber = docNum,
                        dateOfBirth = dob,
                        dateOfExpiry = doe,
                        confidence = 0.7f
                    )
                }
            }
        }

        return null
    }

    /**
     * Find MRZ patterns using regex anywhere in the text.
     * This is a fallback when structured TD1/TD3 parsing fails.
     *
     * IMPORTANT: This fallback is more conservative to avoid reading
     * random text like father's name as document numbers.
     */
    private fun findMrzPatterns(text: String): ScannedMrzData? {
        val cleanText = text.uppercase().replace(" ", "")

        // First, look for lines that contain MRZ-like content (< characters)
        val mrzLikeLines = cleanText.split("\n")
            .map { it.replace(" ", "") }
            .filter { line ->
                // MRZ lines must have < characters and be long enough
                line.length >= 25 && line.contains("<") && line.count { it == '<' } >= 3
            }

        Timber.d("Pattern search - MRZ-like lines: $mrzLikeLines")

        // If we found MRZ-like lines, try to extract from them
        for (line in mrzLikeLines) {
            // Look for document type pattern at start (I<TUR, I<XXX, etc.)
            val typeMatch = Regex("^[IAC]<[A-Z]{3}").find(line)
            if (typeMatch != null) {
                // This looks like TD1 Line 1 - extract doc number at position 5
                if (line.length >= 14) {
                    val docNum = line.substring(5, 14).trimEnd('<').replace("<", "")
                    if (docNum.isNotEmpty() && docNum.length <= 9) {
                        // Now look for dates in subsequent content
                        val dates = findDatesInText(cleanText)
                        if (dates.size >= 2) {
                            Timber.d("Pattern fallback found - DocNum: $docNum, Dates: $dates")
                            return ScannedMrzData(
                                documentNumber = docNum,
                                dateOfBirth = dates[0],
                                dateOfExpiry = dates[1],
                                confidence = 0.6f
                            )
                        }
                    }
                }
            }
        }

        // More conservative fallback - only if we find strong MRZ indicators
        // Don't just grab any alphanumeric string as document number
        Timber.d("Pattern search - No valid MRZ patterns found")
        return null
    }

    /**
     * Find valid YYMMDD dates in text with OCR correction.
     */
    private fun findDatesInText(text: String): List<String> {
        // Apply OCR correction for digits
        val correctedText = text.map { char ->
            when (char) {
                'O', 'Q', 'D' -> '0'
                'I', 'L' -> '1'
                'S' -> '5'
                'B' -> '8'
                else -> char
            }
        }.joinToString("")

        val dateRegex = Regex("\\d{6}")
        return dateRegex.findAll(correctedText)
            .map { it.value }
            .filter { isValidMrzDate(it) }
            .toList()
    }

    /**
     * Extract a date from a string at the given position with OCR correction.
     * Dates should only contain digits, so we can safely apply OCR corrections.
     */
    private fun extractDateWithOcrCorrection(text: String, start: Int, end: Int): String? {
        if (text.length < end) return null

        val rawDate = text.substring(start, minOf(end, text.length))

        // Apply OCR corrections for digits only
        val corrected = rawDate.map { char ->
            when (char) {
                'O', 'o', 'Q', 'D' -> '0'
                'I', 'i', 'l', 'L', '|', '!' -> '1'
                'Z', 'z' -> '2'
                'E' -> '3'
                'A', 'h' -> '4'
                'S', 's' -> '5'
                'G', 'b' -> '6'
                'T' -> '7'
                'B' -> '8'
                'g', 'q' -> '9'
                else -> char
            }
        }.joinToString("")

        val dateStr = corrected.filter { it.isDigit() }.take(6)
        Timber.d("Date extraction: raw='$rawDate' -> corrected='$corrected' -> digits='$dateStr'")

        return if (dateStr.length == 6 && isValidMrzDate(dateStr)) dateStr else null
    }

    /**
     * Extract a date from a string at the given position (legacy, no OCR correction).
     */
    private fun extractDate(text: String, start: Int, end: Int): String? {
        if (text.length < end) return null
        val dateStr = text.substring(start, end).filter { it.isDigit() }.take(6)
        return if (dateStr.length == 6 && isValidMrzDate(dateStr)) dateStr else null
    }

    /**
     * Validate MRZ date format (YYMMDD).
     */
    private fun isValidMrzDate(date: String): Boolean {
        if (date.length != 6) return false
        return try {
            val month = date.substring(2, 4).toInt()
            val day = date.substring(4, 6).toInt()
            month in 1..12 && day in 1..31
        } catch (e: Exception) {
            false
        }
    }

    // ==================== MRZ Check Digit Validation ====================
    // ICAO Doc 9303 specifies check digit calculation using weighted values

    /**
     * Character weights for MRZ check digit calculation.
     * Pattern: 7, 3, 1, 7, 3, 1, ... (repeating)
     */
    private val checkDigitWeights = intArrayOf(7, 3, 1)

    /**
     * Get numeric value of MRZ character for check digit calculation.
     * - Digits 0-9 → 0-9
     * - Letters A-Z → 10-35
     * - Filler '<' → 0
     */
    private fun getMrzCharValue(char: Char): Int {
        return when {
            char.isDigit() -> char.digitToInt()
            char.isLetter() -> char.uppercaseChar() - 'A' + 10
            char == '<' -> 0
            else -> 0
        }
    }

    /**
     * Calculate MRZ check digit for a given string.
     * Uses ICAO Doc 9303 algorithm: sum of (char_value * weight) mod 10
     *
     * @param data The string to calculate check digit for
     * @return The check digit (0-9)
     */
    private fun calculateCheckDigit(data: String): Int {
        var sum = 0
        data.forEachIndexed { index, char ->
            val value = getMrzCharValue(char)
            val weight = checkDigitWeights[index % 3]
            sum += value * weight
        }
        return sum % 10
    }

    /**
     * Verify a field value against its check digit.
     *
     * @param data The field value (e.g., document number, date)
     * @param checkDigit The check digit character from MRZ
     * @return True if the check digit is valid
     */
    private fun verifyCheckDigit(data: String, checkDigit: Char): Boolean {
        if (!checkDigit.isDigit()) return false
        val expected = calculateCheckDigit(data)
        val actual = checkDigit.digitToInt()
        val isValid = expected == actual
        Timber.d("Check digit verification: data='$data', expected=$expected, actual=$actual, valid=$isValid")
        return isValid
    }

    /**
     * Validate TD1 MRZ check digits.
     *
     * TD1 Line 1: [Type:2][Country:3][DocNum:9][Check:1][Optional:15]
     * TD1 Line 2: [DOB:6][Check:1][Sex:1][DOE:6][Check:1][Nat:3][Optional:11][Check:1]
     *
     * Validates:
     * - Document number check digit (Line 1, position 14)
     * - Date of birth check digit (Line 2, position 6)
     * - Date of expiry check digit (Line 2, position 14)
     *
     * @return True if all check digits are valid
     */
    private fun validateTd1CheckDigits(line1: String, line2: String): Boolean {
        if (line1.length < 15 || line2.length < 15) return false

        // Document number: positions 5-13, check digit at 14
        val docNum = line1.substring(5, 14)
        val docNumCheck = line1[14]
        val docNumValid = verifyCheckDigit(docNum, docNumCheck)

        // Date of birth: positions 0-5, check digit at 6
        val dob = line2.substring(0, 6)
        val dobCheck = line2[6]
        val dobValid = verifyCheckDigit(dob, dobCheck)

        // Date of expiry: positions 8-13, check digit at 14
        val doe = line2.substring(8, 14)
        val doeCheck = line2[14]
        val doeValid = verifyCheckDigit(doe, doeCheck)

        Timber.d("TD1 checksum validation - DocNum: $docNumValid, DOB: $dobValid, DOE: $doeValid")

        return docNumValid && dobValid && doeValid
    }

    fun close() {
        recognizer.close()
    }
}
