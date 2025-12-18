package com.rollingcatsoftware.universalnfcreader.data.nfc.eid

import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger

/**
 * MRZ (Machine Readable Zone) Parser for ICAO 9303 documents.
 *
 * Supports:
 * - TD1 format: ID cards (3 lines × 30 characters)
 * - TD3 format: Passports (2 lines × 44 characters)
 *
 * ICAO Doc 9303 Part 3 defines the MRZ structure.
 *
 * Check digit calculation uses weights: 7, 3, 1 (repeating)
 * Character values: 0-9 = 0-9, A-Z = 10-35, < = 0
 */
object MrzParser {

    private const val TAG = "MrzParser"

    /**
     * Document type enumeration.
     */
    enum class DocumentType {
        TD1,    // ID Card (3 × 30)
        TD2,    // Visa (2 × 36) - not commonly used
        TD3,    // Passport (2 × 44)
        UNKNOWN
    }

    /**
     * Parsed MRZ data structure.
     */
    data class MrzData(
        val documentType: DocumentType,
        val documentCode: String,           // P, I, A, C, V
        val issuingCountry: String,         // 3-letter code
        val documentNumber: String,         // Up to 9 chars
        val dateOfBirth: String,            // YYMMDD
        val dateOfExpiry: String,           // YYMMDD
        val sex: String,                    // M, F, <
        val nationality: String,            // 3-letter code
        val surname: String,
        val givenNames: String,
        val personalNumber: String = "",    // Optional data
        val checksumValid: Boolean = false,
        val rawMrz: String = ""
    ) {
        /**
         * Get date of birth formatted as DD/MM/YYYY.
         */
        fun getFormattedDateOfBirth(): String = formatDate(dateOfBirth)

        /**
         * Get date of expiry formatted as DD/MM/YYYY.
         */
        fun getFormattedDateOfExpiry(): String = formatDate(dateOfExpiry)

        /**
         * Full name (given names + surname).
         */
        val fullName: String
            get() = if (givenNames.isNotEmpty()) "$givenNames $surname" else surname

        private fun formatDate(yymmdd: String): String {
            if (yymmdd.length != 6) return yymmdd
            val yy = yymmdd.substring(0, 2).toIntOrNull() ?: return yymmdd
            val mm = yymmdd.substring(2, 4)
            val dd = yymmdd.substring(4, 6)
            // Assume < 30 is 2000s, >= 30 is 1900s for DOB
            // For DOE, assume all are 2000s
            val yyyy = if (yy < 30) 2000 + yy else 1900 + yy
            return "$dd/$mm/$yyyy"
        }
    }

    /**
     * Parses MRZ text and returns structured data.
     *
     * @param mrzText Raw MRZ text (can be multi-line or concatenated)
     * @return Parsed MrzData or null if parsing fails
     */
    fun parse(mrzText: String): MrzData? {
        val normalized = normalizeMrz(mrzText)
        SecureLogger.d(TAG, "Normalized MRZ (${normalized.length} chars)")

        // Detect document type
        val docType = detectDocumentType(normalized)
        SecureLogger.d(TAG, "Detected document type: $docType")

        return when (docType) {
            DocumentType.TD3 -> parseTd3(normalized)
            DocumentType.TD1 -> parseTd1(normalized)
            DocumentType.TD2 -> parseTd2(normalized)
            DocumentType.UNKNOWN -> null
        }
    }

    /**
     * Normalizes MRZ text for parsing.
     */
    private fun normalizeMrz(text: String): String {
        return text.uppercase()
            .replace(" ", "")
            .replace("\r", "")
            .replace("\n", "")
            .replace("«", "<")
            .replace("‹", "<")
            .replace("›", "<")
            .filter { it.isLetterOrDigit() || it == '<' }
    }

    /**
     * Detects document type from MRZ length and content.
     */
    private fun detectDocumentType(mrz: String): DocumentType {
        return when {
            mrz.length == 88 -> DocumentType.TD3      // 2 × 44
            mrz.length == 90 -> DocumentType.TD1      // 3 × 30
            mrz.length in 86..92 && mrz.startsWith("P") -> DocumentType.TD3
            mrz.length in 86..92 && (mrz.startsWith("I") || mrz.startsWith("A") || mrz.startsWith("C")) -> DocumentType.TD1
            mrz.length == 72 -> DocumentType.TD2      // 2 × 36
            else -> {
                // Try to detect by content pattern
                when {
                    mrz.startsWith("P<") || mrz.startsWith("P0") -> DocumentType.TD3
                    mrz.startsWith("I<") || mrz.startsWith("ID") || mrz.startsWith("AC") -> DocumentType.TD1
                    else -> DocumentType.UNKNOWN
                }
            }
        }
    }

    /**
     * Parses TD3 (Passport) MRZ.
     *
     * Line 1 (44 chars): P<ISSSUERNAME<<GIVENNAMES<<<<<<<<<<<<<<<<<<<
     * Line 2 (44 chars): DOCNUMBER#NATYYYMMDD#SYYYMMDD#OPTIONALDATA###C
     *
     * # = check digit, C = composite check digit
     */
    private fun parseTd3(mrz: String): MrzData? {
        if (mrz.length < 88) {
            SecureLogger.w(TAG, "TD3 MRZ too short: ${mrz.length}")
            return null
        }

        return try {
            val line1 = mrz.substring(0, 44)
            val line2 = mrz.substring(44, 88)

            SecureLogger.d(TAG, "TD3 Line 1: ${SecureLogger.maskMrz(line1)}")
            SecureLogger.d(TAG, "TD3 Line 2: ${SecureLogger.maskMrz(line2)}")

            // Line 1 parsing
            val docCode = line1.substring(0, 2).replace("<", "")
            val issuingCountry = line1.substring(2, 5).replace("<", "")
            val namesField = line1.substring(5, 44)
            val (surname, givenNames) = parseNames(namesField)

            // Line 2 parsing
            val documentNumber = line2.substring(0, 9).replace("<", "").trim()
            val docNumCheck = line2[9]
            val nationality = line2.substring(10, 13).replace("<", "")
            val dateOfBirth = line2.substring(13, 19)
            val dobCheck = line2[19]
            val sex = line2.substring(20, 21).replace("<", "X")
            val dateOfExpiry = line2.substring(21, 27)
            val doeCheck = line2[27]
            val personalNumber = line2.substring(28, 42).replace("<", "").trim()
            val personalCheck = line2[42]
            val compositeCheck = line2[43]

            // Validate check digits
            val docNumValid = verifyCheckDigit(line2.substring(0, 9), docNumCheck)
            val dobValid = verifyCheckDigit(dateOfBirth, dobCheck)
            val doeValid = verifyCheckDigit(dateOfExpiry, doeCheck)
            val personalValid = verifyCheckDigit(line2.substring(28, 42), personalCheck)

            // Composite check: docnum+check+dob+check+doe+check+personal+check
            val compositeData = line2.substring(0, 10) + line2.substring(13, 20) +
                               line2.substring(21, 43)
            val compositeValid = verifyCheckDigit(compositeData, compositeCheck)

            val allChecksValid = docNumValid && dobValid && doeValid &&
                                (personalNumber.isEmpty() || personalValid) && compositeValid

            SecureLogger.d(TAG, "TD3 checks - DocNum:$docNumValid DOB:$dobValid DOE:$doeValid Personal:$personalValid Composite:$compositeValid")

            MrzData(
                documentType = DocumentType.TD3,
                documentCode = docCode.ifEmpty { "P" },
                issuingCountry = issuingCountry,
                documentNumber = documentNumber,
                dateOfBirth = dateOfBirth,
                dateOfExpiry = dateOfExpiry,
                sex = sex,
                nationality = nationality,
                surname = surname,
                givenNames = givenNames,
                personalNumber = personalNumber,
                checksumValid = allChecksValid,
                rawMrz = mrz
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse TD3 MRZ", e)
            null
        }
    }

    /**
     * Parses TD1 (ID Card) MRZ.
     *
     * Line 1 (30 chars): IDISSDOCNUMBER#OPTIONAL<<<<<<<
     * Line 2 (30 chars): YYYMMDD#SYYYMMDD#NATOPTION<<<#C
     * Line 3 (30 chars): SURNAME<<GIVENNAMES<<<<<<<<<<<<
     */
    private fun parseTd1(mrz: String): MrzData? {
        if (mrz.length < 90) {
            SecureLogger.w(TAG, "TD1 MRZ too short: ${mrz.length}")
            return null
        }

        return try {
            val line1 = mrz.substring(0, 30)
            val line2 = mrz.substring(30, 60)
            val line3 = mrz.substring(60, 90)

            SecureLogger.d(TAG, "TD1 Line 1: ${SecureLogger.maskMrz(line1)}")
            SecureLogger.d(TAG, "TD1 Line 2: ${SecureLogger.maskMrz(line2)}")
            SecureLogger.d(TAG, "TD1 Line 3: ${SecureLogger.maskMrz(line3)}")

            // Line 1 parsing
            val docCode = line1.substring(0, 2).replace("<", "")
            val issuingCountry = line1.substring(2, 5).replace("<", "")
            val documentNumber = line1.substring(5, 14).replace("<", "").trim()
            val docNumCheck = line1[14]
            val optional1 = line1.substring(15, 30).replace("<", "").trim()

            // Line 2 parsing
            val dateOfBirth = line2.substring(0, 6)
            val dobCheck = line2[6]
            val sex = line2.substring(7, 8).replace("<", "X")
            val dateOfExpiry = line2.substring(8, 14)
            val doeCheck = line2[14]
            val nationality = line2.substring(15, 18).replace("<", "")
            val optional2 = line2.substring(18, 29).replace("<", "").trim()
            val compositeCheck = line2[29]

            // Line 3 parsing (names)
            val (surname, givenNames) = parseNames(line3)

            // Validate check digits
            val docNumValid = verifyCheckDigit(line1.substring(5, 14), docNumCheck)
            val dobValid = verifyCheckDigit(dateOfBirth, dobCheck)
            val doeValid = verifyCheckDigit(dateOfExpiry, doeCheck)

            // Composite check for TD1
            val compositeData = line1.substring(5, 30) + line2.substring(0, 7) +
                               line2.substring(8, 15) + line2.substring(18, 29)
            val compositeValid = verifyCheckDigit(compositeData, compositeCheck)

            val allChecksValid = docNumValid && dobValid && doeValid && compositeValid

            SecureLogger.d(TAG, "TD1 checks - DocNum:$docNumValid DOB:$dobValid DOE:$doeValid Composite:$compositeValid")

            // Extract TCKN from optional fields if present (Turkish specific)
            val personalNumber = extractTckn(optional1) ?: extractTckn(optional2) ?: optional1

            MrzData(
                documentType = DocumentType.TD1,
                documentCode = docCode.ifEmpty { "ID" },
                issuingCountry = issuingCountry,
                documentNumber = documentNumber,
                dateOfBirth = dateOfBirth,
                dateOfExpiry = dateOfExpiry,
                sex = sex,
                nationality = nationality,
                surname = surname,
                givenNames = givenNames,
                personalNumber = personalNumber,
                checksumValid = allChecksValid,
                rawMrz = mrz
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse TD1 MRZ", e)
            null
        }
    }

    /**
     * Parses TD2 (Visa) MRZ - less common format.
     */
    private fun parseTd2(mrz: String): MrzData? {
        // TD2 is 2 × 36 characters - similar to TD3 but shorter
        if (mrz.length < 72) {
            SecureLogger.w(TAG, "TD2 MRZ too short: ${mrz.length}")
            return null
        }

        // TD2 parsing follows TD3 pattern but with different field positions
        // For now, return null as it's not commonly used
        SecureLogger.w(TAG, "TD2 format not fully implemented")
        return null
    }

    /**
     * Parses name field (SURNAME<<GIVENNAMES).
     */
    private fun parseNames(nameField: String): Pair<String, String> {
        val parts = nameField.split("<<")
        val surname = parts.getOrNull(0)?.replace("<", " ")?.trim() ?: ""
        val givenNames = parts.getOrNull(1)?.replace("<", " ")?.trim() ?: ""
        return Pair(surname, givenNames)
    }

    /**
     * Extracts TCKN (11-digit Turkish ID number) from a string.
     */
    private fun extractTckn(data: String): String? {
        val digits = data.filter { it.isDigit() }
        return if (digits.length == 11 && digits.first() != '0') digits else null
    }

    // Check digit calculation constants
    private val WEIGHTS = intArrayOf(7, 3, 1)

    /**
     * Gets numeric value of MRZ character for check digit calculation.
     */
    private fun getCharValue(char: Char): Int {
        return when {
            char.isDigit() -> char.digitToInt()
            char.isLetter() -> char.uppercaseChar() - 'A' + 10
            char == '<' -> 0
            else -> 0
        }
    }

    /**
     * Calculates check digit for given data.
     */
    fun calculateCheckDigit(data: String): Int {
        var sum = 0
        data.forEachIndexed { index, char ->
            val value = getCharValue(char)
            val weight = WEIGHTS[index % 3]
            sum += value * weight
        }
        return sum % 10
    }

    /**
     * Verifies check digit against data.
     */
    fun verifyCheckDigit(data: String, checkDigit: Char): Boolean {
        if (!checkDigit.isDigit() && checkDigit != '<') return false
        val expected = calculateCheckDigit(data)
        val actual = if (checkDigit == '<') 0 else checkDigit.digitToInt()
        return expected == actual
    }

    /**
     * Validates MRZ format without full parsing.
     */
    fun isValidMrzFormat(text: String): Boolean {
        val normalized = normalizeMrz(text)
        return normalized.length in listOf(72, 88, 90) &&
               normalized.all { it.isLetterOrDigit() || it == '<' }
    }

    /**
     * Extracts BAC key material from MRZ.
     *
     * @return Triple of (documentNumber, dateOfBirth, dateOfExpiry) or null
     */
    fun extractBacKeyMaterial(mrz: String): Triple<String, String, String>? {
        val parsed = parse(mrz) ?: return null
        if (parsed.documentNumber.isEmpty() ||
            parsed.dateOfBirth.length != 6 ||
            parsed.dateOfExpiry.length != 6) {
            return null
        }
        return Triple(parsed.documentNumber, parsed.dateOfBirth, parsed.dateOfExpiry)
    }
}
