package com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod

import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.secureClear
import java.security.MessageDigest

/**
 * Verifies data group hashes against the SOD (Security Object Document).
 *
 * This class computes hashes of data group content and compares them with
 * the hashes stored in the LDSSecurityObject of the SOD.
 *
 * COMPLIANCE: se-checklist.md
 * - Section 4: Cryptographic operations (constant-time comparison)
 * - Section 1.1: Secure memory handling (clear intermediate data)
 *
 * Security Notes:
 * - Hash comparison uses constant-time algorithm to prevent timing attacks
 * - Intermediate hash values are cleared after use
 * - Supports SHA-1, SHA-224, SHA-256, SHA-384, SHA-512
 */
object HashVerifier {

    private const val TAG = "HashVerifier"

    /**
     * Result of data group hash verification.
     */
    data class VerificationResult(
        val isValid: Boolean,
        val dgNumber: Int,
        val algorithm: String,
        val errorMessage: String? = null
    ) {
        override fun toString(): String {
            return "DG$dgNumber: ${if (isValid) "VALID" else "INVALID"}${errorMessage?.let { " ($it)" } ?: ""}"
        }
    }

    /**
     * Result of verifying multiple data groups.
     */
    data class BatchVerificationResult(
        val results: Map<Int, VerificationResult>,
        val allValid: Boolean,
        val verifiedCount: Int,
        val failedCount: Int
    ) {
        override fun toString(): String {
            return "Verified: $verifiedCount, Failed: $failedCount, All valid: $allValid"
        }
    }

    /**
     * Verifies a single data group's hash against SOD.
     *
     * @param dgNumber Data group number (1-16)
     * @param dgData Raw data group content
     * @param ldsSecurityObject Parsed LDS security object from SOD
     * @return VerificationResult indicating success or failure
     */
    fun verifyDataGroup(
        dgNumber: Int,
        dgData: ByteArray,
        ldsSecurityObject: LdsSecurityObjectParser.LdsSecurityObject
    ): VerificationResult {
        val algorithm = ldsSecurityObject.hashAlgorithm

        // Get expected hash from SOD
        val expectedHash = ldsSecurityObject.getHash(dgNumber)
        if (expectedHash == null) {
            SecureLogger.w(TAG, "No hash found in SOD for DG$dgNumber")
            return VerificationResult(
                isValid = false,
                dgNumber = dgNumber,
                algorithm = algorithm,
                errorMessage = "Hash not present in SOD"
            )
        }

        return verifyHash(dgNumber, dgData, expectedHash, algorithm)
    }

    /**
     * Verifies a data group's hash with explicit expected hash and algorithm.
     *
     * @param dgNumber Data group number
     * @param dgData Raw data group content
     * @param expectedHash Expected hash from SOD
     * @param algorithm Hash algorithm name (e.g., "SHA-256")
     * @return VerificationResult
     */
    fun verifyHash(
        dgNumber: Int,
        dgData: ByteArray,
        expectedHash: ByteArray,
        algorithm: String
    ): VerificationResult {
        var calculatedHash: ByteArray? = null

        return try {
            SecureLogger.d(TAG, "Verifying DG$dgNumber hash ($algorithm, ${dgData.size} bytes)")

            // Calculate hash of data group
            val javaAlgorithm = LdsSecurityObjectParser.getJavaAlgorithmName(algorithm)
            val digest = MessageDigest.getInstance(javaAlgorithm)
            calculatedHash = digest.digest(dgData)

            // Compare using constant-time comparison
            val isValid = constantTimeEquals(calculatedHash, expectedHash)

            if (isValid) {
                SecureLogger.d(TAG, "DG$dgNumber hash verification: VALID")
            } else {
                SecureLogger.w(TAG, "DG$dgNumber hash verification: INVALID")
                SecureLogger.logHex(TAG, "Expected", expectedHash)
                SecureLogger.logHex(TAG, "Calculated", calculatedHash)
            }

            VerificationResult(
                isValid = isValid,
                dgNumber = dgNumber,
                algorithm = algorithm
            )

        } catch (e: Exception) {
            SecureLogger.e(TAG, "DG$dgNumber hash verification error", e)
            VerificationResult(
                isValid = false,
                dgNumber = dgNumber,
                algorithm = algorithm,
                errorMessage = "Verification error: ${e.message}"
            )
        } finally {
            // Clear calculated hash from memory
            calculatedHash?.secureClear()
        }
    }

    /**
     * Verifies multiple data groups against SOD.
     *
     * @param dataGroups Map of DG number to raw DG content
     * @param ldsSecurityObject Parsed LDS security object from SOD
     * @return BatchVerificationResult with all results
     */
    fun verifyDataGroups(
        dataGroups: Map<Int, ByteArray>,
        ldsSecurityObject: LdsSecurityObjectParser.LdsSecurityObject
    ): BatchVerificationResult {
        val results = mutableMapOf<Int, VerificationResult>()
        var verifiedCount = 0
        var failedCount = 0

        for ((dgNumber, dgData) in dataGroups) {
            val result = verifyDataGroup(dgNumber, dgData, ldsSecurityObject)
            results[dgNumber] = result

            if (result.isValid) {
                verifiedCount++
            } else {
                failedCount++
            }
        }

        val allValid = failedCount == 0 && verifiedCount > 0

        SecureLogger.d(TAG, "Batch verification: $verifiedCount valid, $failedCount failed")

        return BatchVerificationResult(
            results = results,
            allValid = allValid,
            verifiedCount = verifiedCount,
            failedCount = failedCount
        )
    }

    /**
     * Verifies DG1 and DG2 (most common verification scenario).
     *
     * @param dg1Data DG1 (MRZ) raw content
     * @param dg2Data DG2 (Photo) raw content
     * @param ldsSecurityObject Parsed LDS security object from SOD
     * @return BatchVerificationResult
     */
    fun verifyDg1AndDg2(
        dg1Data: ByteArray,
        dg2Data: ByteArray,
        ldsSecurityObject: LdsSecurityObjectParser.LdsSecurityObject
    ): BatchVerificationResult {
        return verifyDataGroups(
            mapOf(1 to dg1Data, 2 to dg2Data),
            ldsSecurityObject
        )
    }

    /**
     * Computes hash of data using specified algorithm.
     *
     * @param data Data to hash
     * @param algorithm Hash algorithm name
     * @return Hash bytes
     */
    fun computeHash(data: ByteArray, algorithm: String): ByteArray {
        val javaAlgorithm = LdsSecurityObjectParser.getJavaAlgorithmName(algorithm)
        val digest = MessageDigest.getInstance(javaAlgorithm)
        return digest.digest(data)
    }

    /**
     * Constant-time byte array comparison to prevent timing attacks.
     *
     * COMPLIANCE: se-checklist.md - Security consideration for cryptographic operations
     *
     * This implementation ensures that the comparison time is constant regardless
     * of where the arrays differ, preventing timing-based side-channel attacks.
     *
     * @param a First byte array
     * @param b Second byte array
     * @return True if arrays are equal, false otherwise
     */
    private fun constantTimeEquals(a: ByteArray, b: ByteArray): Boolean {
        if (a.size != b.size) {
            // Still perform comparison to prevent timing leak on length
            var result = a.size xor b.size
            val minLen = minOf(a.size, b.size)
            for (i in 0 until minLen) {
                result = result or (a[i].toInt() xor b[i].toInt())
            }
            return false
        }

        var result = 0
        for (i in a.indices) {
            result = result or (a[i].toInt() xor b[i].toInt())
        }
        return result == 0
    }

    /**
     * Checks if a hash algorithm is supported.
     */
    fun isAlgorithmSupported(algorithm: String): Boolean {
        return try {
            val javaAlgorithm = LdsSecurityObjectParser.getJavaAlgorithmName(algorithm)
            MessageDigest.getInstance(javaAlgorithm)
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Gets the expected hash length for an algorithm.
     */
    fun getHashLength(algorithm: String): Int {
        return LdsSecurityObjectParser.getHashLength(algorithm)
    }
}
