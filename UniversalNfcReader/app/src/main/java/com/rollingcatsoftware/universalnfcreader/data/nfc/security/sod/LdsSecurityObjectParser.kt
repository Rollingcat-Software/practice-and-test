package com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod

import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import org.bouncycastle.asn1.ASN1InputStream
import org.bouncycastle.asn1.ASN1Integer
import org.bouncycastle.asn1.ASN1ObjectIdentifier
import org.bouncycastle.asn1.ASN1OctetString
import org.bouncycastle.asn1.ASN1Sequence
import org.bouncycastle.asn1.ASN1Set
import org.bouncycastle.asn1.x509.AlgorithmIdentifier

/**
 * Parser for LDSSecurityObject (ICAO Doc 9303 Part 10).
 *
 * The LDSSecurityObject contains:
 * - Version number
 * - Hash algorithm identifier
 * - Data group hash values
 *
 * ASN.1 Structure:
 * ```
 * LDSSecurityObject ::= SEQUENCE {
 *     version             LDSSecurityObjectVersion,
 *     hashAlgorithm       DigestAlgorithmIdentifier,
 *     dataGroupHashValues SEQUENCE OF DataGroupHash
 * }
 *
 * DataGroupHash ::= SEQUENCE {
 *     dataGroupNumber     DataGroupNumber,
 *     dataGroupHashValue  OCTET STRING
 * }
 *
 * DataGroupNumber ::= INTEGER
 * ```
 *
 * COMPLIANCE: se-checklist.md - Section 4: Cryptographic operations
 */
object LdsSecurityObjectParser {

    private const val TAG = "LdsSecurityObjectParser"

    // Common hash algorithm OIDs
    private val HASH_ALGORITHM_NAMES = mapOf(
        "2.16.840.1.101.3.4.2.1" to "SHA-256",
        "2.16.840.1.101.3.4.2.2" to "SHA-384",
        "2.16.840.1.101.3.4.2.3" to "SHA-512",
        "2.16.840.1.101.3.4.2.4" to "SHA-224",
        "1.3.14.3.2.26" to "SHA-1",
        "1.2.840.113549.2.5" to "MD5"  // Legacy, should not be used
    )

    /**
     * Parsed LDS Security Object data.
     */
    data class LdsSecurityObject(
        val version: Int,
        val hashAlgorithm: String,
        val hashAlgorithmOid: String,
        val dataGroupHashes: Map<Int, ByteArray>
    ) {
        /**
         * Gets the hash for a specific data group.
         *
         * @param dgNumber Data group number (1-16)
         * @return Hash bytes or null if not present
         */
        fun getHash(dgNumber: Int): ByteArray? = dataGroupHashes[dgNumber]

        /**
         * Checks if a data group hash is present.
         */
        fun hasHash(dgNumber: Int): Boolean = dataGroupHashes.containsKey(dgNumber)

        override fun toString(): String {
            val dgList = dataGroupHashes.keys.sorted().joinToString(", ") { "DG$it" }
            return "LdsSecurityObject(version=$version, algorithm=$hashAlgorithm, groups=[$dgList])"
        }
    }

    /**
     * Parses the LDSSecurityObject from raw bytes.
     *
     * @param data Raw LDSSecurityObject bytes (content of SOD signedContent)
     * @return Parsed LdsSecurityObject or null if parsing fails
     */
    fun parse(data: ByteArray): LdsSecurityObject? {
        return try {
            SecureLogger.d(TAG, "Parsing LDSSecurityObject (${data.size} bytes)")

            ASN1InputStream(data).use { asn1Stream ->
                val sequence = asn1Stream.readObject() as? ASN1Sequence
                if (sequence == null) {
                    SecureLogger.e(TAG, "Failed to parse as ASN1Sequence")
                    return null
                }

                parseSequence(sequence)
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse LDSSecurityObject", e)
            null
        }
    }

    /**
     * Parses the main LDSSecurityObject sequence.
     */
    private fun parseSequence(sequence: ASN1Sequence): LdsSecurityObject? {
        return try {
            // LDSSecurityObject has 3 elements: version, hashAlgorithm, dataGroupHashValues
            if (sequence.size() < 3) {
                SecureLogger.e(TAG, "Invalid sequence size: ${sequence.size()}, expected at least 3")
                return null
            }

            var index = 0

            // Version (optional in some implementations, may be 0 or 1)
            val version = (sequence.getObjectAt(index) as? ASN1Integer)?.intValueExact() ?: 0
            index++

            SecureLogger.d(TAG, "LDS version: $version")

            // Hash algorithm identifier
            val algorithmSequence = sequence.getObjectAt(index) as? ASN1Sequence
            if (algorithmSequence == null) {
                SecureLogger.e(TAG, "Failed to parse hash algorithm")
                return null
            }
            index++

            val algorithmIdentifier = AlgorithmIdentifier.getInstance(algorithmSequence)
            val algorithmOid = algorithmIdentifier.algorithm.id
            val algorithmName = HASH_ALGORITHM_NAMES[algorithmOid] ?: algorithmOid

            SecureLogger.d(TAG, "Hash algorithm: $algorithmName ($algorithmOid)")

            // Data group hash values
            val hashSequence = sequence.getObjectAt(index) as? ASN1Sequence
            if (hashSequence == null) {
                SecureLogger.e(TAG, "Failed to parse data group hashes")
                return null
            }

            val dataGroupHashes = parseDataGroupHashes(hashSequence)

            SecureLogger.d(TAG, "Parsed ${dataGroupHashes.size} data group hashes")

            LdsSecurityObject(
                version = version,
                hashAlgorithm = algorithmName,
                hashAlgorithmOid = algorithmOid,
                dataGroupHashes = dataGroupHashes
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Error parsing LDSSecurityObject sequence", e)
            null
        }
    }

    /**
     * Parses the SEQUENCE OF DataGroupHash.
     */
    private fun parseDataGroupHashes(sequence: ASN1Sequence): Map<Int, ByteArray> {
        val hashes = mutableMapOf<Int, ByteArray>()

        for (i in 0 until sequence.size()) {
            val dgHashSequence = sequence.getObjectAt(i) as? ASN1Sequence ?: continue

            try {
                val dgHash = parseDataGroupHash(dgHashSequence)
                if (dgHash != null) {
                    hashes[dgHash.first] = dgHash.second
                    SecureLogger.d(TAG, "DG${dgHash.first} hash: ${dgHash.second.size} bytes")
                }
            } catch (e: Exception) {
                SecureLogger.w(TAG, "Failed to parse data group hash at index $i", e)
            }
        }

        return hashes
    }

    /**
     * Parses a single DataGroupHash structure.
     *
     * @return Pair of (DG number, hash bytes) or null if parsing fails
     */
    private fun parseDataGroupHash(sequence: ASN1Sequence): Pair<Int, ByteArray>? {
        if (sequence.size() < 2) {
            return null
        }

        // DataGroupNumber (INTEGER)
        val dgNumber = (sequence.getObjectAt(0) as? ASN1Integer)?.intValueExact() ?: return null

        // DataGroupHashValue (OCTET STRING)
        val hashValue = (sequence.getObjectAt(1) as? ASN1OctetString)?.octets ?: return null

        return Pair(dgNumber, hashValue)
    }

    /**
     * Determines the hash length in bytes for a given algorithm.
     */
    fun getHashLength(algorithm: String): Int {
        return when (algorithm.uppercase()) {
            "SHA-1" -> 20
            "SHA-224" -> 28
            "SHA-256" -> 32
            "SHA-384" -> 48
            "SHA-512" -> 64
            "MD5" -> 16
            else -> 32 // Default to SHA-256 length
        }
    }

    /**
     * Gets the Java Security algorithm name for hash computation.
     */
    fun getJavaAlgorithmName(algorithm: String): String {
        return when (algorithm.uppercase()) {
            "SHA-1" -> "SHA-1"
            "SHA-224" -> "SHA-224"
            "SHA-256" -> "SHA-256"
            "SHA-384" -> "SHA-384"
            "SHA-512" -> "SHA-512"
            "MD5" -> "MD5"
            else -> algorithm
        }
    }
}
