package com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod

import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import org.bouncycastle.cert.X509CertificateHolder
import org.bouncycastle.cms.CMSSignedData
import org.bouncycastle.cms.SignerInformation
import org.bouncycastle.cms.jcajce.JcaSimpleSignerInfoVerifierBuilder
import org.bouncycastle.jce.provider.BouncyCastleProvider
import org.bouncycastle.util.Store
import java.io.ByteArrayInputStream
import java.security.Security
import java.security.cert.X509Certificate

/**
 * Validator for SOD (Security Object Document) from ICAO MRTD documents.
 *
 * The SOD contains:
 * - Hash values of all data groups (DG1, DG2, etc.) in LDSSecurityObject
 * - Digital signature from Document Signer (DS) certificate
 * - DS certificate itself
 *
 * This validator performs:
 * 1. Parsing of CMS SignedData structure
 * 2. Extraction and validation of DS certificate
 * 3. Verification of digital signature
 * 4. Extraction of data group hashes for verification
 * 5. Optional CSCA chain validation
 *
 * COMPLIANCE: se-checklist.md
 * - Section 4: Cryptographic operations
 * - Section 5.2: Secure error handling
 *
 * Security Notes:
 * - SOD validation ensures document authenticity
 * - Full security requires CSCA certificate chain validation
 * - DG hash verification ensures data integrity
 */
class SodValidator(
    private val cscaStore: CscaCertificateStore = CscaCertificateStore.getInstance()
) {
    companion object {
        private const val TAG = "SodValidator"

        init {
            // Ensure BouncyCastle is available
            if (Security.getProvider(BouncyCastleProvider.PROVIDER_NAME) == null) {
                Security.addProvider(BouncyCastleProvider())
            }
        }
    }

    /**
     * Result of SOD validation.
     */
    data class SodValidationResult(
        val isSignatureValid: Boolean,
        val isCertificateValid: Boolean,
        val isCscaChainValid: Boolean,
        val documentSignerCert: X509Certificate?,
        val cscaCert: X509Certificate?,
        val ldsSecurityObject: LdsSecurityObjectParser.LdsSecurityObject?,
        val errorMessage: String?
    ) {
        /**
         * Overall validation status.
         * True only if signature is valid and certificate is valid.
         * CSCA chain validation is optional but recommended.
         */
        val isValid: Boolean
            get() = isSignatureValid && isCertificateValid

        /**
         * Full validation status including CSCA chain.
         */
        val isFullyValid: Boolean
            get() = isSignatureValid && isCertificateValid && isCscaChainValid

        override fun toString(): String {
            return buildString {
                append("SodValidationResult(")
                append("sig=${if (isSignatureValid) "OK" else "FAIL"}, ")
                append("cert=${if (isCertificateValid) "OK" else "FAIL"}, ")
                append("csca=${if (isCscaChainValid) "OK" else "N/A"}")
                if (!isValid && errorMessage != null) {
                    append(", error=$errorMessage")
                }
                append(")")
            }
        }
    }

    /**
     * Validates SOD and extracts information.
     *
     * @param sodData Raw SOD data bytes (EF.SOD content)
     * @param countryCode Optional country code hint for CSCA lookup
     * @return SodValidationResult with validation details
     */
    fun validate(sodData: ByteArray, countryCode: String? = null): SodValidationResult {
        return try {
            SecureLogger.d(TAG, "Validating SOD (${sodData.size} bytes)")

            // Parse SOD as CMS SignedData
            val signedData = parseSod(sodData)
            if (signedData == null) {
                return SodValidationResult(
                    isSignatureValid = false,
                    isCertificateValid = false,
                    isCscaChainValid = false,
                    documentSignerCert = null,
                    cscaCert = null,
                    ldsSecurityObject = null,
                    errorMessage = "Failed to parse SOD as CMS SignedData"
                )
            }

            // Extract Document Signer certificate
            val dsCert = extractDocumentSignerCert(signedData)
            if (dsCert == null) {
                return SodValidationResult(
                    isSignatureValid = false,
                    isCertificateValid = false,
                    isCscaChainValid = false,
                    documentSignerCert = null,
                    cscaCert = null,
                    ldsSecurityObject = null,
                    errorMessage = "Failed to extract Document Signer certificate"
                )
            }

            SecureLogger.d(TAG, "DS certificate subject: ${SecureLogger.maskName(dsCert.subjectDN.name)}")

            // Verify digital signature
            val isSignatureValid = verifySignature(signedData, dsCert)
            SecureLogger.d(TAG, "Signature validation: ${if (isSignatureValid) "VALID" else "INVALID"}")

            // Verify DS certificate validity (expiration)
            val isCertificateValid = verifyCertificateValidity(dsCert)
            SecureLogger.d(TAG, "Certificate validity: ${if (isCertificateValid) "VALID" else "INVALID"}")

            // Validate CSCA chain
            val cscaValidation = cscaStore.validateCertificateChain(dsCert, countryCode)
            val isCscaChainValid = cscaValidation.isValid
            val cscaCert = cscaValidation.trustAnchor
            SecureLogger.d(TAG, "CSCA chain: ${if (isCscaChainValid) "VALID" else "NOT VALIDATED"}")

            // Extract LDSSecurityObject with data group hashes
            val ldsSecurityObject = extractLdsSecurityObject(signedData)
            if (ldsSecurityObject != null) {
                SecureLogger.d(TAG, "LDS: $ldsSecurityObject")
            } else {
                SecureLogger.w(TAG, "Failed to extract LDSSecurityObject")
            }

            SodValidationResult(
                isSignatureValid = isSignatureValid,
                isCertificateValid = isCertificateValid,
                isCscaChainValid = isCscaChainValid,
                documentSignerCert = dsCert,
                cscaCert = cscaCert,
                ldsSecurityObject = ldsSecurityObject,
                errorMessage = if (!isSignatureValid) "Signature verification failed"
                else if (!isCertificateValid) "Certificate validation failed"
                else null
            )

        } catch (e: Exception) {
            SecureLogger.e(TAG, "SOD validation error", e)
            SodValidationResult(
                isSignatureValid = false,
                isCertificateValid = false,
                isCscaChainValid = false,
                documentSignerCert = null,
                cscaCert = null,
                ldsSecurityObject = null,
                errorMessage = "Validation error: ${e.message}"
            )
        }
    }

    /**
     * Parses SOD data as CMS SignedData.
     */
    private fun parseSod(sodData: ByteArray): CMSSignedData? {
        return try {
            // SOD may have ASN.1 wrapper (tag 77) - skip if present
            val contentData = if (sodData.isNotEmpty() && sodData[0] == 0x77.toByte()) {
                skipAsn1Wrapper(sodData)
            } else {
                sodData
            }

            CMSSignedData(contentData)
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse SOD as CMSSignedData", e)
            null
        }
    }

    /**
     * Skips ASN.1 wrapper (tag 0x77) used in some SOD implementations.
     */
    private fun skipAsn1Wrapper(data: ByteArray): ByteArray {
        return try {
            if (data.size < 2) return data

            var offset = 1 // Skip tag 0x77

            // Parse length
            val lengthByte = data[offset].toInt() and 0xFF
            offset++

            when {
                lengthByte < 0x80 -> {
                    // Short form length
                    data.copyOfRange(offset, offset + lengthByte)
                }
                lengthByte == 0x81 -> {
                    // Long form, 1 byte length
                    val length = data[offset].toInt() and 0xFF
                    offset++
                    data.copyOfRange(offset, offset + length)
                }
                lengthByte == 0x82 -> {
                    // Long form, 2 byte length
                    val length = ((data[offset].toInt() and 0xFF) shl 8) or
                            (data[offset + 1].toInt() and 0xFF)
                    offset += 2
                    data.copyOfRange(offset, offset + length)
                }
                else -> data
            }
        } catch (e: Exception) {
            SecureLogger.w(TAG, "Failed to skip ASN.1 wrapper, using original data")
            data
        }
    }

    /**
     * Extracts Document Signer certificate from SOD.
     */
    private fun extractDocumentSignerCert(signedData: CMSSignedData): X509Certificate? {
        return try {
            @Suppress("UNCHECKED_CAST")
            val certStore: Store<X509CertificateHolder> = signedData.certificates as Store<X509CertificateHolder>
            val certHolders = certStore.getMatches(null)

            if (certHolders.isEmpty()) {
                SecureLogger.e(TAG, "No certificates found in SOD")
                return null
            }

            // Usually there's only one certificate (DS certificate)
            val certHolder = certHolders.first()

            // Convert to X509Certificate
            val certFactory = java.security.cert.CertificateFactory.getInstance("X.509")
            certFactory.generateCertificate(
                ByteArrayInputStream(certHolder.encoded)
            ) as X509Certificate

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to extract Document Signer certificate", e)
            null
        }
    }

    /**
     * Verifies the digital signature of SOD.
     */
    private fun verifySignature(signedData: CMSSignedData, dsCert: X509Certificate): Boolean {
        return try {
            val signers = signedData.signerInfos
            if (signers.signers.isEmpty()) {
                SecureLogger.e(TAG, "No signers found in SOD")
                return false
            }

            val signer: SignerInformation = signers.signers.first()

            // Build verifier using DS certificate public key
            val verifier = JcaSimpleSignerInfoVerifierBuilder()
                .setProvider(BouncyCastleProvider.PROVIDER_NAME)
                .build(dsCert.publicKey)

            // Verify signature
            signer.verify(verifier)

        } catch (e: Exception) {
            SecureLogger.e(TAG, "Signature verification failed", e)
            false
        }
    }

    /**
     * Verifies DS certificate validity (not expired).
     */
    private fun verifyCertificateValidity(dsCert: X509Certificate): Boolean {
        return try {
            dsCert.checkValidity()
            true
        } catch (e: java.security.cert.CertificateExpiredException) {
            SecureLogger.w(TAG, "DS certificate has expired")
            false
        } catch (e: java.security.cert.CertificateNotYetValidException) {
            SecureLogger.w(TAG, "DS certificate is not yet valid")
            false
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Certificate validity check failed", e)
            false
        }
    }

    /**
     * Extracts LDSSecurityObject from SOD signed content.
     */
    private fun extractLdsSecurityObject(signedData: CMSSignedData): LdsSecurityObjectParser.LdsSecurityObject? {
        return try {
            val content = signedData.signedContent?.content as? ByteArray
            if (content == null) {
                SecureLogger.e(TAG, "No signed content in SOD")
                return null
            }

            LdsSecurityObjectParser.parse(content)
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to extract LDSSecurityObject", e)
            null
        }
    }

    /**
     * Gets the hash algorithm from SOD.
     *
     * @param sodData Raw SOD data
     * @return Algorithm name or null if extraction fails
     */
    fun getHashAlgorithm(sodData: ByteArray): String? {
        val result = validate(sodData)
        return result.ldsSecurityObject?.hashAlgorithm
    }

    /**
     * Gets expected hash for a data group from SOD.
     *
     * @param sodData Raw SOD data
     * @param dgNumber Data group number (1-16)
     * @return Hash bytes or null if not found
     */
    fun getDataGroupHash(sodData: ByteArray, dgNumber: Int): ByteArray? {
        val result = validate(sodData)
        return result.ldsSecurityObject?.getHash(dgNumber)
    }
}
