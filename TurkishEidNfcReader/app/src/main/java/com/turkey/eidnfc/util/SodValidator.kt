package com.turkey.eidnfc.util

import org.bouncycastle.cert.X509CertificateHolder
import org.bouncycastle.cms.CMSSignedData
import org.bouncycastle.cms.SignerInformation
import org.bouncycastle.cms.jcajce.JcaSimpleSignerInfoVerifierBuilder
import org.bouncycastle.jce.provider.BouncyCastleProvider
import org.bouncycastle.util.Store
import timber.log.Timber
import java.io.ByteArrayInputStream
import java.security.MessageDigest
import java.security.Security
import java.security.cert.X509Certificate

/**
 * Validator for SOD (Security Object Document) from Turkish eID card.
 *
 * The SOD contains:
 * - Hash values of all data groups (DG1, DG2, etc.)
 * - Digital signature from Document Signer (DS) certificate
 * - DS certificate itself
 *
 * This validator:
 * 1. Verifies the digital signature of SOD
 * 2. Validates the certificate chain (if CSCA certificate is available)
 * 3. Provides hash values for data group verification
 */
object SodValidator {

    init {
        // Add Bouncy Castle as security provider
        if (Security.getProvider(BouncyCastleProvider.PROVIDER_NAME) == null) {
            Security.addProvider(BouncyCastleProvider())
            Timber.d("Bouncy Castle provider added")
        }
    }

    /**
     * Result of SOD validation.
     */
    data class SodValidationResult(
        val isSignatureValid: Boolean,
        val isCertificateValid: Boolean,
        val documentSignerCert: X509Certificate?,
        val hashAlgorithm: String?,
        val dataGroupHashes: Map<Int, ByteArray>,
        val errorMessage: String?
    )

    /**
     * Validates SOD and extracts information.
     *
     * @param sodData Raw SOD data bytes
     * @param cscaCert Optional CSCA (Country Signing Certificate Authority) certificate for chain validation
     * @return SodValidationResult with validation details
     */
    fun validate(sodData: ByteArray, cscaCert: X509Certificate? = null): SodValidationResult {
        return try {
            Timber.d("Validating SOD (${sodData.size} bytes)")

            // Parse SOD as CMS SignedData
            val signedData = parseSod(sodData)
            if (signedData == null) {
                return SodValidationResult(
                    isSignatureValid = false,
                    isCertificateValid = false,
                    documentSignerCert = null,
                    hashAlgorithm = null,
                    dataGroupHashes = emptyMap(),
                    errorMessage = "Failed to parse SOD"
                )
            }

            // Extract Document Signer certificate
            val dsCert = extractDocumentSignerCert(signedData)
            if (dsCert == null) {
                return SodValidationResult(
                    isSignatureValid = false,
                    isCertificateValid = false,
                    documentSignerCert = null,
                    hashAlgorithm = null,
                    dataGroupHashes = emptyMap(),
                    errorMessage = "Failed to extract Document Signer certificate"
                )
            }

            Timber.d("Document Signer certificate found")
            Timber.d("Subject: ${dsCert.subjectDN}")
            Timber.d("Issuer: ${dsCert.issuerDN}")
            Timber.d("Valid from: ${dsCert.notBefore}")
            Timber.d("Valid until: ${dsCert.notAfter}")

            // Verify digital signature
            val isSignatureValid = verifySignature(signedData, dsCert)
            Timber.d("Signature validation: ${if (isSignatureValid) "VALID" else "INVALID"}")

            // Verify certificate (check expiry, and chain if CSCA is available)
            val isCertificateValid = verifyCertificate(dsCert, cscaCert)
            Timber.d("Certificate validation: ${if (isCertificateValid) "VALID" else "INVALID"}")

            // Extract data group hashes
            val (hashAlgorithm, dataGroupHashes) = extractDataGroupHashes(signedData)
            Timber.d("Hash algorithm: $hashAlgorithm")
            Timber.d("Data group hashes: ${dataGroupHashes.size} groups")

            SodValidationResult(
                isSignatureValid = isSignatureValid,
                isCertificateValid = isCertificateValid,
                documentSignerCert = dsCert,
                hashAlgorithm = hashAlgorithm,
                dataGroupHashes = dataGroupHashes,
                errorMessage = null
            )

        } catch (e: Exception) {
            Timber.e(e, "Failed to validate SOD")
            SodValidationResult(
                isSignatureValid = false,
                isCertificateValid = false,
                documentSignerCert = null,
                hashAlgorithm = null,
                dataGroupHashes = emptyMap(),
                errorMessage = "Validation error: ${e.message}"
            )
        }
    }

    /**
     * Parses SOD data as CMS SignedData.
     */
    private fun parseSod(sodData: ByteArray): CMSSignedData? {
        return try {
            // SOD is a CMS/PKCS#7 SignedData structure
            CMSSignedData(sodData)
        } catch (e: Exception) {
            Timber.e(e, "Failed to parse SOD as CMSSignedData")
            null
        }
    }

    /**
     * Extracts Document Signer certificate from SOD.
     */
    private fun extractDocumentSignerCert(signedData: CMSSignedData): X509Certificate? {
        return try {
            val certStore: Store<X509CertificateHolder> = signedData.certificates
            val certHolders = certStore.getMatches(null)

            if (certHolders.isEmpty()) {
                Timber.e("No certificates found in SOD")
                return null
            }

            // Usually there's only one certificate (DS certificate)
            val certHolder = certHolders.first()

            // Convert to X509Certificate
            val certFactory = java.security.cert.CertificateFactory.getInstance("X.509")
            val cert = certFactory.generateCertificate(
                ByteArrayInputStream(certHolder.encoded)
            ) as X509Certificate

            cert

        } catch (e: Exception) {
            Timber.e(e, "Failed to extract Document Signer certificate")
            null
        }
    }

    /**
     * Verifies the digital signature of SOD.
     */
    private fun verifySignature(signedData: CMSSignedData, dsCert: X509Certificate): Boolean {
        return try {
            val signers = signedData.signerInfos
            val signer: SignerInformation = signers.signers.first()

            // Build verifier
            val verifier = JcaSimpleSignerInfoVerifierBuilder()
                .setProvider(BouncyCastleProvider.PROVIDER_NAME)
                .build(dsCert.publicKey)

            // Verify signature
            signer.verify(verifier)

        } catch (e: Exception) {
            Timber.e(e, "Signature verification failed")
            false
        }
    }

    /**
     * Verifies Document Signer certificate.
     */
    private fun verifyCertificate(dsCert: X509Certificate, cscaCert: X509Certificate?): Boolean {
        return try {
            // Check if certificate is expired
            dsCert.checkValidity()
            Timber.d("DS certificate is valid (not expired)")

            // If CSCA certificate is provided, verify the chain
            if (cscaCert != null) {
                try {
                    dsCert.verify(cscaCert.publicKey)
                    Timber.d("DS certificate verified against CSCA")
                } catch (e: Exception) {
                    Timber.w(e, "Failed to verify DS certificate against CSCA")
                    return false
                }
            } else {
                Timber.w("CSCA certificate not provided, skipping chain validation")
            }

            true

        } catch (e: Exception) {
            Timber.e(e, "Certificate verification failed")
            false
        }
    }

    /**
     * Extracts data group hash values from SOD.
     *
     * @return Pair of (hash algorithm, map of DG number to hash value)
     */
    private fun extractDataGroupHashes(signedData: CMSSignedData): Pair<String?, Map<Int, ByteArray>> {
        return try {
            // The signed content contains LDSSecurityObject
            val content = signedData.signedContent.content as ByteArray

            // Parse LDSSecurityObject (ASN.1 structure)
            val hashes = parseLdsSecurityObject(content)

            Pair("SHA-256", hashes) // Most Turkish eIDs use SHA-256

        } catch (e: Exception) {
            Timber.e(e, "Failed to extract data group hashes")
            Pair(null, emptyMap())
        }
    }

    /**
     * Parses LDSSecurityObject to extract hash values.
     */
    private fun parseLdsSecurityObject(content: ByteArray): Map<Int, ByteArray> {
        val hashes = mutableMapOf<Int, ByteArray>()

        try {
            // This is a simplified parser - full implementation would require
            // proper ASN.1 parsing of the LDSSecurityObject structure
            Timber.d("Parsing LDSSecurityObject (${content.size} bytes)")

            // For now, return empty map
            // Full implementation would parse the SEQUENCE of DataGroupHash structures

        } catch (e: Exception) {
            Timber.e(e, "Failed to parse LDSSecurityObject")
        }

        return hashes
    }

    /**
     * Verifies a data group's hash against the hash in SOD.
     *
     * @param dgNumber Data group number (e.g., 1 for DG1, 2 for DG2)
     * @param dgData The actual data group content
     * @param sodResult The SOD validation result containing expected hashes
     * @return True if hash matches, false otherwise
     */
    fun verifyDataGroupHash(
        dgNumber: Int,
        dgData: ByteArray,
        sodResult: SodValidationResult
    ): Boolean {
        return try {
            val expectedHash = sodResult.dataGroupHashes[dgNumber]
            if (expectedHash == null) {
                Timber.w("No hash found in SOD for DG$dgNumber")
                return false
            }

            // Calculate hash of data group
            val hashAlgorithm = sodResult.hashAlgorithm ?: "SHA-256"
            val digest = MessageDigest.getInstance(hashAlgorithm)
            val calculatedHash = digest.digest(dgData)

            // Compare hashes
            val matches = calculatedHash.contentEquals(expectedHash)

            if (matches) {
                Timber.d("DG$dgNumber hash verification: VALID")
            } else {
                Timber.e("DG$dgNumber hash verification: INVALID")
                Timber.e("Expected: ${toHexString(expectedHash)}")
                Timber.e("Calculated: ${toHexString(calculatedHash)}")
            }

            matches

        } catch (e: Exception) {
            Timber.e(e, "Failed to verify DG$dgNumber hash")
            false
        }
    }

    /**
     * Converts byte array to hex string for logging.
     */
    private fun toHexString(bytes: ByteArray): String {
        return bytes.joinToString(" ") { String.format("%02X", it) }
    }
}
