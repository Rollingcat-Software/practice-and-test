package com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod

import android.content.Context
import com.rollingcatsoftware.universalnfcreader.data.nfc.security.SecureLogger
import org.bouncycastle.jce.provider.BouncyCastleProvider
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.security.KeyStore
import java.security.Security
import java.security.cert.CertificateFactory
import java.security.cert.PKIXParameters
import java.security.cert.TrustAnchor
import java.security.cert.X509Certificate
import java.util.concurrent.ConcurrentHashMap
import javax.security.auth.x500.X500Principal

/**
 * Stores and manages CSCA (Country Signing Certificate Authority) certificates.
 *
 * CSCA certificates are root certificates issued by countries to sign Document Signer (DS)
 * certificates, which in turn sign the SOD (Security Object Document) in passports and ID cards.
 *
 * Certificate Chain:
 * CSCA (Root) -> DS Certificate -> SOD Signature
 *
 * This store provides:
 * 1. Lookup of CSCA certificates by country code
 * 2. Trust anchor management for certificate chain validation
 * 3. Support for bundled and user-imported certificates
 *
 * COMPLIANCE: se-checklist.md - Section 4: Cryptographic operations
 *
 * Security Notes:
 * - CSCA certificates must be obtained from official sources
 * - Some countries don't publish their CSCA certificates publicly
 * - The ICAO PKD (Public Key Directory) is the authoritative source
 */
class CscaCertificateStore {

    companion object {
        private const val TAG = "CscaCertificateStore"

        // Singleton instance
        @Volatile
        private var instance: CscaCertificateStore? = null

        fun getInstance(): CscaCertificateStore {
            return instance ?: synchronized(this) {
                instance ?: CscaCertificateStore().also { instance = it }
            }
        }

        init {
            // Ensure BouncyCastle is available
            if (Security.getProvider(BouncyCastleProvider.PROVIDER_NAME) == null) {
                Security.addProvider(BouncyCastleProvider())
            }
        }
    }

    // Country code (ISO 3166-1 alpha-3) -> List of CSCA certificates
    private val cscaCertificates = ConcurrentHashMap<String, MutableList<X509Certificate>>()

    // Certificate factory for parsing
    private val certificateFactory = CertificateFactory.getInstance("X.509")

    /**
     * Result of certificate chain validation.
     */
    data class ValidationResult(
        val isValid: Boolean,
        val trustAnchor: X509Certificate? = null,
        val errorMessage: String? = null
    )

    /**
     * Adds a CSCA certificate for a specific country.
     *
     * @param countryCode ISO 3166-1 alpha-3 country code (e.g., "TUR", "DEU", "USA")
     * @param certificate The CSCA certificate
     */
    fun addCertificate(countryCode: String, certificate: X509Certificate) {
        val normalizedCode = countryCode.uppercase()
        cscaCertificates.getOrPut(normalizedCode) { mutableListOf() }.add(certificate)
        SecureLogger.d(TAG, "Added CSCA certificate for $normalizedCode: ${certificate.subjectDN}")
    }

    /**
     * Adds a CSCA certificate from PEM-encoded bytes.
     *
     * @param countryCode ISO 3166-1 alpha-3 country code
     * @param pemData PEM-encoded certificate data
     * @return True if certificate was added successfully
     */
    fun addCertificateFromPem(countryCode: String, pemData: ByteArray): Boolean {
        return try {
            val certificate = certificateFactory.generateCertificate(
                ByteArrayInputStream(pemData)
            ) as X509Certificate
            addCertificate(countryCode, certificate)
            true
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse PEM certificate for $countryCode", e)
            false
        }
    }

    /**
     * Adds a CSCA certificate from DER-encoded bytes.
     *
     * @param countryCode ISO 3166-1 alpha-3 country code
     * @param derData DER-encoded certificate data
     * @return True if certificate was added successfully
     */
    fun addCertificateFromDer(countryCode: String, derData: ByteArray): Boolean {
        return try {
            val certificate = certificateFactory.generateCertificate(
                ByteArrayInputStream(derData)
            ) as X509Certificate
            addCertificate(countryCode, certificate)
            true
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to parse DER certificate for $countryCode", e)
            false
        }
    }

    /**
     * Loads CSCA certificates from an asset file.
     *
     * @param context Android context
     * @param assetPath Path to the certificate file in assets
     * @param countryCode ISO 3166-1 alpha-3 country code
     * @return True if certificate was loaded successfully
     */
    fun loadFromAsset(context: Context, assetPath: String, countryCode: String): Boolean {
        return try {
            context.assets.open(assetPath).use { inputStream ->
                val certificate = certificateFactory.generateCertificate(inputStream) as X509Certificate
                addCertificate(countryCode, certificate)
                true
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to load certificate from asset: $assetPath", e)
            false
        }
    }

    /**
     * Loads multiple CSCA certificates from a PKCS#7 or certificate chain file.
     *
     * @param inputStream Input stream containing certificates
     * @param countryCode ISO 3166-1 alpha-3 country code
     * @return Number of certificates loaded
     */
    fun loadCertificateChain(inputStream: InputStream, countryCode: String): Int {
        return try {
            val certificates = certificateFactory.generateCertificates(inputStream)
            var count = 0
            for (cert in certificates) {
                if (cert is X509Certificate) {
                    addCertificate(countryCode, cert)
                    count++
                }
            }
            SecureLogger.d(TAG, "Loaded $count certificates for $countryCode")
            count
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to load certificate chain for $countryCode", e)
            0
        }
    }

    /**
     * Gets all CSCA certificates for a country.
     *
     * @param countryCode ISO 3166-1 alpha-3 country code
     * @return List of certificates or empty list if none found
     */
    fun getCertificates(countryCode: String): List<X509Certificate> {
        return cscaCertificates[countryCode.uppercase()]?.toList() ?: emptyList()
    }

    /**
     * Finds a CSCA certificate that can verify the given DS certificate.
     *
     * @param dsCertificate Document Signer certificate to verify
     * @param countryCode Optional country code hint (extracted from DS cert issuer if not provided)
     * @return The matching CSCA certificate or null if not found
     */
    fun findVerifyingCsca(
        dsCertificate: X509Certificate,
        countryCode: String? = null
    ): X509Certificate? {
        val issuer = dsCertificate.issuerX500Principal

        // Try country code if provided
        if (countryCode != null) {
            val csca = findCscaForIssuer(countryCode, issuer, dsCertificate)
            if (csca != null) return csca
        }

        // Try to extract country from DS certificate issuer
        val extractedCountry = extractCountryFromDN(issuer)
        if (extractedCountry != null && extractedCountry != countryCode) {
            val csca = findCscaForIssuer(extractedCountry, issuer, dsCertificate)
            if (csca != null) return csca
        }

        // Search all countries as last resort
        for ((code, certs) in cscaCertificates) {
            for (csca in certs) {
                if (canVerify(csca, dsCertificate)) {
                    SecureLogger.d(TAG, "Found CSCA in country $code")
                    return csca
                }
            }
        }

        SecureLogger.w(TAG, "No matching CSCA found for DS certificate issuer: $issuer")
        return null
    }

    /**
     * Finds CSCA certificate for a specific country and issuer.
     */
    private fun findCscaForIssuer(
        countryCode: String,
        issuer: X500Principal,
        dsCertificate: X509Certificate
    ): X509Certificate? {
        val certs = getCertificates(countryCode)
        for (csca in certs) {
            if (canVerify(csca, dsCertificate)) {
                return csca
            }
        }
        return null
    }

    /**
     * Checks if a CSCA certificate can verify a DS certificate.
     */
    private fun canVerify(csca: X509Certificate, dsCertificate: X509Certificate): Boolean {
        return try {
            // Check if CSCA subject matches DS issuer
            if (csca.subjectX500Principal != dsCertificate.issuerX500Principal) {
                return false
            }

            // Verify signature
            dsCertificate.verify(csca.publicKey)
            true
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Validates a Document Signer certificate chain against stored CSCA certificates.
     *
     * @param dsCertificate Document Signer certificate to validate
     * @param countryCode Optional country code hint
     * @return ValidationResult indicating success or failure
     */
    fun validateCertificateChain(
        dsCertificate: X509Certificate,
        countryCode: String? = null
    ): ValidationResult {
        return try {
            // Check DS certificate validity (not expired)
            dsCertificate.checkValidity()

            // Find matching CSCA
            val csca = findVerifyingCsca(dsCertificate, countryCode)
            if (csca == null) {
                return ValidationResult(
                    isValid = false,
                    errorMessage = "No matching CSCA certificate found"
                )
            }

            // Check CSCA validity
            csca.checkValidity()

            // Verify the chain
            dsCertificate.verify(csca.publicKey)

            SecureLogger.d(TAG, "Certificate chain validation successful")
            ValidationResult(
                isValid = true,
                trustAnchor = csca
            )

        } catch (e: java.security.cert.CertificateExpiredException) {
            SecureLogger.w(TAG, "Certificate expired", e)
            ValidationResult(
                isValid = false,
                errorMessage = "Certificate has expired"
            )
        } catch (e: java.security.cert.CertificateNotYetValidException) {
            SecureLogger.w(TAG, "Certificate not yet valid", e)
            ValidationResult(
                isValid = false,
                errorMessage = "Certificate is not yet valid"
            )
        } catch (e: java.security.SignatureException) {
            SecureLogger.w(TAG, "Signature verification failed", e)
            ValidationResult(
                isValid = false,
                errorMessage = "Certificate signature verification failed"
            )
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Certificate validation error", e)
            ValidationResult(
                isValid = false,
                errorMessage = "Validation error: ${e.message}"
            )
        }
    }

    /**
     * Creates PKIXParameters for certificate path validation.
     *
     * @param countryCode Optional country code to limit trust anchors
     * @return PKIXParameters or null if no trust anchors available
     */
    fun createPKIXParameters(countryCode: String? = null): PKIXParameters? {
        val trustAnchors = mutableSetOf<TrustAnchor>()

        val certsToUse = if (countryCode != null) {
            getCertificates(countryCode)
        } else {
            cscaCertificates.values.flatten()
        }

        for (cert in certsToUse) {
            trustAnchors.add(TrustAnchor(cert, null))
        }

        if (trustAnchors.isEmpty()) {
            SecureLogger.w(TAG, "No trust anchors available")
            return null
        }

        return try {
            PKIXParameters(trustAnchors).apply {
                isRevocationEnabled = false // CRL checking disabled for now
            }
        } catch (e: Exception) {
            SecureLogger.e(TAG, "Failed to create PKIXParameters", e)
            null
        }
    }

    /**
     * Extracts country code from X.500 Distinguished Name.
     */
    private fun extractCountryFromDN(dn: X500Principal): String? {
        return try {
            val name = dn.name
            // Look for C= (country) field
            val regex = Regex("""C=([A-Z]{2,3})""")
            val match = regex.find(name)
            match?.groupValues?.getOrNull(1)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Gets list of all registered country codes.
     */
    fun getRegisteredCountries(): Set<String> {
        return cscaCertificates.keys.toSet()
    }

    /**
     * Gets total number of stored certificates.
     */
    fun getCertificateCount(): Int {
        return cscaCertificates.values.sumOf { it.size }
    }

    /**
     * Checks if any CSCA certificates are available for a country.
     */
    fun hasCertificatesFor(countryCode: String): Boolean {
        return cscaCertificates[countryCode.uppercase()]?.isNotEmpty() == true
    }

    /**
     * Clears all stored certificates.
     */
    fun clear() {
        cscaCertificates.clear()
        SecureLogger.d(TAG, "All CSCA certificates cleared")
    }

    /**
     * Removes certificates for a specific country.
     */
    fun removeCertificates(countryCode: String) {
        cscaCertificates.remove(countryCode.uppercase())
        SecureLogger.d(TAG, "Removed CSCA certificates for $countryCode")
    }
}
