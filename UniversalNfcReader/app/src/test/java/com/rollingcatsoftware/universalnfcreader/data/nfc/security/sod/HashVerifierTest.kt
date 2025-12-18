package com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod

import com.google.common.truth.Truth.assertThat
import org.junit.Test
import java.security.MessageDigest

/**
 * Unit tests for HashVerifier.
 */
class HashVerifierTest {

    @Test
    fun `computeHash returns correct SHA-256 hash`() {
        val data = "Hello, World!".toByteArray()
        val hash = HashVerifier.computeHash(data, "SHA-256")

        // Pre-computed SHA-256 hash of "Hello, World!"
        val expected = MessageDigest.getInstance("SHA-256").digest(data)
        assertThat(hash).isEqualTo(expected)
    }

    @Test
    fun `computeHash returns correct SHA-1 hash`() {
        val data = "Test".toByteArray()
        val hash = HashVerifier.computeHash(data, "SHA-1")

        val expected = MessageDigest.getInstance("SHA-1").digest(data)
        assertThat(hash).isEqualTo(expected)
        assertThat(hash).hasLength(20)
    }

    @Test
    fun `isAlgorithmSupported returns true for supported algorithms`() {
        assertThat(HashVerifier.isAlgorithmSupported("SHA-256")).isTrue()
        assertThat(HashVerifier.isAlgorithmSupported("SHA-1")).isTrue()
        assertThat(HashVerifier.isAlgorithmSupported("SHA-384")).isTrue()
        assertThat(HashVerifier.isAlgorithmSupported("SHA-512")).isTrue()
        assertThat(HashVerifier.isAlgorithmSupported("MD5")).isTrue()
    }

    @Test
    fun `isAlgorithmSupported returns false for unsupported algorithms`() {
        assertThat(HashVerifier.isAlgorithmSupported("INVALID-ALGO")).isFalse()
        assertThat(HashVerifier.isAlgorithmSupported("SHA-999")).isFalse()
    }

    @Test
    fun `getHashLength returns correct lengths`() {
        assertThat(HashVerifier.getHashLength("SHA-256")).isEqualTo(32)
        assertThat(HashVerifier.getHashLength("SHA-1")).isEqualTo(20)
        assertThat(HashVerifier.getHashLength("SHA-384")).isEqualTo(48)
        assertThat(HashVerifier.getHashLength("SHA-512")).isEqualTo(64)
    }

    @Test
    fun `verifyHash returns valid for matching hash`() {
        val data = "Test data for verification".toByteArray()
        val expectedHash = MessageDigest.getInstance("SHA-256").digest(data)

        val result = HashVerifier.verifyHash(
            dgNumber = 1,
            dgData = data,
            expectedHash = expectedHash,
            algorithm = "SHA-256"
        )

        assertThat(result.isValid).isTrue()
        assertThat(result.dgNumber).isEqualTo(1)
        assertThat(result.algorithm).isEqualTo("SHA-256")
        assertThat(result.errorMessage).isNull()
    }

    @Test
    fun `verifyHash returns invalid for mismatched hash`() {
        val data = "Test data".toByteArray()
        val wrongHash = ByteArray(32) { 0xFF.toByte() }

        val result = HashVerifier.verifyHash(
            dgNumber = 2,
            dgData = data,
            expectedHash = wrongHash,
            algorithm = "SHA-256"
        )

        assertThat(result.isValid).isFalse()
        assertThat(result.dgNumber).isEqualTo(2)
    }

    @Test
    fun `verifyDataGroup returns invalid when hash not in SOD`() {
        val ldsObject = LdsSecurityObjectParser.LdsSecurityObject(
            version = 0,
            hashAlgorithm = "SHA-256",
            hashAlgorithmOid = "2.16.840.1.101.3.4.2.1",
            dataGroupHashes = mapOf(1 to ByteArray(32))
        )

        val result = HashVerifier.verifyDataGroup(
            dgNumber = 2, // Not in SOD
            dgData = ByteArray(100),
            ldsSecurityObject = ldsObject
        )

        assertThat(result.isValid).isFalse()
        assertThat(result.errorMessage).contains("not present")
    }

    @Test
    fun `verifyDataGroups returns correct batch result`() {
        val dg1Data = "DG1 content".toByteArray()
        val dg2Data = "DG2 content".toByteArray()

        val dg1Hash = MessageDigest.getInstance("SHA-256").digest(dg1Data)
        val dg2Hash = MessageDigest.getInstance("SHA-256").digest(dg2Data)

        val ldsObject = LdsSecurityObjectParser.LdsSecurityObject(
            version = 0,
            hashAlgorithm = "SHA-256",
            hashAlgorithmOid = "2.16.840.1.101.3.4.2.1",
            dataGroupHashes = mapOf(1 to dg1Hash, 2 to dg2Hash)
        )

        val result = HashVerifier.verifyDataGroups(
            dataGroups = mapOf(1 to dg1Data, 2 to dg2Data),
            ldsSecurityObject = ldsObject
        )

        assertThat(result.allValid).isTrue()
        assertThat(result.verifiedCount).isEqualTo(2)
        assertThat(result.failedCount).isEqualTo(0)
    }

    @Test
    fun `verifyDataGroups detects failures`() {
        val dg1Data = "DG1 content".toByteArray()
        val dg2Data = "DG2 content".toByteArray()

        val dg1Hash = MessageDigest.getInstance("SHA-256").digest(dg1Data)
        val wrongDg2Hash = ByteArray(32) { 0xFF.toByte() }

        val ldsObject = LdsSecurityObjectParser.LdsSecurityObject(
            version = 0,
            hashAlgorithm = "SHA-256",
            hashAlgorithmOid = "2.16.840.1.101.3.4.2.1",
            dataGroupHashes = mapOf(1 to dg1Hash, 2 to wrongDg2Hash)
        )

        val result = HashVerifier.verifyDataGroups(
            dataGroups = mapOf(1 to dg1Data, 2 to dg2Data),
            ldsSecurityObject = ldsObject
        )

        assertThat(result.allValid).isFalse()
        assertThat(result.verifiedCount).isEqualTo(1)
        assertThat(result.failedCount).isEqualTo(1)
        assertThat(result.results[1]?.isValid).isTrue()
        assertThat(result.results[2]?.isValid).isFalse()
    }

    @Test
    fun `VerificationResult toString contains expected info`() {
        val result = HashVerifier.VerificationResult(
            isValid = true,
            dgNumber = 1,
            algorithm = "SHA-256"
        )

        assertThat(result.toString()).contains("DG1")
        assertThat(result.toString()).contains("VALID")
    }

    @Test
    fun `BatchVerificationResult toString contains expected info`() {
        val result = HashVerifier.BatchVerificationResult(
            results = emptyMap(),
            allValid = true,
            verifiedCount = 2,
            failedCount = 0
        )

        assertThat(result.toString()).contains("2")
        assertThat(result.toString()).contains("0")
    }
}
