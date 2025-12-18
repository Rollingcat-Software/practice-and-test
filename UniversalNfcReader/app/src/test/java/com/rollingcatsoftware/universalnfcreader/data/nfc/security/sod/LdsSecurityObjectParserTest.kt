package com.rollingcatsoftware.universalnfcreader.data.nfc.security.sod

import com.google.common.truth.Truth.assertThat
import org.junit.Test

/**
 * Unit tests for LdsSecurityObjectParser.
 */
class LdsSecurityObjectParserTest {

    @Test
    fun `getHashLength returns correct length for SHA-256`() {
        assertThat(LdsSecurityObjectParser.getHashLength("SHA-256")).isEqualTo(32)
    }

    @Test
    fun `getHashLength returns correct length for SHA-1`() {
        assertThat(LdsSecurityObjectParser.getHashLength("SHA-1")).isEqualTo(20)
    }

    @Test
    fun `getHashLength returns correct length for SHA-384`() {
        assertThat(LdsSecurityObjectParser.getHashLength("SHA-384")).isEqualTo(48)
    }

    @Test
    fun `getHashLength returns correct length for SHA-512`() {
        assertThat(LdsSecurityObjectParser.getHashLength("SHA-512")).isEqualTo(64)
    }

    @Test
    fun `getJavaAlgorithmName returns correct names`() {
        assertThat(LdsSecurityObjectParser.getJavaAlgorithmName("SHA-256")).isEqualTo("SHA-256")
        assertThat(LdsSecurityObjectParser.getJavaAlgorithmName("sha-256")).isEqualTo("SHA-256")
        assertThat(LdsSecurityObjectParser.getJavaAlgorithmName("SHA-1")).isEqualTo("SHA-1")
        assertThat(LdsSecurityObjectParser.getJavaAlgorithmName("MD5")).isEqualTo("MD5")
    }

    @Test
    fun `parse returns null for empty data`() {
        val result = LdsSecurityObjectParser.parse(byteArrayOf())
        assertThat(result).isNull()
    }

    @Test
    fun `parse returns null for invalid ASN1 data`() {
        val invalidData = byteArrayOf(0x01, 0x02, 0x03)
        val result = LdsSecurityObjectParser.parse(invalidData)
        assertThat(result).isNull()
    }

    @Test
    fun `LdsSecurityObject hasHash returns correct values`() {
        val ldsObject = LdsSecurityObjectParser.LdsSecurityObject(
            version = 0,
            hashAlgorithm = "SHA-256",
            hashAlgorithmOid = "2.16.840.1.101.3.4.2.1",
            dataGroupHashes = mapOf(
                1 to ByteArray(32),
                2 to ByteArray(32)
            )
        )

        assertThat(ldsObject.hasHash(1)).isTrue()
        assertThat(ldsObject.hasHash(2)).isTrue()
        assertThat(ldsObject.hasHash(3)).isFalse()
    }

    @Test
    fun `LdsSecurityObject getHash returns correct hash`() {
        val dg1Hash = ByteArray(32) { it.toByte() }
        val ldsObject = LdsSecurityObjectParser.LdsSecurityObject(
            version = 0,
            hashAlgorithm = "SHA-256",
            hashAlgorithmOid = "2.16.840.1.101.3.4.2.1",
            dataGroupHashes = mapOf(1 to dg1Hash)
        )

        assertThat(ldsObject.getHash(1)).isEqualTo(dg1Hash)
        assertThat(ldsObject.getHash(2)).isNull()
    }

    @Test
    fun `LdsSecurityObject toString contains expected info`() {
        val ldsObject = LdsSecurityObjectParser.LdsSecurityObject(
            version = 0,
            hashAlgorithm = "SHA-256",
            hashAlgorithmOid = "2.16.840.1.101.3.4.2.1",
            dataGroupHashes = mapOf(
                1 to ByteArray(32),
                2 to ByteArray(32)
            )
        )

        val str = ldsObject.toString()
        assertThat(str).contains("version=0")
        assertThat(str).contains("SHA-256")
        assertThat(str).contains("DG1")
        assertThat(str).contains("DG2")
    }
}
