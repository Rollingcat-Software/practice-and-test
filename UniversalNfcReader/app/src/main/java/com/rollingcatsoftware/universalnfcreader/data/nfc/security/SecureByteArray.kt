package com.rollingcatsoftware.universalnfcreader.data.nfc.security

import java.io.Closeable
import java.security.SecureRandom
import java.util.Arrays

/**
 * Secure byte array wrapper that ensures sensitive data is zeroed from memory when no longer needed.
 *
 * COMPLIANCE: se-checklist.md
 * - Section 1.1: "Clear PIN bytes from memory after use"
 * - Section 1.2: "PIN should only exist in memory during authentication"
 *
 * This class provides:
 * 1. Automatic memory zeroing via [close] method
 * 2. Integration with Kotlin's [use] block for automatic cleanup
 * 3. Overwrite with random data before zeroing to prevent cold boot attacks
 * 4. Immutable size after creation
 *
 * Usage:
 * ```kotlin
 * SecureByteArray.wrap(sensitiveData).use { secure ->
 *     // Use secure.toByteArray() or secure[index]
 * } // Automatically zeroed here
 * ```
 *
 * Security Notes:
 * - Always use [use] blocks to ensure cleanup even if exceptions occur
 * - Avoid copying data out unnecessarily (use [withData] for operations)
 * - The underlying array is overwritten with random data then zeroed on close
 *
 * @property size The size of the underlying byte array
 */
class SecureByteArray private constructor(
    @PublishedApi internal val data: ByteArray
) : Closeable {

    @Volatile
    @PublishedApi internal var isClosed = false

    /**
     * The size of the underlying byte array.
     */
    val size: Int
        get() {
            checkNotClosed()
            return data.size
        }

    /**
     * Returns true if this SecureByteArray has been closed and zeroed.
     */
    val closed: Boolean
        get() = isClosed

    /**
     * Gets a byte at the specified index.
     *
     * @param index The index to read from
     * @return The byte value at the index
     * @throws IllegalStateException if the array has been closed
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    operator fun get(index: Int): Byte {
        checkNotClosed()
        return data[index]
    }

    /**
     * Sets a byte at the specified index.
     *
     * @param index The index to write to
     * @param value The byte value to set
     * @throws IllegalStateException if the array has been closed
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    operator fun set(index: Int, value: Byte) {
        checkNotClosed()
        data[index] = value
    }

    /**
     * Returns a copy of the underlying byte array.
     *
     * WARNING: The returned array is NOT secure - caller is responsible for clearing it.
     * Prefer using [withData] when possible to avoid creating insecure copies.
     *
     * @return A copy of the data
     * @throws IllegalStateException if the array has been closed
     */
    fun toByteArray(): ByteArray {
        checkNotClosed()
        return data.copyOf()
    }

    /**
     * Returns a copy of a range of the underlying byte array.
     *
     * WARNING: The returned array is NOT secure - caller is responsible for clearing it.
     *
     * @param fromIndex Start index (inclusive)
     * @param toIndex End index (exclusive)
     * @return A copy of the specified range
     * @throws IllegalStateException if the array has been closed
     */
    fun copyOfRange(fromIndex: Int, toIndex: Int): ByteArray {
        checkNotClosed()
        return data.copyOfRange(fromIndex, toIndex)
    }

    /**
     * Performs an operation with direct access to the underlying data.
     *
     * This is the preferred way to access data as it doesn't create copies.
     * The lambda receives a read-only view of the data.
     *
     * @param block The operation to perform with the data
     * @return The result of the operation
     * @throws IllegalStateException if the array has been closed
     */
    inline fun <R> withData(block: (ByteArray) -> R): R {
        checkNotClosed()
        return block(data)
    }

    /**
     * Fills the entire array with a specific byte value.
     *
     * @param value The byte value to fill with
     * @throws IllegalStateException if the array has been closed
     */
    fun fill(value: Byte) {
        checkNotClosed()
        Arrays.fill(data, value)
    }

    /**
     * XORs this array with another byte array, storing the result in this array.
     *
     * @param other The array to XOR with (must be same length)
     * @throws IllegalStateException if the array has been closed
     * @throws IllegalArgumentException if arrays have different lengths
     */
    fun xorWith(other: ByteArray) {
        checkNotClosed()
        require(other.size == data.size) { "Arrays must have the same length" }
        for (i in data.indices) {
            data[i] = (data[i].toInt() xor other[i].toInt()).toByte()
        }
    }

    /**
     * XORs this array with another SecureByteArray, storing the result in this array.
     *
     * @param other The SecureByteArray to XOR with (must be same length)
     * @throws IllegalStateException if either array has been closed
     * @throws IllegalArgumentException if arrays have different lengths
     */
    fun xorWith(other: SecureByteArray) {
        checkNotClosed()
        other.checkNotClosed()
        require(other.size == data.size) { "Arrays must have the same length" }
        for (i in data.indices) {
            data[i] = (data[i].toInt() xor other.data[i].toInt()).toByte()
        }
    }

    /**
     * Copies data from a source array into this SecureByteArray.
     *
     * @param src Source array
     * @param srcPos Starting position in source array
     * @param destPos Starting position in this array
     * @param length Number of bytes to copy
     * @throws IllegalStateException if the array has been closed
     */
    fun copyFrom(src: ByteArray, srcPos: Int = 0, destPos: Int = 0, length: Int = src.size) {
        checkNotClosed()
        System.arraycopy(src, srcPos, data, destPos, length)
    }

    /**
     * Securely closes this array by overwriting with random data then zeroing.
     *
     * This method is idempotent - calling it multiple times has no additional effect.
     *
     * The two-phase wipe (random then zero) provides protection against:
     * 1. Simple memory scanning (looking for non-zero sensitive data)
     * 2. Cold boot attacks (random data obscures patterns)
     */
    override fun close() {
        if (isClosed) return

        synchronized(this) {
            if (isClosed) return

            // Phase 1: Overwrite with random data to obscure patterns
            try {
                SecureRandom().nextBytes(data)
            } catch (e: Exception) {
                // If SecureRandom fails, still zero the data
            }

            // Phase 2: Zero all bytes
            Arrays.fill(data, 0.toByte())

            isClosed = true
        }
    }

    /**
     * Checks if the array has been closed and throws if so.
     */
    @PublishedApi internal fun checkNotClosed() {
        if (isClosed) {
            throw IllegalStateException("SecureByteArray has been closed and data has been wiped")
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SecureByteArray) return false
        if (isClosed || other.isClosed) return false
        return data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        if (isClosed) return 0
        return data.contentHashCode()
    }

    /**
     * Returns a string representation without exposing the data.
     */
    override fun toString(): String {
        return if (isClosed) {
            "SecureByteArray[closed]"
        } else {
            "SecureByteArray[size=$size]"
        }
    }

    companion object {
        /**
         * Wraps an existing byte array in a SecureByteArray.
         *
         * NOTE: This takes ownership of the array. The original array reference
         * should not be used after wrapping.
         *
         * @param data The byte array to wrap
         * @return A new SecureByteArray wrapping the data
         */
        fun wrap(data: ByteArray): SecureByteArray = SecureByteArray(data)

        /**
         * Creates a new SecureByteArray by copying from an existing byte array.
         *
         * The original array is NOT modified or cleared - caller is responsible for it.
         *
         * @param data The byte array to copy from
         * @return A new SecureByteArray containing a copy of the data
         */
        fun copyOf(data: ByteArray): SecureByteArray = SecureByteArray(data.copyOf())

        /**
         * Allocates a new SecureByteArray of the specified size, initialized to zeros.
         *
         * @param size The size of the array to allocate
         * @return A new SecureByteArray of the specified size
         */
        fun allocate(size: Int): SecureByteArray = SecureByteArray(ByteArray(size))

        /**
         * Creates a SecureByteArray from a hex string.
         *
         * @param hex The hex string (e.g., "DEADBEEF")
         * @return A new SecureByteArray containing the decoded bytes
         * @throws IllegalArgumentException if the hex string is invalid
         */
        fun fromHex(hex: String): SecureByteArray {
            require(hex.length % 2 == 0) { "Hex string must have even length" }
            val data = ByteArray(hex.length / 2)
            for (i in data.indices) {
                val index = i * 2
                val byte = hex.substring(index, index + 2).toInt(16)
                data[i] = byte.toByte()
            }
            return SecureByteArray(data)
        }

        /**
         * Concatenates multiple SecureByteArrays into a new one.
         *
         * The source arrays are NOT closed - caller is responsible for them.
         *
         * @param arrays The arrays to concatenate
         * @return A new SecureByteArray containing all the data
         */
        fun concat(vararg arrays: SecureByteArray): SecureByteArray {
            val totalSize = arrays.sumOf { it.size }
            val result = ByteArray(totalSize)
            var offset = 0
            for (array in arrays) {
                array.withData { data ->
                    System.arraycopy(data, 0, result, offset, data.size)
                }
                offset += array.size
            }
            return SecureByteArray(result)
        }

        /**
         * Concatenates byte arrays into a new SecureByteArray.
         *
         * @param arrays The arrays to concatenate
         * @return A new SecureByteArray containing all the data
         */
        fun concat(vararg arrays: ByteArray): SecureByteArray {
            val totalSize = arrays.sumOf { it.size }
            val result = ByteArray(totalSize)
            var offset = 0
            for (array in arrays) {
                System.arraycopy(array, 0, result, offset, array.size)
                offset += array.size
            }
            return SecureByteArray(result)
        }

        /**
         * Generates a SecureByteArray filled with cryptographically secure random bytes.
         *
         * @param size The number of random bytes to generate
         * @return A new SecureByteArray containing random data
         */
        fun random(size: Int): SecureByteArray {
            val data = ByteArray(size)
            SecureRandom().nextBytes(data)
            return SecureByteArray(data)
        }
    }
}

/**
 * Extension function to securely clear a ByteArray.
 *
 * This fills the array with zeros. For more secure clearing (random + zero),
 * use [SecureByteArray] instead.
 */
fun ByteArray.secureClear() {
    Arrays.fill(this, 0.toByte())
}

/**
 * Extension function to securely clear a ByteArray with random data then zeros.
 *
 * Provides the same two-phase wipe as [SecureByteArray.close].
 */
fun ByteArray.secureWipe() {
    try {
        SecureRandom().nextBytes(this)
    } catch (e: Exception) {
        // Continue to zeroing even if random fails
    }
    Arrays.fill(this, 0.toByte())
}
