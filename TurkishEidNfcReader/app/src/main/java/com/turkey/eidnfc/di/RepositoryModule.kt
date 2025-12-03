package com.turkey.eidnfc.di

import com.turkey.eidnfc.data.repository.EidRepository
import com.turkey.eidnfc.data.repository.EidRepositoryImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module for providing repository implementations.
 *
 * This module uses @Binds instead of @Provides for better performance,
 * as it generates less code and is more efficient for interface-to-implementation bindings.
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    /**
     * Binds the EidRepository interface to its implementation.
     *
     * This tells Hilt to use EidRepositoryImpl whenever EidRepository is requested.
     *
     * @param impl The implementation to bind
     * @return The interface type
     */
    @Binds
    @Singleton
    abstract fun bindEidRepository(
        impl: EidRepositoryImpl
    ): EidRepository
}
