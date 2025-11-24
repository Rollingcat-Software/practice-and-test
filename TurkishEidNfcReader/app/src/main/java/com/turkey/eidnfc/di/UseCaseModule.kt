package com.turkey.eidnfc.di

import com.turkey.eidnfc.data.repository.EidRepository
import com.turkey.eidnfc.domain.usecase.ReadEidCardUseCase
import com.turkey.eidnfc.domain.usecase.ValidatePinUseCase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.components.ViewModelComponent
import dagger.hilt.android.scopes.ViewModelScoped

/**
 * Hilt module for providing use cases.
 *
 * Use cases are scoped to ViewModelComponent because:
 * - They're primarily used by ViewModels
 * - They don't hold state
 * - ViewModelScoped ensures proper lifecycle management
 */
@Module
@InstallIn(ViewModelComponent::class)
object UseCaseModule {

    /**
     * Provides ValidatePinUseCase.
     *
     * This use case has no dependencies, so it's simple to provide.
     */
    @Provides
    @ViewModelScoped
    fun provideValidatePinUseCase(): ValidatePinUseCase {
        return ValidatePinUseCase()
    }

    /**
     * Provides ReadEidCardUseCase.
     *
     * This use case depends on:
     * - EidRepository (for data access)
     * - ValidatePinUseCase (for PIN validation)
     */
    @Provides
    @ViewModelScoped
    fun provideReadEidCardUseCase(
        repository: EidRepository,
        validatePinUseCase: ValidatePinUseCase
    ): ReadEidCardUseCase {
        return ReadEidCardUseCase(repository, validatePinUseCase)
    }
}
