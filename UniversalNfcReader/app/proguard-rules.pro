# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Keep NFC-related classes
-keep class android.nfc.** { *; }
-keep class com.rollingcatsoftware.universalnfcreader.domain.model.** { *; }
-keep class com.rollingcatsoftware.universalnfcreader.data.nfc.** { *; }

# Keep sealed classes for exhaustive when expressions
-keep class com.rollingcatsoftware.universalnfcreader.domain.model.CardData { *; }
-keep class com.rollingcatsoftware.universalnfcreader.domain.model.CardError { *; }
-keep class com.rollingcatsoftware.universalnfcreader.domain.model.Result { *; }

# Kotlin serialization (if used)
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}

# Hilt
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keep class * extends dagger.hilt.android.internal.managers.ComponentSupplier { *; }
-keep class * extends dagger.hilt.android.internal.managers.ViewComponentManager.ViewComponentBuilderEntryPoint { *; }

# Keep enum members
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Parcelable implementations
-keep class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator *;
}

# Preserve the line number information for debugging stack traces.
-keepattributes SourceFile,LineNumberTable

# Hide the original source file name.
-renamesourcefileattribute SourceFile
