# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Keep Bouncy Castle classes
-keep class org.bouncycastle.** { *; }
-dontwarn org.bouncycastle.**

# Keep JAI ImageIO classes
-keep class com.github.jaiimageio.** { *; }
-dontwarn com.github.jaiimageio.**
-dontwarn javax.imageio.**

# Keep NFC-related classes
-keep class android.nfc.** { *; }

# Keep data classes and models
-keep class com.turkey.eidnfc.domain.model.** { *; }

# Uncomment this to preserve the line number information for debugging
-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to hide the original source file name
-renamesourcefileattribute SourceFile
