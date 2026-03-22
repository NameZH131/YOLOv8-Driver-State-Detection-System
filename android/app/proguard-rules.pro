# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in Android SDK tools.
# For more details, see
#   https://developer.android.com/build/shrink-code

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep NCNN related classes
-keep class com.tencent.ncnn.** { *; }

# Keep detection result classes
-keep class com.yolo.driver.analyzer.** { *; }
-keep class com.yolo.driver.data.** { *; }
