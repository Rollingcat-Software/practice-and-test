# Launcher Icons - Action Required

## ⚠️ Important: Launcher Icons Need to be Added

The app is fully functional, but you need to add launcher icons before the app can be built and installed.

Android Studio requires launcher icon files in the mipmap directories. The code references:
- `@mipmap/ic_launcher` (regular icon)
- `@mipmap/ic_launcher_round` (round icon for some devices)

## Quick Fix Options

### Option 1: Use Android Studio's Asset Studio (Recommended)

1. Right-click on `app/src/main/res` in Android Studio
2. Select **New → Image Asset**
3. Select **Launcher Icons (Adaptive and Legacy)**
4. Choose your icon source:
   - Use the **Clip Art** option and select an NFC or ID card icon
   - Or upload your own image
5. Click **Next** then **Finish**

This will automatically generate all required icon sizes and formats.

### Option 2: Use an Online Icon Generator

1. Go to https://icon.kitchen or https://romannurik.github.io/AndroidAssetStudio/
2. Upload an icon image (512x512 recommended)
3. Customize the look
4. Download the generated icon pack
5. Extract the contents to `app/src/main/res/`

### Option 3: Use Placeholder Icons Temporarily

Create simple placeholder icons using ImageMagick:

```bash
# Install ImageMagick if needed
# Ubuntu: sudo apt-get install imagemagick
# macOS: brew install imagemagick

cd app/src/main/res

# Generate placeholder icons (red background with white "NFC" text)
for size in "48:mdpi" "72:hdpi" "96:xhdpi" "144:xxhdpi" "192:xxxhdpi"; do
  px=$(echo $size | cut -d: -f1)
  dpi=$(echo $size | cut -d: -f2)

  convert -size ${px}x${px} xc:#E30A17 \
    -font Arial -pointsize $((px/3)) -fill white \
    -gravity center -annotate +0+0 "NFC" \
    mipmap-${dpi}/ic_launcher.png

  cp mipmap-${dpi}/ic_launcher.png mipmap-${dpi}/ic_launcher_round.png
done
```

### Option 4: Copy from Another Project

If you have another Android project, you can copy the launcher icons:

```bash
cp -r /path/to/other/project/app/src/main/res/mipmap-* \
      app/src/main/res/
```

## Recommended Icon Design

For a Turkish eID NFC Reader app, consider these design ideas:

1. **NFC Symbol** + Turkish Flag Colors (Red & White)
2. **ID Card Icon** with NFC waves
3. **Simple "T.C." text** with NFC symbol
4. **Crescent and Star** (Turkish flag symbol) with tech element

## Icon Specifications

- **Size**: 512x512 pixels (source)
- **Format**: PNG with transparency
- **Shape**: Adaptive (both round and square versions)
- **Colors**: Use Turkish flag red (#E30A17) as primary

## Generated Sizes Needed

| Density | Size | Directory |
|---------|------|-----------|
| mdpi | 48x48 | mipmap-mdpi |
| hdpi | 72x72 | mipmap-hdpi |
| xhdpi | 96x96 | mipmap-xhdpi |
| xxhdpi | 144x144 | mipmap-xxhdpi |
| xxxhdpi | 192x192 | mipmap-xxxhdpi |

## After Adding Icons

Once you've added the launcher icons:

1. Rebuild the project:
   ```bash
   ./gradlew clean build
   ```

2. Verify the icons are in place:
   ```bash
   ls -la app/src/main/res/mipmap-*/ic_launcher.png
   ```

3. Install and test:
   ```bash
   ./gradlew installDebug
   ```

4. Check your app drawer - the icon should now be visible!

---

**Note**: The app will NOT build/install until launcher icons are added. This is a requirement by Android.
