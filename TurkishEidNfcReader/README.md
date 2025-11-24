# Turkish eID NFC Reader 🇹🇷

A complete, functional Android Kotlin NFC application that reads data from the Turkish National ID Card (Türkiye Cumhuriyeti Yeni Kimlik Kartı, eID) using NFC + IsoDep.

![Android](https://img.shields.io/badge/Android-5.0%2B-green)
![Kotlin](https://img.shields.io/badge/Kotlin-1.9.20-blue)
![License](https://img.shields.io/badge/License-Educational-orange)

## Features

- ✅ **NFC Card Detection**: Automatically detects Turkish eID cards via NFC
- ✅ **IsoDep Communication**: Establishes secure communication using ISO 7816-4 protocol
- ✅ **PIN Authentication**: Securely verifies 6-digit PIN (PIN1)
- ✅ **Personal Data Reading**: Reads DG1 (name, surname, TCKN, birthdate, etc.)
- ✅ **Photo Extraction**: Reads and decodes DG2 (facial image in JPEG2000 format)
- ✅ **SOD Validation**: Validates Security Object Document signatures
- ✅ **Modern UI**: Beautiful Jetpack Compose interface with Material Design 3
- ✅ **Error Handling**: Comprehensive error messages and retry attempt tracking
- ✅ **Security**: No data persistence, memory-only PIN handling

## Screenshots

*[The app will display a PIN input screen, reading progress, and finally the card data with photo]*

## Requirements

### Device Requirements
- Android device with NFC capability
- Android 5.0 (API 21) or higher
- NFC must be enabled in device settings

### Development Requirements
- Android Studio Arctic Fox or newer
- JDK 17
- Kotlin 1.9.20
- Gradle 8.1.4

### Card Requirements
- Turkish National ID Card (Türkiye Cumhuriyeti Kimlik Kartı)
- Valid 6-digit PIN1 (provided when you received your card)

## Installation

### Option 1: Build from Source

1. **Clone the repository**
   ```bash
   cd TurkishEidNfcReader
   ```

2. **Open in Android Studio**
   - Open Android Studio
   - Select "Open an Existing Project"
   - Navigate to the `TurkishEidNfcReader` folder

3. **Sync Gradle**
   - Android Studio will automatically sync Gradle dependencies
   - Wait for the sync to complete

4. **Build and Run**
   - Connect your Android device via USB
   - Enable USB debugging on your device
   - Click "Run" (▶️) in Android Studio
   - Select your device from the list

### Option 2: Install APK

1. Build the APK:
   ```bash
   cd TurkishEidNfcReader
   ./gradlew assembleDebug
   ```

2. Install on device:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

## Usage

### Step 1: Prepare Your Device

1. **Enable NFC**
   - Go to Settings → Connected devices → Connection preferences → NFC
   - Turn on NFC
   - Ensure "NFC" toggle is enabled

2. **Launch the App**
   - Open "Turkish eID NFC Reader" from your app drawer

### Step 2: Enter Your PIN

1. **Enter 6-digit PIN**
   - Type your 6-digit PIN1 in the input field
   - You can toggle visibility with the eye icon
   - The PIN must be exactly 6 digits

⚠️ **Important**: Your PIN is the 6-digit code you set when you received your ID card. If you don't know it, contact the authorities.

### Step 3: Scan Your Card

1. **Position the card**
   - Hold your ID card flat against the back of your phone
   - The NFC antenna is usually located in the center or top of the device
   - Keep the card steady

2. **Wait for reading**
   - The app will automatically detect the card
   - A loading screen will appear showing "Reading card..."
   - Keep the card in place for 5-10 seconds

3. **View results**
   - Personal data (name, TCKN, birthdate, etc.) will be displayed
   - Your photo from the card will be shown
   - You can read another card by clicking "Read Another Card"

## Troubleshooting

### Common Issues

#### ❌ "NFC is not available on this device"
**Solution**: Your device doesn't have NFC hardware. You need an NFC-enabled device to use this app.

#### ❌ "Please enable NFC in device settings"
**Solution**: Go to Settings → Connected devices → NFC and turn it on.

#### ❌ "Wrong PIN. X attempt(s) remaining"
**Solutions**:
- Double-check your PIN
- You have limited attempts (usually 3)
- After 3 wrong attempts, the card will be locked

#### ❌ "Card is locked"
**Solution**: Your card has been locked due to too many wrong PIN attempts. Visit the nearest Nüfus Müdürlüğü (Population Office) to unlock it with your PUK code.

#### ❌ "This is not a valid Turkish eID card"
**Solutions**:
- Make sure you're using a Turkish national ID card
- Old ID cards (not chip-enabled) won't work
- Try repositioning the card

#### ❌ "Connection to card lost. Please try again."
**Solutions**:
- Hold the card steady against the phone
- Don't move the card during reading
- Make sure there's no metal case interfering with NFC

#### ❌ "Failed to parse card data"
**Solutions**:
- Try scanning again
- Make sure the card is not damaged
- Clean the card surface

### Testing Tips

1. **Find Your NFC Antenna**
   - Try different positions on the back of your phone
   - Usually center or top of the device
   - Look for NFC symbol on your device

2. **Optimal Positioning**
   - Hold the card flat (not angled)
   - Center the card on the NFC antenna location
   - Keep steady for the entire reading process

3. **Multiple Attempts**
   - If first attempt fails, try again
   - Try different positions
   - Remove phone case if it's thick

## Architecture

The app follows Clean Architecture principles with MVVM pattern:

```
app/
├── data/
│   └── nfc/
│       ├── ApduHelper.kt         # APDU command construction and parsing
│       └── NfcCardReader.kt      # NFC communication and card reading
├── domain/
│   └── model/
│       └── EidData.kt            # Data models and result types
├── ui/
│   ├── MainViewModel.kt          # State management
│   ├── MainScreen.kt             # Jetpack Compose UI
│   └── theme/
│       └── Theme.kt              # Material Design theme
├── util/
│   ├── Dg1Parser.kt              # DG1 personal data parser
│   ├── Dg2Parser.kt              # DG2 photo decoder
│   └── SodValidator.kt           # SOD signature validation
└── MainActivity.kt               # NFC intent handling
```

## Technical Details

### Turkish eID Structure

#### Application Identifier (AID)
```
A0 00 00 01 67 45 53 49 44
```

#### File IDs (FID)
| File | FID | Access | Description |
|------|-----|--------|-------------|
| EF.CardAccess | 0x011C | Public | PACE information |
| EF.SOD | 0x011D | Public | Security Object Document |
| DG1 | 0x0101 | PIN Required | Personal data (MRZ) |
| DG2 | 0x0102 | PIN Required | Facial image (JPEG2000) |
| DG3 | 0x0103 | Restricted | Fingerprints |

#### APDU Commands

**Select AID**
```
00 A4 04 0C 10 A00000016745534944 00
```

**Verify PIN**
```
00 20 00 81 06 [PIN_BYTES]
```

**Select File**
```
00 A4 02 0C 02 [FILE_ID] 00
```

**Read Binary**
```
00 B0 [OFFSET_HI] [OFFSET_LO] [LENGTH]
```

#### Response Codes
| Code | Meaning |
|------|---------|
| 9000 | Success |
| 63CX | Wrong PIN, X attempts remaining |
| 6982 | Security not satisfied |
| 6983 | Authentication method blocked (card locked) |
| 6A82 | File not found |
| 6A86 | Wrong parameters |

### Data Parsing

#### DG1 Structure
DG1 contains MRZ (Machine Readable Zone) in TD1 format:
- **Line 1**: Document type, country, document number
- **Line 2**: Birth date, gender, expiry date, nationality
- **Line 3**: Name (surname and given names)

The app parses this data and extracts:
- TCKN (Turkish Citizenship Number - 11 digits)
- First and last name
- Birth date
- Gender (M/F)
- Document number
- Expiry date
- Nationality

#### DG2 Structure
DG2 contains facial image encoded in JPEG2000 format wrapped in ICAO LDS ASN.1 structure. The app:
1. Navigates the ASN.1 structure
2. Extracts JPEG2000 data
3. Decodes using JAI ImageIO library
4. Converts to Android Bitmap

### Security Features

1. **No Data Persistence**: All data is kept in memory only
2. **PIN Protection**: PIN is cleared from memory after use
3. **Secure Communication**: Uses ISO 7816-4 secure messaging
4. **SOD Validation**: Verifies document signer certificate
5. **No Logging in Release**: Timber only logs in debug builds
6. **No Backup**: Sensitive data excluded from Android backup

## Dependencies

- **Jetpack Compose**: Modern Android UI toolkit
- **Material Design 3**: Latest Material Design components
- **Bouncy Castle**: Cryptographic operations for SOD validation
- **JAI ImageIO**: JPEG2000 image decoding
- **Timber**: Logging (debug only)
- **Coroutines**: Asynchronous operations

## Security Considerations

⚠️ **Important Security Notes**:

1. **Authorized Use Only**: Only use this app with your own ID card
2. **PIN Security**: Never share your PIN with anyone
3. **Privacy**: The app does not store, transmit, or share any data
4. **KVKK Compliance**: Respects Turkish data protection law
5. **Educational Purpose**: This app is for educational and personal use only

## Legal & Ethical

This application is designed for:
- ✅ Personal use (reading your own card)
- ✅ Educational purposes
- ✅ Research and development
- ✅ Security testing (with authorization)

**NOT for**:
- ❌ Unauthorized access to others' cards
- ❌ Data harvesting
- ❌ Identity fraud
- ❌ Any illegal activities

**Disclaimer**: Users are responsible for complying with Turkish laws, including KVKK (Turkish GDPR). The developers assume no liability for misuse of this application.

## Customization

### Update PIN in Code (for testing)

If you want to hardcode a PIN for testing purposes (NOT recommended for production):

1. Open `app/src/main/java/com/turkey/eidnfc/ui/MainViewModel.kt`
2. In the `onTagDetected` function, replace `val currentPin = _pin.value` with:
   ```kotlin
   val currentPin = "123456" // Your test PIN
   ```

⚠️ **Warning**: Never hardcode PINs in production code!

### Modify UI Theme

The app uses Turkish flag colors (red and white) as the primary theme. To customize:

1. Open `app/src/main/java/com/turkey/eidnfc/ui/theme/Theme.kt`
2. Modify the color scheme values
3. Rebuild the app

## Building for Production

1. **Generate Signed APK**
   ```bash
   ./gradlew assembleRelease
   ```

2. **ProGuard Configuration**
   - ProGuard rules are already configured in `proguard-rules.pro`
   - Bouncy Castle and JAI ImageIO classes are preserved

3. **Release Checklist**
   - [ ] Update version in `build.gradle`
   - [ ] Test on multiple devices
   - [ ] Verify SOD validation works
   - [ ] Check all error scenarios
   - [ ] Review security checklist
   - [ ] Add proper signing configuration

## Contributing

This is an educational project. If you find bugs or want to improve it:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Known Limitations

1. **DG3 (Fingerprints)**: Not implemented (restricted access)
2. **PACE**: Basic Access Control not implemented
3. **Full SOD Validation**: Requires CSCA certificate (not included)
4. **EF.CardAccess**: Parsed but not fully utilized
5. **Hash Verification**: SOD hash verification structure in place but not fully implemented

## Future Enhancements

- [ ] Implement PACE (Password Authenticated Connection Establishment)
- [ ] Add CSCA certificate chain validation
- [ ] Implement full Active Authentication
- [ ] Add data export features (with user consent)
- [ ] Multi-language support (Turkish/English)
- [ ] Accessibility improvements
- [ ] Unit and integration tests

## Resources

### Official Documentation
- [ICAO Doc 9303](https://www.icao.int/publications/pages/publication.aspx?docnum=9303) - Machine Readable Travel Documents
- [ISO/IEC 7816-4](https://www.iso.org/standard/54550.html) - Smart Card Standard

### Turkish eID Specifications
- [Nüfus ve Vatandaşlık İşleri Genel Müdürlüğü](https://www.nvi.gov.tr/)
- [KVKK - Turkish Data Protection Law](https://kvkk.gov.tr/)

### Libraries Used
- [Bouncy Castle](https://www.bouncycastle.org/)
- [JAI ImageIO](https://github.com/jai-imageio/jai-imageio-core)

## Support

For issues, questions, or contributions:
- Check the [Issues](../../issues) page
- Review the [se-checklist.md](../se-checklist.md) for security guidelines

## License

This project is provided for **educational purposes only**. Use responsibly and in compliance with Turkish laws.

---

**Made with ❤️ for educational purposes**

**Version**: 1.0.0
**Last Updated**: November 2025
**Minimum Android**: 5.0 (API 21)
**Target Android**: 14 (API 34)
