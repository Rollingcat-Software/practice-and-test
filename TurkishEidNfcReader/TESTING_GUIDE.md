# Testing Guide - Turkish eID NFC Reader

This guide provides detailed step-by-step instructions for testing the Turkish eID NFC Reader application.

## Pre-Testing Checklist

Before you begin testing, ensure you have:

- [ ] Android device with NFC capability
- [ ] Android 5.0 (API 21) or higher
- [ ] Turkish National ID Card (chip-enabled)
- [ ] Your 6-digit PIN1 code
- [ ] Android Studio installed (for developers)
- [ ] USB debugging enabled on your device

## Setup Instructions

### 1. Install the Application

#### Option A: Build and Install from Android Studio

1. Open Android Studio
2. Open the `TurkishEidNfcReader` project
3. Connect your Android device via USB
4. Wait for Gradle sync to complete
5. Click the "Run" button (▶️)
6. Select your device from the list
7. Wait for the app to install and launch

#### Option B: Install Pre-built APK

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 2. Enable NFC on Your Device

1. Open **Settings** on your Android device
2. Navigate to **Connected devices** → **Connection preferences**
3. Find **NFC** and toggle it **ON**
4. Verify the NFC icon appears in your status bar

### 3. Enable Developer Options (Optional, for debugging)

1. Go to **Settings** → **About phone**
2. Tap **Build number** 7 times
3. Go back to **Settings** → **System** → **Developer options**
4. Enable **USB debugging**
5. Connect to Android Studio for logcat viewing

## Test Cases

### Test Case 1: First Launch

**Objective**: Verify app launches correctly and displays initial screen

**Steps**:
1. Launch the app from the app drawer
2. Observe the initial screen

**Expected Results**:
- ✅ App opens without crashing
- ✅ "Turkish eID Card Reader" title is displayed
- ✅ NFC icon is visible
- ✅ PIN input field is present
- ✅ Instructions are clear and readable
- ✅ "Enter your 6-digit PIN" label is visible

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 2: NFC Availability Check

**Objective**: Verify app detects NFC availability

**Steps**:
1. Launch app on device without NFC
2. Observe any error messages

**Expected Results**:
- ✅ On non-NFC device: Shows "NFC is not available on this device"
- ✅ On NFC device with NFC disabled: Shows "Please enable NFC in device settings"
- ✅ On NFC device with NFC enabled: No error message

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 3: PIN Input Validation

**Objective**: Verify PIN input field validation

**Steps**:
1. Try entering letters → Should be rejected
2. Try entering special characters → Should be rejected
3. Try entering 7 digits → Should limit to 6
4. Enter 6 digits → Should be accepted
5. Test the show/hide PIN toggle

**Expected Results**:
- ✅ Only digits 0-9 are accepted
- ✅ Maximum 6 characters enforced
- ✅ Character count shows "X/6"
- ✅ Eye icon toggles PIN visibility
- ✅ PIN is masked by default (dots)

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 4: Successful Card Read

**Objective**: Read card data with correct PIN

**Prerequisites**: Know your correct 6-digit PIN

**Steps**:
1. Enter your correct 6-digit PIN
2. Hold your ID card against the back of the phone
3. Keep steady for 5-10 seconds
4. Observe the reading process
5. View the results

**Expected Results**:
- ✅ Loading screen appears with "Reading card..."
- ✅ Progress indicator is shown
- ✅ Success screen displays after 5-10 seconds
- ✅ Personal data is correctly displayed:
  - Name and surname
  - TCKN (11 digits)
  - Birth date
  - Gender
  - Nationality
  - Document number
  - Expiry date
- ✅ Photo is displayed (if available)
- ✅ "Read Another Card" button is present

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

**Verification**:
Compare displayed data with your physical ID card:
- Name matches: [ ] Yes [ ] No
- TCKN matches: [ ] Yes [ ] No
- Birth date matches: [ ] Yes [ ] No
- Photo matches: [ ] Yes [ ] No

---

### Test Case 5: Wrong PIN (First Attempt)

**Objective**: Verify wrong PIN handling

**Steps**:
1. Enter an incorrect 6-digit PIN (e.g., "000000")
2. Hold your ID card against the phone
3. Observe the error message

**Expected Results**:
- ✅ Error screen appears
- ✅ Message shows "Wrong PIN. X attempt(s) remaining"
- ✅ Number of remaining attempts is displayed (should be 2)
- ✅ "Try Again" button is present
- ✅ PIN field is cleared for security

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 6: Wrong PIN (Multiple Attempts)

**Objective**: Verify retry attempt tracking

⚠️ **WARNING**: Only perform this test if you can afford to lock your card. Have your PUK ready.

**Steps**:
1. Enter wrong PIN three times in a row
2. Observe the behavior after each attempt

**Expected Results**:
- ✅ First wrong attempt: "2 attempt(s) remaining"
- ✅ Second wrong attempt: "1 attempt(s) remaining"
- ✅ Third wrong attempt: "Card is locked" message
- ✅ Clear warning that card needs to be unlocked with PUK

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________
- [ ] Skipped (don't want to lock card)

---

### Test Case 7: Card Positioning

**Objective**: Test different card positions to find NFC antenna

**Steps**:
1. Enter correct PIN
2. Try different positions on the back of the phone:
   - Top center
   - Middle center
   - Bottom center
   - Top left
   - Top right

**Expected Results**:
- ✅ Card is successfully read in at least one position
- ✅ Reading fails if card is not positioned correctly
- ✅ Connection lost message if card is moved during reading

**Actual Results**:
- Best position: _______________
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 8: Connection Lost During Reading

**Objective**: Verify error handling when card is removed during reading

**Steps**:
1. Enter correct PIN
2. Hold card against phone
3. When reading starts, move the card away
4. Observe the error

**Expected Results**:
- ✅ Error message: "Connection to card lost"
- ✅ "Try Again" button is present
- ✅ App doesn't crash
- ✅ Can retry immediately

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 9: Read Multiple Cards

**Objective**: Verify app can read multiple cards in sequence

**Steps**:
1. Successfully read first card
2. Click "Read Another Card"
3. Enter PIN for second card (can be same or different)
4. Read second card

**Expected Results**:
- ✅ App returns to idle state
- ✅ PIN field is cleared
- ✅ Previous card data is cleared
- ✅ Second card reads successfully
- ✅ New card data is displayed

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 10: App Lifecycle

**Objective**: Test app behavior during lifecycle events

**Steps**:
1. Start reading a card
2. Press Home button during reading
3. Return to app
4. Try reading again

**Expected Results**:
- ✅ Reading is cancelled when app goes to background
- ✅ App resumes in correct state
- ✅ No crash on resume
- ✅ Can start new read after resume

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 11: Device Rotation

**Objective**: Verify UI handles rotation correctly

**Steps**:
1. Start the app in portrait mode
2. Rotate to landscape
3. Rotate back to portrait
4. Try reading a card in landscape mode

**Expected Results**:
- ✅ UI adapts to rotation
- ✅ PIN input is preserved during rotation
- ✅ No crash on rotation
- ✅ All UI elements remain accessible

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

## Logcat Monitoring

For developers, monitor logcat for detailed information:

### 1. Filter by Application

```bash
adb logcat | grep "TurkishEid"
```

### 2. Filter by Timber Tag

```bash
adb logcat | grep "Turkey"
```

### 3. Key Log Messages to Watch For

**Successful Flow**:
```
D/MainActivity: Tag detected
D/NfcCardReader: Connected to card
D/ApduHelper: → APDU Command: 00 A4 04 0C...
D/ApduHelper: ← APDU Response: ... 90 00
D/ApduHelper: Status: Success (0x9000)
D/NfcCardReader: Turkish eID application selected successfully
D/NfcCardReader: PIN verified successfully
D/Dg1Parser: Parsing DG1 data
D/Dg2Parser: Parsing DG2 data
D/NfcCardReader: Card reading completed successfully
```

**Error Flow (Wrong PIN)**:
```
E/ApduHelper: Wrong PIN. 2 attempt(s) remaining
E/NfcCardReader: Card reading failed: Wrong PIN
```

**Error Flow (Card Locked)**:
```
E/ApduHelper: Authentication method blocked. Card is locked
E/NfcCardReader: Card is blocked
```

---

## Performance Testing

### Timing Benchmarks

Measure the time for each operation:

1. **Card Detection**: From tap to "Reading card..." message
   - Expected: < 1 second
   - Actual: _______ seconds

2. **Complete Read**: From "Reading card..." to data display
   - Expected: 5-10 seconds
   - Actual: _______ seconds

3. **DG1 Parse**: Time to parse personal data
   - Expected: < 1 second
   - Actual: _______ seconds

4. **DG2 Decode**: Time to decode photo
   - Expected: 1-3 seconds
   - Actual: _______ seconds

---

## Compatibility Testing

Test on multiple devices and Android versions:

| Device Model | Android Version | NFC Chip | Result | Notes |
|--------------|----------------|----------|--------|-------|
| | | | [ ] Pass [ ] Fail | |
| | | | [ ] Pass [ ] Fail | |
| | | | [ ] Pass [ ] Fail | |

---

## Security Testing

### Test Case 12: PIN Security

**Objective**: Verify PIN is not leaked or stored

**Steps**:
1. Enter PIN and read card successfully
2. Check app data directory
3. Check logcat output (in release build)
4. Check for any persisted files

**Expected Results**:
- ✅ No PIN in logcat (release build)
- ✅ No PIN in shared preferences
- ✅ No PIN in any files
- ✅ PIN is cleared after successful read

**Verification Commands**:
```bash
# Check app data
adb shell run-as com.turkey.eidnfc ls -la

# Check shared prefs
adb shell run-as com.turkey.eidnfc cat shared_prefs/*.xml
```

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 13: Data Persistence

**Objective**: Verify card data is not persisted

**Steps**:
1. Read a card successfully
2. Close the app
3. Reopen the app
4. Check app data

**Expected Results**:
- ✅ Previous card data is not displayed
- ✅ App starts in idle state
- ✅ No card data in app storage

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

## Edge Cases

### Test Case 14: Old/Damaged Card

**Objective**: Handle cards that may not work

**Steps**:
1. Try with an old (non-chip) ID card
2. Try with a slightly damaged card

**Expected Results**:
- ✅ Old card: "This is not a valid Turkish eID card"
- ✅ Damaged card: May work or show parse error
- ✅ App doesn't crash

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

### Test Case 15: NFC Interference

**Objective**: Test with potential interference

**Steps**:
1. Try reading with phone case on
2. Try with metal objects nearby
3. Try with multiple cards stacked

**Expected Results**:
- ✅ Thick cases may block NFC
- ✅ Metal interference causes connection issues
- ✅ Multiple cards cause read failures
- ✅ Error messages are appropriate

**Actual Results**:
- [ ] Pass
- [ ] Fail (describe issue): _______________

---

## Bug Report Template

If you encounter a bug, use this template:

**Bug Title**: [Brief description]

**Severity**: [ ] Critical [ ] Major [ ] Minor

**Device Info**:
- Model: _______________
- Android Version: _______________
- App Version: _______________

**Steps to Reproduce**:
1.
2.
3.

**Expected Result**:

**Actual Result**:

**Logcat Output** (if available):
```
[Paste relevant logcat here]
```

**Screenshots** (if applicable):

---

## Test Summary

### Overall Results

- Total Test Cases: 15
- Passed: _____
- Failed: _____
- Skipped: _____

### Critical Issues Found

1.
2.
3.

### Recommendations

1.
2.
3.

### Sign-off

- Tester Name: _______________
- Date: _______________
- Environment: _______________
- Overall Assessment: [ ] Ready [ ] Needs Work [ ] Not Ready

---

## Additional Notes

Use this space for any additional observations or comments:

---

**End of Testing Guide**
