# Security & Engineering Checklist - Turkish eID NFC Reader

## 🔒 Security Requirements

### 1. PIN Handling
- [ ] **Never log or store PIN in plain text**
  - PIN should only exist in memory during authentication
  - Clear PIN bytes from memory after use
  - Use SecureRandom for any PIN-related operations

- [ ] **PIN Input Validation**
  - Validate PIN is exactly 6 digits
  - Sanitize input to prevent injection attacks
  - Display masked input (dots/asterisks)

- [ ] **Retry Attempt Monitoring**
  - Parse `63CX` response to show remaining attempts
  - Warn user when attempts are low (< 3)
  - Block further attempts after card lockout

### 2. Data Protection
- [ ] **Secure Data Storage**
  - Never persist sensitive data (DG1 personal info, DG2 photo) without encryption
  - Use Android Keystore for any necessary encryption keys
  - Prefer in-memory only processing

- [ ] **Data Transmission**
  - If sharing data, use encrypted channels only
  - Implement certificate pinning for any network communication
  - No analytics/telemetry on sensitive data

### 3. NFC Security
- [ ] **IsoDep Connection Security**
  - Verify AID matches expected Turkish eID AID
  - Validate APDU responses before processing
  - Implement timeout handling (prevent DoS)

- [ ] **APDU Command Validation**
  - Validate all response codes
  - Implement proper error handling for security-related responses
  - Log security-relevant events (6982, 6983, etc.)

### 4. Cryptographic Operations
- [ ] **SOD (Security Object Document) Validation**
  - Verify digital signature using CSCA public key
  - Validate certificate chain
  - Check certificate expiration dates
  - Implement proper hash verification of data groups

- [ ] **Random Number Generation**
  - Use `SecureRandom` for any cryptographic operations
  - Never use predictable random sources

### 5. Code Security
- [ ] **Input Validation**
  - Validate all user inputs
  - Sanitize data before display
  - Prevent buffer overflows in byte operations

- [ ] **Error Handling**
  - Never expose sensitive data in error messages
  - Log errors securely (no PII in logs)
  - Fail securely (deny by default)

- [ ] **Dependencies**
  - Use latest stable versions
  - Monitor for security vulnerabilities
  - Minimize third-party dependencies

## 🛠️ Engineering Requirements

### 1. Architecture
- [ ] **Clean Architecture**
  - Separate concerns: UI, Business Logic, Data Access
  - Use MVVM or MVI pattern with Jetpack Compose
  - Implement proper dependency injection

- [ ] **Modular Design**
  ```
  app/
  ├── ui/           # Compose UI, ViewModels
  ├── domain/       # Business logic, use cases
  ├── data/         # NFC reader, APDU handler
  └── util/         # Parsers, converters
  ```

### 2. Code Quality
- [ ] **Kotlin Best Practices**
  - Use coroutines for async operations
  - Leverage sealed classes for state management
  - Use data classes for models
  - Implement proper null safety

- [ ] **Documentation**
  - Clear comments explaining APDU sequences
  - Document all file IDs and their purposes
  - Explain cryptographic operations
  - Include usage examples

- [ ] **Error Handling**
  - Use Result/Either types for error propagation
  - Implement comprehensive error messages
  - Handle all APDU response codes

### 3. Testing Strategy
- [ ] **Unit Tests**
  - Test APDU command construction
  - Test DG1 parsing logic
  - Test error handling scenarios

- [ ] **Integration Tests**
  - Mock IsoDep for testing NFC flow
  - Test PIN verification logic
  - Test file selection sequences

- [ ] **Manual Testing Checklist**
  - Test with valid PIN
  - Test with invalid PIN
  - Test with locked card
  - Test NFC connection failures
  - Test partial reads

### 4. Performance
- [ ] **Efficient Operations**
  - Use coroutines with Dispatchers.IO for NFC operations
  - Implement proper cancellation handling
  - Avoid blocking UI thread

- [ ] **Memory Management**
  - Clear large byte arrays after use
  - Handle bitmap memory efficiently
  - Implement proper lifecycle management

### 5. User Experience
- [ ] **Clear UI States**
  - Loading state during NFC scan
  - Success state with parsed data
  - Error state with actionable messages
  - Idle state with instructions

- [ ] **Informative Feedback**
  - Clear instructions for card placement
  - Progress indicators during read
  - Helpful error messages
  - Remaining PIN attempts display

### 6. Compatibility
- [ ] **Android Version Support**
  - Minimum SDK: 21 (Android 5.0)
  - Target SDK: Latest stable (34+)
  - Test on multiple Android versions

- [ ] **Device Support**
  - Verify NFC hardware availability
  - Check IsoDep support
  - Handle devices without NFC gracefully

## 📋 APDU Response Codes Reference

| Code  | Meaning | Action |
|-------|---------|--------|
| 9000  | Success | Continue processing |
| 63CX  | Wrong PIN, X attempts left | Show remaining attempts |
| 6982  | Security not satisfied | Request PIN |
| 6983  | Auth method blocked | Card locked, warn user |
| 6A82  | File not found | Check file selection |
| 6A86  | Wrong parameters P1/P2 | Fix APDU command |
| 6700  | Wrong length | Adjust Lc field |

## 📚 Turkish eID Specifications

### File Structure (ICAO LDS)
- **EF.CardAccess**: 011C (public, PACE info)
- **EF.SOD**: 011D (public, signatures)
- **DG1**: 0101 (requires PIN, personal data)
- **DG2**: 0102 (requires PIN, facial image)
- **DG3**: 0103 (requires PIN, fingerprints - restricted)

### AID (Application Identifier)
```
A0 00 00 01 67 45 53 49 44
```

### PIN Requirements
- **PIN1**: 6 digits, user authentication
- **PUK**: 8 digits, unblocking PIN1

## ⚠️ Legal & Ethical Considerations

- [ ] **Authorized Use Only**
  - App should only be used on user's own ID card
  - Include terms of use
  - Add disclaimer about proper usage

- [ ] **Privacy**
  - Comply with KVKK (Turkish GDPR)
  - No unauthorized data collection
  - User consent for data processing

- [ ] **Responsible Disclosure**
  - Report any security vulnerabilities responsibly
  - Don't publish exploit code
  - Contact relevant authorities

## 🎯 Definition of Done

A feature is considered complete when:
1. ✅ All security requirements are met
2. ✅ Code is well-documented and commented
3. ✅ Error handling is comprehensive
4. ✅ UI provides clear feedback
5. ✅ Testing checklist is completed
6. ✅ Code review is passed
7. ✅ README with testing instructions exists

---

**Last Updated**: 2025-11-24
**Project**: Turkish eID NFC Reader
**Technology**: Android Kotlin, NFC, IsoDep, ICAO LDS
