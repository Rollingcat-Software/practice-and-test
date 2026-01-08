"""
Biometric Demo - Optimized Real-Time Version
=============================================
A fully-featured biometric demonstration with clean architecture.

Architecture: Hexagonal (Ports & Adapters)
- domain/: Core business logic
- application/: Use cases and services
- infrastructure/: External adapters
- presentation/: UI and camera

Features:
- Face detection (MediaPipe Tasks API)
- Facial landmarks (468 points)
- Demographics analysis (age, gender, emotion)
- Liveness detection (passive + active puzzle)
- Face enrollment & verification
- Card detection (YOLO)
- Video recording
"""

__version__ = "2.0.0"
__author__ = "FIVUCSAS Team"
