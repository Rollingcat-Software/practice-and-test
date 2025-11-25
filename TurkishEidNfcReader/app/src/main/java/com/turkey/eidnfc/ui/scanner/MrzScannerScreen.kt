package com.turkey.eidnfc.ui.scanner

import android.Manifest
import android.app.Activity
import android.content.Context
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.RotateLeft
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.LifecycleOwner
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import timber.log.Timber
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * MRZ Scanner Screen with camera preview and OCR.
 * Displayed as a fullscreen dialog to avoid layout measurement issues
 * when called from within scrollable containers.
 */
@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
fun MrzScannerScreen(
    onMrzScanned: (ScannedMrzData) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current

    // Hide system bars for immersive fullscreen
    DisposableEffect(Unit) {
        val activity = context as? Activity
        val window = activity?.window
        val decorView = window?.decorView

        if (window != null && decorView != null) {
            val insetsController = WindowCompat.getInsetsController(window, decorView)
            insetsController.hide(WindowInsetsCompat.Type.navigationBars())
            insetsController.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }

        onDispose {
            // Restore system bars when dialog closes
            if (window != null && decorView != null) {
                val insetsController = WindowCompat.getInsetsController(window, decorView)
                insetsController.show(WindowInsetsCompat.Type.navigationBars())
            }
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            dismissOnBackPress = true,
            dismissOnClickOutside = false,
            decorFitsSystemWindows = false
        )
    ) {
        val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            when {
                cameraPermissionState.status.isGranted -> {
                    CameraPreviewContent(
                        onMrzScanned = onMrzScanned,
                        onDismiss = onDismiss
                    )
                }

                cameraPermissionState.status.shouldShowRationale -> {
                    PermissionRationaleContent(
                        onRequestPermission = { cameraPermissionState.launchPermissionRequest() },
                        onDismiss = onDismiss
                    )
                }

                else -> {
                    LaunchedEffect(Unit) {
                        cameraPermissionState.launchPermissionRequest()
                    }
                    PermissionRequestContent(
                        onDismiss = onDismiss
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CameraPreviewContent(
    onMrzScanned: (ScannedMrzData) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var scannedData by remember { mutableStateOf<ScannedMrzData?>(null) }
    var isFlashOn by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var camera by remember { mutableStateOf<androidx.camera.core.Camera?>(null) }
    var detectedText by remember { mutableStateOf<String?>(null) }
    var isScanning by remember { mutableStateOf(true) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // Camera Preview (full screen behind everything)
        CameraPreview(
            context = context,
            lifecycleOwner = lifecycleOwner,
            executor = cameraExecutor,
            onCameraReady = { cam -> camera = cam },
            onMrzDetected = { data ->
                scannedData = data
                isScanning = false
            },
            onTextDetected = { text ->
                detectedText = text
            },
            onError = { error ->
                errorMessage = error
            }
        )

        // Full card overlay with MRZ guide
        IdCardScanningOverlay(
            isScanning = isScanning && scannedData == null,
            detectedText = detectedText,
            hasResult = scannedData != null
        )

        // Top section: close button, title, flash, and live OCR
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .align(Alignment.TopCenter)
        ) {
            // Top bar with close and flash buttons
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Close button
                IconButton(
                    onClick = onDismiss,
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = Color.Black.copy(alpha = 0.5f),
                        contentColor = Color.White
                    )
                ) {
                    Icon(Icons.Default.Close, "Close")
                }

                // Title
                Text(
                    text = "Scan ID Card",
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                // Flash button
                IconButton(
                    onClick = {
                        isFlashOn = !isFlashOn
                        camera?.cameraControl?.enableTorch(isFlashOn)
                    },
                    colors = IconButtonDefaults.iconButtonColors(
                        containerColor = Color.Black.copy(alpha = 0.5f),
                        contentColor = if (isFlashOn) Color(0xFFFFC107) else Color.White
                    )
                ) {
                    Icon(
                        if (isFlashOn) Icons.Default.FlashOn else Icons.Default.FlashOff,
                        contentDescription = if (isFlashOn) "Turn off flash" else "Turn on flash"
                    )
                }
            }

            // Live OCR text display at top (below title bar)
            AnimatedVisibility(
                visible = detectedText != null && scannedData == null,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically()
            ) {
                detectedText?.let { text ->
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        LiveOcrDisplay(text = text)
                    }
                }
            }
        }

        // Bottom content area
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(bottom = 24.dp)
                .padding(horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Scanned data display
            AnimatedVisibility(
                visible = scannedData != null,
                enter = fadeIn() + slideInVertically { it / 2 },
                exit = fadeOut() + slideOutVertically { it / 2 }
            ) {
                scannedData?.let { data ->
                    ScannedDataCard(
                        data = data,
                        onConfirm = {
                            onMrzScanned(data)
                            onDismiss()
                        },
                        onRetry = {
                            scannedData = null
                            detectedText = null
                            isScanning = true
                        }
                    )
                }
            }

            // Scanning hint when no result yet
            AnimatedVisibility(
                visible = scannedData == null,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Black.copy(alpha = 0.8f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp)
                    ) {
                        // Rotation guide animation
                        RotationGuideAnimation()

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = "Rotate card to the left & place in frame",
                            color = Color.White,
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                            fontWeight = FontWeight.Medium
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Back side with MRZ lines (<<<) facing camera",
                            color = Color.White.copy(alpha = 0.7f),
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        }

        // Error message
        errorMessage?.let { error ->
            Snackbar(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
                    .padding(bottom = 180.dp)
            ) {
                Text(error)
            }
            LaunchedEffect(error) {
                kotlinx.coroutines.delay(3000)
                errorMessage = null
            }
        }
    }
}

@Composable
private fun CameraPreview(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    executor: ExecutorService,
    onCameraReady: (androidx.camera.core.Camera) -> Unit,
    onMrzDetected: (ScannedMrzData) -> Unit,
    onTextDetected: (String) -> Unit,
    onError: (String) -> Unit
) {
    val previewView = remember { PreviewView(context) }
    var analyzer: MrzAnalyzer? by remember { mutableStateOf(null) }

    DisposableEffect(Unit) {
        onDispose {
            analyzer?.close()
        }
    }

    AndroidView(
        factory = { ctx ->
            previewView.apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }

            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                val mrzAnalyzer = MrzAnalyzer(
                    onMrzDetected = onMrzDetected,
                    onError = onError,
                    onTextDetected = onTextDetected
                )
                analyzer = mrzAnalyzer

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(executor, mrzAnalyzer)
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                try {
                    cameraProvider.unbindAll()
                    val camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                    onCameraReady(camera)
                } catch (e: Exception) {
                    Timber.e(e, "Camera binding failed")
                    onError("Failed to start camera: ${e.message}")
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}

/**
 * Professional ID card scanning overlay with VERTICAL (rotated) card template.
 *
 * In portrait phone mode, user holds the ID card vertically (rotated 90°)
 * to fill more of the camera view for better OCR accuracy.
 *
 * TD1 card dimensions: 85.6 x 54mm (ISO/IEC 7810 ID-1, credit card size)
 * When rotated: width becomes height, height becomes width
 * MRZ zone: Now on the RIGHT side of the rotated card
 */
@Composable
private fun IdCardScanningOverlay(
    isScanning: Boolean,
    detectedText: String?,
    hasResult: Boolean
) {
    val primaryColor = MaterialTheme.colorScheme.primary
    val successColor = Color(0xFF4CAF50)
    val warningColor = Color(0xFFFFC107)
    val textMeasurer = rememberTextMeasurer()

    // Animations
    val infiniteTransition = rememberInfiniteTransition(label = "scan")

    // Scanning line animation (horizontal sweep for vertical MRZ)
    val scanLinePosition by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "scanLine"
    )

    // Pulse animation for corners
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.6f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(800),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    // Glow animation for MRZ area when detecting
    val mrzGlow by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.6f,
        animationSpec = infiniteRepeatable(
            animation = tween(600),
            repeatMode = RepeatMode.Reverse
        ),
        label = "mrzGlow"
    )

    val borderColor = when {
        hasResult -> successColor
        detectedText != null -> warningColor
        else -> primaryColor
    }

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val density = LocalDensity.current
        val canvasWidth = with(density) { maxWidth.toPx() }
        val canvasHeight = with(density) { maxHeight.toPx() }

        // Card rotated 90° - so original width (85.6mm) becomes height
        // Original aspect ratio: 85.6/54 = 1.585
        // Rotated aspect ratio: 54/85.6 = 0.631 (height/width when vertical)
        val cardAspectRatio = 54f / 85.6f  // Inverted for vertical orientation
        val cardWidth = canvasWidth * 0.85f
        val cardHeight = cardWidth / cardAspectRatio  // Taller than wide
        val cardLeft = (canvasWidth - cardWidth) / 2
        val cardTop = (canvasHeight - cardHeight) / 2 - (canvasHeight * 0.02f)
        val cornerRadius = 16f

        // MRZ zone is on the RIGHT side when card is rotated (was bottom)
        // MRZ takes about 28% of the card's original height = 28% of rotated width
        val mrzWidthRatio = 0.28f
        val mrzWidth = cardWidth * mrzWidthRatio
        val mrzLeft = cardLeft + cardWidth - mrzWidth - 12f  // Right side with padding
        val mrzPadding = 12f

        Canvas(modifier = Modifier.fillMaxSize()) {
            // Semi-transparent overlay
            drawRect(
                color = Color.Black.copy(alpha = 0.7f),
                size = size
            )

            // Clear the card area (cutout)
            drawRoundRect(
                color = Color.Transparent,
                topLeft = Offset(cardLeft, cardTop),
                size = Size(cardWidth, cardHeight),
                cornerRadius = CornerRadius(cornerRadius, cornerRadius),
                blendMode = BlendMode.Clear
            )

            // Card border
            val cardBorderWidth = if (hasResult) 4f else 2.5f
            drawRoundRect(
                color = borderColor.copy(alpha = if (hasResult) 1f else pulseAlpha),
                topLeft = Offset(cardLeft, cardTop),
                size = Size(cardWidth, cardHeight),
                cornerRadius = CornerRadius(cornerRadius, cornerRadius),
                style = Stroke(width = cardBorderWidth)
            )

            // Corner brackets for visibility
            val cornerLength = 40f
            val cornerStroke = 5f
            val cornerOffset = 3f

            // Top-left corner
            drawPath(
                path = Path().apply {
                    moveTo(cardLeft - cornerOffset, cardTop + cornerLength)
                    lineTo(cardLeft - cornerOffset, cardTop - cornerOffset)
                    lineTo(cardLeft + cornerLength, cardTop - cornerOffset)
                },
                color = borderColor,
                style = Stroke(width = cornerStroke, cap = StrokeCap.Round)
            )

            // Top-right corner
            drawPath(
                path = Path().apply {
                    moveTo(cardLeft + cardWidth - cornerLength, cardTop - cornerOffset)
                    lineTo(cardLeft + cardWidth + cornerOffset, cardTop - cornerOffset)
                    lineTo(cardLeft + cardWidth + cornerOffset, cardTop + cornerLength)
                },
                color = borderColor,
                style = Stroke(width = cornerStroke, cap = StrokeCap.Round)
            )

            // Bottom-left corner
            drawPath(
                path = Path().apply {
                    moveTo(cardLeft - cornerOffset, cardTop + cardHeight - cornerLength)
                    lineTo(cardLeft - cornerOffset, cardTop + cardHeight + cornerOffset)
                    lineTo(cardLeft + cornerLength, cardTop + cardHeight + cornerOffset)
                },
                color = borderColor,
                style = Stroke(width = cornerStroke, cap = StrokeCap.Round)
            )

            // Bottom-right corner
            drawPath(
                path = Path().apply {
                    moveTo(cardLeft + cardWidth - cornerLength, cardTop + cardHeight + cornerOffset)
                    lineTo(cardLeft + cardWidth + cornerOffset, cardTop + cardHeight + cornerOffset)
                    lineTo(cardLeft + cardWidth + cornerOffset, cardTop + cardHeight - cornerLength)
                },
                color = borderColor,
                style = Stroke(width = cornerStroke, cap = StrokeCap.Round)
            )

            // MRZ zone indicator (on RIGHT side of rotated card)
            val mrzColor = when {
                hasResult -> successColor
                detectedText != null -> warningColor.copy(alpha = mrzGlow + 0.4f)
                else -> Color.White.copy(alpha = 0.5f)
            }

            // MRZ zone highlight background when detecting
            if (detectedText != null && !hasResult) {
                drawRoundRect(
                    color = warningColor.copy(alpha = mrzGlow * 0.3f),
                    topLeft = Offset(mrzLeft, cardTop + mrzPadding),
                    size = Size(mrzWidth, cardHeight - mrzPadding * 2),
                    cornerRadius = CornerRadius(8f, 8f)
                )
            }

            // MRZ zone border (dashed when scanning, solid when detected)
            drawRoundRect(
                color = mrzColor,
                topLeft = Offset(mrzLeft, cardTop + mrzPadding),
                size = Size(mrzWidth, cardHeight - mrzPadding * 2),
                cornerRadius = CornerRadius(8f, 8f),
                style = Stroke(
                    width = 2f,
                    pathEffect = if (hasResult) null else PathEffect.dashPathEffect(floatArrayOf(10f, 8f), 0f)
                )
            )

            // MRZ placeholder lines (vertical lines for rotated MRZ - 3 lines)
            if (!hasResult) {
                val lineSpacing = mrzWidth / 4

                for (i in 1..3) {
                    val lineX = mrzLeft + (lineSpacing * i)
                    drawLine(
                        color = Color.White.copy(alpha = 0.25f),
                        start = Offset(lineX, cardTop + mrzPadding * 2),
                        end = Offset(lineX, cardTop + cardHeight - mrzPadding * 2),
                        strokeWidth = 1.5f,
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 4f), 0f)
                    )
                }
            }

            // Scanning line animation (horizontal sweep across MRZ zone)
            if (isScanning && !hasResult) {
                val lineX = mrzLeft + (mrzWidth * scanLinePosition)
                drawLine(
                    color = primaryColor.copy(alpha = 0.9f),
                    start = Offset(lineX, cardTop + mrzPadding * 2),
                    end = Offset(lineX, cardTop + cardHeight - mrzPadding * 2),
                    strokeWidth = 3f,
                    cap = StrokeCap.Round
                )
            }

            // Card content hints
            if (!hasResult) {
                // "BACK" indicator on the left side of the card
                val backStyle = TextStyle(
                    color = Color.White.copy(alpha = 0.3f),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
                val backText = "BACK OF ID"
                val backLayout = textMeasurer.measure(backText, backStyle)
                drawText(
                    textLayoutResult = backLayout,
                    topLeft = Offset(
                        cardLeft + 20f,
                        cardTop + cardHeight / 2 - backLayout.size.height / 2
                    )
                )

                // "MRZ" label above the MRZ zone
                val mrzLabelStyle = TextStyle(
                    color = mrzColor.copy(alpha = 0.8f),
                    fontSize = 9.sp,
                    fontWeight = FontWeight.SemiBold
                )
                val mrzLabel = "MRZ"
                val mrzLabelLayout = textMeasurer.measure(mrzLabel, mrzLabelStyle)
                drawText(
                    textLayoutResult = mrzLabelLayout,
                    topLeft = Offset(
                        mrzLeft + (mrzWidth - mrzLabelLayout.size.width) / 2,
                        cardTop + mrzPadding - mrzLabelLayout.size.height - 4f
                    )
                )
            }
        }

        // Success indicator
        AnimatedVisibility(
            visible = hasResult,
            enter = scaleIn(spring(dampingRatio = 0.6f)) + fadeIn(),
            exit = scaleOut() + fadeOut(),
            modifier = Modifier
                .align(Alignment.Center)
                .offset(y = with(density) { (-cardHeight / 4).toDp() })
        ) {
            Box(
                modifier = Modifier
                    .size(72.dp)
                    .background(successColor.copy(alpha = 0.2f), RoundedCornerShape(36.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = "Detected",
                    tint = successColor,
                    modifier = Modifier.size(56.dp)
                )
            }
        }
    }
}

/**
 * Live OCR text display with character-by-character highlighting
 */
@Composable
private fun LiveOcrDisplay(text: String) {
    val lines = text.take(90).chunked(30) // TD1 has 30 chars per line, 3 lines

    Column(
        modifier = Modifier
            .background(
                Color.Black.copy(alpha = 0.7f),
                RoundedCornerShape(8.dp)
            )
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Detecting...",
            color = Color(0xFF4CAF50),
            fontSize = 10.sp,
            fontWeight = FontWeight.Medium
        )
        Spacer(modifier = Modifier.height(4.dp))
        lines.forEach { line ->
            Row {
                line.forEachIndexed { index, char ->
                    val isValid = char.isLetterOrDigit() || char == '<'
                    Text(
                        text = char.toString(),
                        color = if (isValid) Color(0xFF4CAF50) else Color(0xFFFF5722),
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

/**
 * Animated rotation guide showing counter-clockwise (left) rotation.
 * Shows a mini card rotating from horizontal to vertical position.
 */
@Composable
private fun RotationGuideAnimation() {
    val infiniteTransition = rememberInfiniteTransition(label = "rotationGuide")

    // Rotation animation: 0° (horizontal) -> -90° (vertical, rotated left)
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = -90f,
        animationSpec = infiniteRepeatable(
            animation = keyframes {
                durationMillis = 2500
                0f at 0 using LinearEasing           // Start horizontal
                0f at 500 using LinearEasing         // Hold horizontal
                -90f at 1500 using FastOutSlowInEasing // Rotate to vertical
                -90f at 2500 using LinearEasing      // Hold vertical
            },
            repeatMode = RepeatMode.Restart
        ),
        label = "cardRotation"
    )

    // Arrow pulse animation
    val arrowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(600),
            repeatMode = RepeatMode.Reverse
        ),
        label = "arrowPulse"
    )

    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
        modifier = Modifier.fillMaxWidth()
    ) {
        // Mini card that rotates
        Box(
            modifier = Modifier
                .size(width = 48.dp, height = 30.dp)
                .graphicsLayer {
                    rotationZ = rotation
                }
                .background(
                    color = Color.White.copy(alpha = 0.9f),
                    shape = RoundedCornerShape(4.dp)
                )
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.primary,
                    shape = RoundedCornerShape(4.dp)
                ),
            contentAlignment = Alignment.CenterEnd
        ) {
            // MRZ indicator on the card
            Box(
                modifier = Modifier
                    .width(12.dp)
                    .fillMaxHeight()
                    .padding(2.dp)
                    .background(
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.3f),
                        shape = RoundedCornerShape(2.dp)
                    )
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        // Rotation arrow icon
        Icon(
            imageVector = Icons.AutoMirrored.Filled.RotateLeft,
            contentDescription = "Rotate left",
            tint = Color(0xFF4CAF50).copy(alpha = arrowAlpha),
            modifier = Modifier.size(28.dp)
        )

        Spacer(modifier = Modifier.width(12.dp))

        // Result: vertical card
        Box(
            modifier = Modifier
                .size(width = 30.dp, height = 48.dp)
                .background(
                    color = Color(0xFF4CAF50).copy(alpha = 0.2f),
                    shape = RoundedCornerShape(4.dp)
                )
                .border(
                    width = 2.dp,
                    color = Color(0xFF4CAF50),
                    shape = RoundedCornerShape(4.dp)
                ),
            contentAlignment = Alignment.CenterEnd
        ) {
            // MRZ indicator on right side (vertical)
            Box(
                modifier = Modifier
                    .width(8.dp)
                    .fillMaxHeight()
                    .padding(2.dp)
                    .background(
                        color = Color(0xFF4CAF50).copy(alpha = 0.5f),
                        shape = RoundedCornerShape(2.dp)
                    )
            )
        }
    }
}

@Composable
private fun ScannedDataCard(
    data: ScannedMrzData,
    onConfirm: () -> Unit,
    onRetry: () -> Unit
) {
    val verifiedColor = Color(0xFF4CAF50) // Green
    val warningColor = Color(0xFFFF9800) // Orange
    val statusColor = if (data.checksumValid) verifiedColor else warningColor

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Header with verification status
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = statusColor,
                    modifier = Modifier.size(28.dp)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "MRZ Data Detected",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    // Checksum verification badge
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(top = 2.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .background(
                                    color = statusColor.copy(alpha = 0.15f),
                                    shape = RoundedCornerShape(4.dp)
                                )
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text = if (data.checksumValid) "✓ Checksum Verified" else "⚠ Checksum Failed",
                                style = MaterialTheme.typography.labelSmall,
                                color = statusColor,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }
            }

            // Warning message if checksum failed
            if (!data.checksumValid) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Data may contain OCR errors. Please verify before using.",
                    style = MaterialTheme.typography.bodySmall,
                    color = warningColor,
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            warningColor.copy(alpha = 0.1f),
                            RoundedCornerShape(4.dp)
                        )
                        .padding(8.dp)
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Display detected data with highlighting
            DetectedDataRow("Document Number", data.documentNumber, data.checksumValid)
            DetectedDataRow("Date of Birth", formatDateDisplay(data.dateOfBirth), data.checksumValid)
            DetectedDataRow("Date of Expiry", formatDateDisplay(data.dateOfExpiry), data.checksumValid)

            Spacer(modifier = Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedButton(
                    onClick = onRetry,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        Icons.Default.Refresh,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Rescan")
                }
                Button(
                    onClick = onConfirm,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = statusColor
                    )
                ) {
                    Text(if (data.checksumValid) "Use This" else "Use Anyway")
                }
            }
        }
    }
}

@Composable
private fun DetectedDataRow(label: String, value: String, isVerified: Boolean = true) {
    val valueColor = if (isVerified) Color(0xFF4CAF50) else Color(0xFFFF9800)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.Medium,
            fontFamily = FontFamily.Monospace,
            color = valueColor
        )
    }
}

private fun formatDateDisplay(date: String): String {
    if (date.length != 6) return date
    return try {
        val yy = date.substring(0, 2)
        val mm = date.substring(2, 4)
        val dd = date.substring(4, 6)
        val year = if (yy.toInt() > 50) "19$yy" else "20$yy"
        "$dd/$mm/$year"
    } catch (e: Exception) {
        date
    }
}

@Composable
private fun PermissionRationaleContent(
    onRequestPermission: () -> Unit,
    onDismiss: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Default.CameraAlt,
            contentDescription = null,
            modifier = Modifier.size(80.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            "Camera Permission Required",
            style = MaterialTheme.typography.titleLarge,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "To scan the MRZ from your ID card, we need access to your camera.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onRequestPermission) {
            Text("Grant Permission")
        }
        Spacer(modifier = Modifier.height(8.dp))
        TextButton(onClick = onDismiss) {
            Text("Cancel")
        }
    }
}

@Composable
private fun PermissionRequestContent(
    onDismiss: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "Requesting camera permission...",
            style = MaterialTheme.typography.bodyMedium
        )
        Spacer(modifier = Modifier.height(24.dp))
        TextButton(onClick = onDismiss) {
            Text("Cancel")
        }
    }
}
