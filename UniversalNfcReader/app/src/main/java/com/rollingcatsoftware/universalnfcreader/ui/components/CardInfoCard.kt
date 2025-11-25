package com.rollingcatsoftware.universalnfcreader.ui.components

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.Contactless
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DirectionsBus
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Nfc
import androidx.compose.material.icons.filled.School
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.rollingcatsoftware.universalnfcreader.domain.model.CardData
import com.rollingcatsoftware.universalnfcreader.domain.model.CardType
import com.rollingcatsoftware.universalnfcreader.domain.model.GenericCardData
import com.rollingcatsoftware.universalnfcreader.domain.model.Iso15693Data
import com.rollingcatsoftware.universalnfcreader.domain.model.IstanbulkartData
import com.rollingcatsoftware.universalnfcreader.domain.model.MifareClassicData
import com.rollingcatsoftware.universalnfcreader.domain.model.MifareDesfireData
import com.rollingcatsoftware.universalnfcreader.domain.model.MifareUltralightData
import com.rollingcatsoftware.universalnfcreader.domain.model.NdefData
import com.rollingcatsoftware.universalnfcreader.domain.model.SectorData
import com.rollingcatsoftware.universalnfcreader.domain.model.StudentCardData
import com.rollingcatsoftware.universalnfcreader.domain.model.TurkishEidData
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Card displaying information about a read NFC card.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun CardInfoCard(
    cardData: CardData,
    modifier: Modifier = Modifier,
    isCompact: Boolean = false
) {
    var isExpanded by remember { mutableStateOf(!isCompact) }
    val clipboardManager = LocalClipboardManager.current

    Card(
        modifier = modifier
            .fillMaxWidth()
            .animateContentSize(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = getCardIcon(cardData.cardType),
                    contentDescription = null,
                    modifier = Modifier.size(40.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = cardData.cardType.displayName,
                        style = MaterialTheme.typography.titleMedium
                    )
                    Text(
                        text = formatTimestamp(cardData.readTimestamp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (isCompact) {
                    IconButton(onClick = { isExpanded = !isExpanded }) {
                        Icon(
                            imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = if (isExpanded) "Collapse" else "Expand"
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // UID with copy button
            OutlinedCard(
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "UID",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = cardData.uid,
                            style = MaterialTheme.typography.bodyMedium,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    IconButton(
                        onClick = {
                            clipboardManager.setText(AnnotatedString(cardData.uid))
                        }
                    ) {
                        Icon(
                            imageVector = Icons.Default.ContentCopy,
                            contentDescription = "Copy UID",
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }

            // Technologies
            if (cardData.technologies.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    cardData.technologies.forEach { tech ->
                        AssistChip(
                            onClick = { },
                            label = { Text(tech, style = MaterialTheme.typography.labelSmall) }
                        )
                    }
                }
            }

            // Expanded content
            if (isExpanded || !isCompact) {
                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider()
                Spacer(modifier = Modifier.height(12.dp))

                // Card-specific details
                CardSpecificContent(cardData)
            }
        }
    }
}

@Composable
private fun CardSpecificContent(cardData: CardData) {
    when (cardData) {
        is IstanbulkartData -> IstanbulkartContent(cardData)
        is MifareDesfireData -> DesfireContent(cardData)
        is MifareClassicData -> MifareClassicContent(cardData)
        is MifareUltralightData -> UltralightContent(cardData)
        is NdefData -> NdefContent(cardData)
        is StudentCardData -> StudentCardContent(cardData)
        is Iso15693Data -> Iso15693Content(cardData)
        is TurkishEidData -> TurkishEidContent(cardData)
        is GenericCardData -> GenericContent(cardData)
    }
}

@Composable
private fun IstanbulkartContent(data: IstanbulkartData) {
    val clipboardManager = LocalClipboardManager.current

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // DESFire Version Details - Always shown
        data.desfireVersion?.let { version ->
            Text(
                text = "═══ CARD INFORMATION ═══",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )
            InfoRow(
                "Hardware Version",
                "${version.hardwareMajorVersion}.${version.hardwareMinorVersion}"
            )
            InfoRow(
                "Software Version",
                "${version.softwareMajorVersion}.${version.softwareMinorVersion}"
            )
            InfoRow(
                "Hardware Vendor ID",
                "0x${
                    version.hardwareVendorId.toString(16).uppercase().padStart(2, '0')
                } (${if (version.hardwareVendorId == 0x04) "NXP" else "Unknown"})"
            )
            InfoRow(
                "Hardware Type",
                "0x${version.hardwareType.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Hardware SubType",
                "0x${version.hardwareSubType.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Hardware Protocol",
                "0x${version.hardwareProtocol.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Software Vendor ID",
                "0x${version.softwareVendorId.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Software Type",
                "0x${version.softwareType.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Software SubType",
                "0x${version.softwareSubType.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Software Protocol",
                "0x${version.softwareProtocol.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow(
                "Storage Size Code",
                "0x${version.hardwareStorageSize.toString(16).uppercase().padStart(2, '0')}"
            )
            InfoRow("Storage Size", "${version.storageSizeBytes} bytes")
            InfoRow("Card UID (from version)", version.uid.joinToString("") { "%02X".format(it) })
            InfoRow("Batch Number", version.batchNumber.joinToString("") { "%02X".format(it) })
            InfoRow("Production Week", version.productionWeek.toString())
            InfoRow("Production Year", "20${version.productionYear.toString().padStart(2, '0')}")
        }

        data.freeMemory?.let { mem ->
            InfoRow("Free Memory", "$mem bytes")
        }

        // Applications - Always shown
        if (data.applicationIds.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "═══ APPLICATIONS (${data.applicationIds.size}) ═══",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )
            data.applicationIds.forEachIndexed { index, appId ->
                InfoRow("Application ${index + 1}", "0x$appId")
            }
        }

        // Raw Data Map - Always shown
        if (data.rawData.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "═══ RAW DATA ═══",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )
            data.rawData.forEach { (key, value) ->
                InfoRow(key, value.toString())
            }
        }

        // Copy All Data Button
        Spacer(modifier = Modifier.height(8.dp))
        TextButton(
            onClick = {
                val allData = buildIstanbulkartDataString(data)
                clipboardManager.setText(AnnotatedString(allData))
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(
                Icons.Default.ContentCopy,
                contentDescription = null,
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text("Copy All Data to Clipboard")
        }

        Text(
            text = "Note: Balance and transaction history require proprietary IBB keys",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun buildIstanbulkartDataString(data: IstanbulkartData): String {
    return buildString {
        appendLine("=== ISTANBULKART DATA ===")
        appendLine("UID: ${data.uid}")
        appendLine("Card Type: ${data.cardType.displayName}")
        appendLine("Technologies: ${data.technologies.joinToString(", ")}")
        appendLine()
        data.desfireVersion?.let { v ->
            appendLine("=== DESFIRE VERSION ===")
            appendLine("Hardware: ${v.hardwareMajorVersion}.${v.hardwareMinorVersion}")
            appendLine("Software: ${v.softwareMajorVersion}.${v.softwareMinorVersion}")
            appendLine("Hardware Vendor ID: 0x${v.hardwareVendorId.toString(16).uppercase()}")
            appendLine("Hardware Type: 0x${v.hardwareType.toString(16).uppercase()}")
            appendLine("Hardware SubType: 0x${v.hardwareSubType.toString(16).uppercase()}")
            appendLine("Hardware Protocol: 0x${v.hardwareProtocol.toString(16).uppercase()}")
            appendLine("Software Vendor ID: 0x${v.softwareVendorId.toString(16).uppercase()}")
            appendLine("Software Type: 0x${v.softwareType.toString(16).uppercase()}")
            appendLine("Software SubType: 0x${v.softwareSubType.toString(16).uppercase()}")
            appendLine("Software Protocol: 0x${v.softwareProtocol.toString(16).uppercase()}")
            appendLine(
                "Storage Size: ${v.storageSizeBytes} bytes (code: 0x${
                    v.hardwareStorageSize.toString(
                        16
                    ).uppercase()
                })"
            )
            appendLine("Card UID: ${v.uid.joinToString("") { "%02X".format(it) }}")
            appendLine("Batch Number: ${v.batchNumber.joinToString("") { "%02X".format(it) }}")
            appendLine(
                "Production: Week ${v.productionWeek}, 20${
                    v.productionYear.toString().padStart(2, '0')
                }"
            )
            appendLine()
        }
        data.freeMemory?.let { appendLine("Free Memory: $it bytes") }
        if (data.applicationIds.isNotEmpty()) {
            appendLine()
            appendLine("=== APPLICATIONS ===")
            data.applicationIds.forEachIndexed { i, id -> appendLine("App ${i + 1}: 0x$id") }
        }
        appendLine()
        appendLine("=== RAW DATA ===")
        data.rawData.forEach { (k, v) -> appendLine("$k: $v") }
    }
}

@Composable
private fun DesfireContent(data: MifareDesfireData) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        data.version?.let { version ->
            InfoRow(
                "Hardware Version",
                "${version.hardwareMajorVersion}.${version.hardwareMinorVersion}"
            )
            InfoRow(
                "Software Version",
                "${version.softwareMajorVersion}.${version.softwareMinorVersion}"
            )
            InfoRow("Storage Size", "${version.storageSizeBytes} bytes")
        }
        InfoRow("Applications", data.applicationIds.size.toString())
        data.freeMemory?.let { InfoRow("Free Memory", "$it bytes") }
    }
}

@Composable
private fun MifareClassicContent(data: MifareClassicData) {
    val clipboardManager = LocalClipboardManager.current

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = "═══ CARD INFORMATION ═══",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary
        )
        InfoRow("Size", "${data.size} bytes")
        InfoRow("Total Sectors", data.sectorCount.toString())
        InfoRow("Accessible Sectors", data.accessibleSectors.toString())
        InfoRow("Total Blocks", data.blockCount.toString())
        InfoRow("SAK", "0x${data.sak.toString(16).uppercase().padStart(2, '0')}")
        if (data.atqa.isNotEmpty()) {
            InfoRow("ATQA", data.atqa.joinToString("") { "%02X".format(it) })
        }

        // Always show all sector data
        if (data.sectorsRead.isNotEmpty()) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "═══ BLOCK DATA (${data.sectorsRead.size} sectors) ═══",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )

            data.sectorsRead.forEach { sector ->
                SectorDataCard(sector)
            }

            // Copy All Data Button
            Spacer(modifier = Modifier.height(8.dp))
            TextButton(
                onClick = {
                    val allData = buildMifareClassicDataString(data)
                    clipboardManager.setText(AnnotatedString(allData))
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    Icons.Default.ContentCopy,
                    contentDescription = null,
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text("Copy All Data to Clipboard")
            }
        }

        // Show protected sectors info
        val protectedSectors = (0 until data.sectorCount).filter { sectorNum ->
            data.sectorsRead.none { it.sectorNumber == sectorNum }
        }
        if (protectedSectors.isNotEmpty()) {
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Protected sectors (no default key): ${protectedSectors.joinToString(", ")}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error
            )
        }
    }
}

@Composable
private fun SectorDataCard(sector: SectorData) {
    OutlinedCard(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            Text(
                text = "══ Sector ${sector.sectorNumber} (Key ${sector.keyType.name}) ══",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )
            sector.accessBits?.let { bits ->
                Text(
                    text = "Access Bits: ${bits.joinToString("") { "%02X".format(it) }}",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            sector.blocks.forEachIndexed { index, block ->
                val blockNum = sector.sectorNumber * 4 + index
                val hexString = block.joinToString(" ") { "%02X".format(it) }
                val asciiString = block.map { b ->
                    val c = b.toInt() and 0xFF
                    if (c in 0x20..0x7E) c.toChar() else '.'
                }.joinToString("")

                Text(
                    text = "Block $blockNum:",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = hexString,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    text = "ASCII: $asciiString",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.tertiary
                )
                Spacer(modifier = Modifier.height(4.dp))
            }
        }
    }
}

private fun buildMifareClassicDataString(data: MifareClassicData): String {
    return buildString {
        appendLine("=== MIFARE CLASSIC DATA ===")
        appendLine("UID: ${data.uid}")
        appendLine("Card Type: ${data.cardType.displayName}")
        appendLine("Technologies: ${data.technologies.joinToString(", ")}")
        appendLine("Size: ${data.size} bytes")
        appendLine("SAK: 0x${data.sak.toString(16).uppercase().padStart(2, '0')}")
        appendLine("ATQA: ${data.atqa.joinToString("") { "%02X".format(it) }}")
        appendLine("Total Sectors: ${data.sectorCount}")
        appendLine("Accessible Sectors: ${data.accessibleSectors}")
        appendLine("Total Blocks: ${data.blockCount}")
        appendLine()
        appendLine("=== BLOCK DATA ===")
        data.sectorsRead.forEach { sector ->
            appendLine()
            appendLine("--- Sector ${sector.sectorNumber} (Key: ${sector.keyType.name}) ---")
            sector.accessBits?.let { bits ->
                appendLine("Access Bits: ${bits.joinToString("") { "%02X".format(it) }}")
            }
            sector.blocks.forEachIndexed { index, block ->
                val blockNum = sector.sectorNumber * 4 + index
                val hexString = block.joinToString(" ") { "%02X".format(it) }
                val asciiString = block.map { b ->
                    val c = b.toInt() and 0xFF
                    if (c in 0x20..0x7E) c.toChar() else '.'
                }.joinToString("")
                appendLine("Block $blockNum: $hexString")
                appendLine("  ASCII: $asciiString")
            }
        }
        val protectedSectors = (0 until data.sectorCount).filter { sectorNum ->
            data.sectorsRead.none { it.sectorNumber == sectorNum }
        }
        if (protectedSectors.isNotEmpty()) {
            appendLine()
            appendLine("=== PROTECTED SECTORS ===")
            appendLine("Sectors ${protectedSectors.joinToString(", ")} could not be read (no default key)")
        }
    }
}

@Composable
private fun UltralightContent(data: MifareUltralightData) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        InfoRow("Type", data.ultralightType.name.replace("_", " "))
        InfoRow("Pages Read", data.pageCount.toString())
        data.ndefMessage?.let { ndef ->
            InfoRow("NDEF Content", ndef.take(100))
        }
    }
}

@Composable
private fun NdefContent(data: NdefData) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        InfoRow("Records", data.records.size.toString())
        InfoRow("Writable", if (data.isWritable) "Yes" else "No")
        InfoRow("Size", "${data.usedSize} / ${data.maxSize} bytes")

        data.records.forEachIndexed { index, record ->
            record.payloadAsString?.let { content ->
                Spacer(modifier = Modifier.height(4.dp))
                OutlinedCard(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(8.dp)) {
                        Text(
                            text = "Record ${index + 1}",
                            style = MaterialTheme.typography.labelSmall
                        )
                        Text(
                            text = content.take(200),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StudentCardContent(data: StudentCardData) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        data.studentId?.let { InfoRow("Student ID", it) }
        data.studentName?.let { InfoRow("Name", it) }
        data.universityName?.let { InfoRow("University", it) }
        data.department?.let { InfoRow("Department", it) }
        InfoRow("Sectors Read", "${data.sectorsRead} / ${data.totalSectors}")
    }
}

@Composable
private fun Iso15693Content(data: Iso15693Data) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        InfoRow("Manufacturer", data.manufacturer)
        InfoRow("Block Size", "${data.blockSize} bytes")
        InfoRow("Block Count", data.blockCount.toString())
        InfoRow("Blocks Read", data.blocks.size.toString())
    }
}

@Composable
private fun TurkishEidContent(data: TurkishEidData) {
    val clipboardManager = LocalClipboardManager.current
    var showPhotoPopup by remember { mutableStateOf(false) }

    // Photo Popup Viewer
    ImagePopupViewer(
        bitmap = data.photo,
        isVisible = showPhotoPopup,
        onDismiss = { showPhotoPopup = false }
    )

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (data.bacSuccessful) {
            // Photo and Name Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.Top
            ) {
                // Photo - Clickable to open popup
                data.photo?.let { bitmap ->
                    Surface(
                        modifier = Modifier
                            .size(100.dp)
                            .clickable { showPhotoPopup = true },
                        shape = RoundedCornerShape(8.dp),
                        shadowElevation = 2.dp
                    ) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "ID Photo - Tap to enlarge",
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp)),
                            contentScale = ContentScale.Crop
                        )
                    }
                } ?: run {
                    // Placeholder if no photo
                    Surface(
                        modifier = Modifier.size(100.dp),
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant
                    ) {
                        Icon(
                            imageVector = Icons.Default.Badge,
                            contentDescription = null,
                            modifier = Modifier
                                .padding(24.dp)
                                .size(52.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                // Name and basic info
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "${data.givenNames} ${data.surname}".trim().ifEmpty { "Unknown" },
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                    if (data.nationality.isNotEmpty()) {
                        Text(
                            text = data.nationality,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    if (data.personalNumber.isNotEmpty()) {
                        AssistChip(
                            onClick = { clipboardManager.setText(AnnotatedString(data.personalNumber)) },
                            label = {
                                Text(
                                    text = "TCKN: ${data.personalNumber}",
                                    style = MaterialTheme.typography.labelSmall,
                                    fontFamily = FontFamily.Monospace
                                )
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Personal Details Section
            Text(
                text = "PERSONAL INFORMATION",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )

            data.surname.takeIf { it.isNotEmpty() }?.let { InfoRow("Surname", it) }
            data.givenNames.takeIf { it.isNotEmpty() }?.let { InfoRow("Given Names", it) }
            data.dateOfBirth.takeIf { it.isNotEmpty() }?.let { InfoRow("Date of Birth", it) }
            data.sex.takeIf { it.isNotEmpty() }
                ?.let { InfoRow("Sex", if (it == "M") "Male" else if (it == "F") "Female" else it) }
            data.nationality.takeIf { it.isNotEmpty() }?.let { InfoRow("Nationality", it) }

            Spacer(modifier = Modifier.height(8.dp))

            // Document Details Section
            Text(
                text = "DOCUMENT INFORMATION",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )

            data.documentNumber.takeIf { it.isNotEmpty() }?.let { InfoRow("Document Number", it) }
            data.dateOfExpiry.takeIf { it.isNotEmpty() }?.let { InfoRow("Expiry Date", it) }
            data.personalNumber.takeIf { it.isNotEmpty() }
                ?.let { InfoRow("Personal Number (TCKN)", it) }

            // Authentication Status
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                AssistChip(
                    onClick = { },
                    label = { Text("BAC: Success", style = MaterialTheme.typography.labelSmall) }
                )
                data.sodValid?.let { valid ->
                    AssistChip(
                        onClick = { },
                        label = {
                            Text(
                                "SOD: ${if (valid) "Valid" else "Invalid"}",
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                    )
                }
            }

            // Copy All Data Button
            Spacer(modifier = Modifier.height(8.dp))
            TextButton(
                onClick = {
                    val allData = buildTurkishEidDataString(data)
                    clipboardManager.setText(AnnotatedString(allData))
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(
                    Icons.Default.ContentCopy,
                    contentDescription = null,
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text("Copy All Data to Clipboard")
            }

        } else {
            // Not authenticated yet
            OutlinedCard(
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Badge,
                        contentDescription = null,
                        modifier = Modifier.size(48.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Turkish eID Card Detected",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "BAC authentication required to read personal data.\nEnter MRZ information (document number, date of birth, expiry date) to authenticate.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

private fun buildTurkishEidDataString(data: TurkishEidData): String {
    return buildString {
        appendLine("=== TURKISH eID DATA ===")
        appendLine("UID: ${data.uid}")
        appendLine("Card Type: ${data.cardType.displayName}")
        appendLine("Technologies: ${data.technologies.joinToString(", ")}")
        appendLine()
        appendLine("=== PERSONAL INFORMATION ===")
        appendLine("Surname: ${data.surname}")
        appendLine("Given Names: ${data.givenNames}")
        appendLine("Date of Birth: ${data.dateOfBirth}")
        appendLine("Sex: ${data.sex}")
        appendLine("Nationality: ${data.nationality}")
        appendLine("Personal Number (TCKN): ${data.personalNumber}")
        appendLine()
        appendLine("=== DOCUMENT INFORMATION ===")
        appendLine("Document Number: ${data.documentNumber}")
        appendLine("Date of Expiry: ${data.dateOfExpiry}")
        appendLine()
        appendLine("=== AUTHENTICATION ===")
        appendLine("BAC: ${if (data.bacSuccessful) "Success" else "Not Performed"}")
        data.sodValid?.let { appendLine("SOD: ${if (it) "Valid" else "Invalid"}") }
        appendLine()
        appendLine("Photo: ${if (data.photo != null) "Available (${data.photo.width}x${data.photo.height})" else "Not Available"}")
    }
}

@Composable
private fun GenericContent(data: GenericCardData) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        data.sak?.let { InfoRow("SAK", "0x${it.toString(16).uppercase()}") }
        data.atqa?.let { InfoRow("ATQA", it.joinToString("") { b -> "%02X".format(b) }) }
        data.ats?.let { InfoRow("ATS", it.joinToString("") { b -> "%02X".format(b) }) }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace
        )
    }
}

private fun getCardIcon(cardType: CardType): ImageVector {
    return when (cardType) {
        CardType.ISTANBULKART -> Icons.Default.DirectionsBus
        CardType.STUDENT_CARD_CLASSIC,
        CardType.STUDENT_CARD_DESFIRE -> Icons.Default.School

        CardType.MIFARE_CLASSIC_1K,
        CardType.MIFARE_CLASSIC_4K,
        CardType.MIFARE_DESFIRE,
        CardType.MIFARE_ULTRALIGHT,
        CardType.MIFARE_ULTRALIGHT_C -> Icons.Default.Memory

        CardType.TURKISH_EID -> Icons.Default.Badge
        CardType.NDEF -> Icons.Default.Nfc
        else -> Icons.Default.Contactless
    }
}

private fun formatTimestamp(timestamp: Long): String {
    val formatter = SimpleDateFormat("dd MMM yyyy, HH:mm:ss", Locale.getDefault())
    return formatter.format(Date(timestamp))
}
