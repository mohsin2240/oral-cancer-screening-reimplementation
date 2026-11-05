package org.oralscreen.app

import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { AppTheme { ScreenApp() } }
    }
}

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val context = LocalContext.current
    val scheme = when {
        Build.VERSION.SDK_INT >= 31 -> if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        dark -> darkColorScheme(primary = Color(0xFF7FD3E3))
        else -> lightColorScheme(primary = Color(0xFF0E5E6F))
    }
    MaterialTheme(colorScheme = scheme, content = content)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScreenApp(vm: ScreenViewModel = viewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current

    // The system camera app writes the photo to our cache through a FileProvider URI.
    val captureUri: Uri = remember {
        val file = File(context.cacheDir, "photos/capture.jpg").apply { parentFile?.mkdirs() }
        FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    }
    val takePhoto = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
        if (saved) vm.analyze(captureUri)
    }
    val pickPhoto = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) vm.analyze(uri)
    }

    Scaffold(topBar = { TopAppBar(title = { Text("Oral Screen") }) }) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Disclaimer()
            PhotoBox(state)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
                Button(onClick = { takePhoto.launch(captureUri) }, enabled = !state.busy, modifier = Modifier.weight(1f)) {
                    Text("Take photo")
                }
                OutlinedButton(
                    onClick = { pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    enabled = !state.busy,
                    modifier = Modifier.weight(1f),
                ) { Text("Choose photo") }
            }
            ResultCard(state, onShowAnyway = vm::showAnyway)
            About()
        }
    }
}

@Composable
private fun Disclaimer() {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer)) {
        Text(
            "Research prototype, not a medical device. It cannot diagnose oral cancer; a suspicious result means " +
                "the lesion should be examined by a specialist, and a non-suspicious result does not rule anything out.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onTertiaryContainer,
            modifier = Modifier.padding(12.dp),
        )
    }
}

@Composable
private fun PhotoBox(state: UiState) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(300.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant),
        contentAlignment = Alignment.Center,
    ) {
        val image = state.image
        if (image != null) {
            Image(image.asImageBitmap(), contentDescription = "Photo being analysed", contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize())
        } else {
            Text(
                "Take or choose a well-lit, in-focus photo of the lesion or oral cavity.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(24.dp),
            )
        }
    }
}

@Composable
private fun ResultCard(state: UiState, onShowAnyway: () -> Unit) {
    val result = state.result
    when {
        state.busy -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(Modifier.size(24.dp))
            Spacer(Modifier.size(12.dp))
            Text("Analysing on the device…")
        }
        state.error != null -> Text(state.error, color = MaterialTheme.colorScheme.error)
        result != null && !result.looksOral && !state.showAnyway -> NotOralCard(result, onShowAnyway)
        result != null -> {
            val scheme = MaterialTheme.colorScheme
            val (bg, fg) = if (result.suspicious) scheme.errorContainer to scheme.onErrorContainer
            else scheme.secondaryContainer to scheme.onSecondaryContainer
            Card(colors = CardDefaults.cardColors(containerColor = bg, contentColor = fg)) {
                Column(Modifier.padding(16.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        if (result.suspicious) "Suspicious" else "Non-suspicious",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        if (result.suspicious) "Refer for examination by an oral-medicine specialist."
                        else "No suspicious features detected in this photo.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    LinearProgressIndicator(
                        progress = { result.probability },
                        modifier = Modifier.fillMaxWidth(),
                        color = fg,
                        trackColor = fg.copy(alpha = 0.2f),
                    )
                    Text(
                        "Model score %.2f (threshold %.2f) · %d ms on device".format(
                            result.probability, result.threshold, result.millis,
                        ),
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (!result.looksOral) {
                        Text(
                            "Shown on request: this photo was not recognised as an oral photo, so the result is " +
                                "unreliable.",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
            }
        }
    }
}

/** The gate did not recognise an oral photo. Not an all-clear: ask for a retake, but let the user see the result. */
@Composable
private fun NotOralCard(result: Prediction, onShowAnyway: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(16.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Not recognised as an oral photo", style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold)
            Text(
                "The screening model only works on close-up photos of the lips, tongue or inside of the mouth. " +
                    "Retake the photo close up, in good light and in focus. This is not a screening result: " +
                    "if there is a lesion, it still needs to be checked.",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text("Oral-photo confidence %.2f (needs %.2f)".format(result.pOral, OralClassifier.ORAL_THRESHOLD),
                style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = onShowAnyway, contentPadding = PaddingValues(0.dp)) {
                Text("Show the screening result anyway")
            }
        }
    }
}

@Composable
private fun About() {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("About the model", style = MaterialTheme.typography.titleSmall)
        Text(
            "MobileViTv2 (4.4 M parameters) trained with geometric augmentation and weighted cross-entropy, " +
                "re-implementing \"Robust Lightweight Deep Learning Models for Oral Cancer Screening\" on about 420 " +
                "public images. On its held-out test set (43 suspicious, 41 non-suspicious images) this checkpoint " +
                "reached 81.4% sensitivity and 92.7% specificity. A second network first checks that the photo " +
                "shows the mouth (it accepted 98.8% of the test photos). Photos never leave the phone.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
