package org.oralscreen.app

import android.app.Application
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.roundToInt

data class UiState(
    val image: Bitmap? = null,
    val result: Prediction? = null,
    val busy: Boolean = false,
    val error: String? = null,
    // The user chose to see the screening result although the photo was not recognised as oral.
    val showAnyway: Boolean = false,
)

class ScreenViewModel(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    // Loading the models (about 34 MB) takes a moment, so start it right away, off the main thread.
    private val classifier = viewModelScope.async(Dispatchers.Default) { OralClassifier(app) }

    fun showAnyway() = _state.update { it.copy(showAnyway = true) }

    fun analyze(uri: Uri) {
        _state.update { it.copy(busy = true, result = null, error = null, showAnyway = false) }
        viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) { decode(uri) }
                _state.update { it.copy(image = bitmap) }
                val result = withContext(Dispatchers.Default) { classifier.await().predict(bitmap) }
                _state.update { it.copy(result = result, busy = false) }
            } catch (e: Exception) {
                _state.update { it.copy(busy = false, error = "Could not analyse this image: ${e.message}") }
            }
        }
    }

    /** Decodes with EXIF rotation applied and the longest side scaled down to OralClassifier.MAX_SIDE. */
    private fun decode(uri: Uri): Bitmap {
        val source = ImageDecoder.createSource(getApplication<Application>().contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE // getPixels() needs a CPU-side bitmap
            val (w, h) = info.size.width to info.size.height
            val scale = OralClassifier.MAX_SIDE.toFloat() / max(w, h)
            if (scale < 1f) decoder.setTargetSize((w * scale).roundToInt(), (h * scale).roundToInt())
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    override fun onCleared() {
        if (classifier.isCompleted) classifier.getCompleted().close() else classifier.cancel()
    }
}
