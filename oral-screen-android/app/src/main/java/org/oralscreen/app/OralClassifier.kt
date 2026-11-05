package org.oralscreen.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

/** The screening result for one photo, whether it looks like an oral photo, and the on-device inference time. */
data class Prediction(val probability: Float, val threshold: Float, val pOral: Float, val millis: Long) {
    val suspicious get() = probability >= threshold
    val looksOral get() = pOral >= OralClassifier.ORAL_THRESHOLD
}

/**
 * Runs the exported models (oral-screen/src/export_onnx.py and fit_oral_gate.py) with ONNX Runtime on the CPU.
 *
 * preprocess.onnx takes the RGB photo in [0, 1] at any size and resizes it bicubically to 256 x 256 exactly as in
 * training. Its output feeds model.onnx, the MobileViTv2 screening model (P(suspicious)), and gate.onnx, an ImageNet
 * MobileViTv2 with a logistic-regression head that estimates whether the photo shows the oral cavity at all.
 */
class OralClassifier(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    private val preprocess = session(context, "preprocess.onnx")
    private val model = session(context, "model.onnx")
    private val gate = session(context, "gate.onnx")

    private fun session(context: Context, asset: String) =
        env.createSession(context.assets.open(asset).use { it.readBytes() }, options)

    /** The bitmap should already be scaled so its longest side is at most [MAX_SIDE], like the training copies. */
    fun predict(bitmap: Bitmap): Prediction {
        val start = System.nanoTime()
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h).also { bitmap.getPixels(it, 0, w, 0, 0, w, h) }
        val plane = w * h
        val chw = FloatBuffer.allocate(3 * plane)
        for (i in 0 until plane) {
            val p = pixels[i]
            chw.put(i, ((p shr 16) and 0xFF) / 255f)
            chw.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            chw.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        return OnnxTensor.createTensor(env, chw, longArrayOf(1, 3, h.toLong(), w.toLong())).use { image ->
            preprocess.run(mapOf("image" to image)).use { resized ->
                val input = mapOf("resized" to resized[0] as OnnxTensor)
                val prob = model.run(input).use { (it[0].value as FloatArray)[0] }
                val pOral = gate.run(input).use { (it[0].value as FloatArray)[0] }
                Prediction(prob, THRESHOLD, pOral, (System.nanoTime() - start) / 1_000_000)
            }
        }
    }

    override fun close() {
        preprocess.close()
        model.close()
        gate.close()
        options.close()
    }

    companion object {
        const val MAX_SIDE = 640
        const val THRESHOLD = 0.5f
        const val ORAL_THRESHOLD = 0.5f
    }
}
