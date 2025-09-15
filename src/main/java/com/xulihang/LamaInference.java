package com.xulihang;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.*;

public class LamaInference {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // 将 Mat 转换为 NCHW 格式 float[]
    public static float[] getImage(Mat img, boolean isMask) {
        Mat converted = new Mat();
        if (!isMask && img.channels() == 3) {
            Imgproc.cvtColor(img, converted, Imgproc.COLOR_BGR2RGB);
        } else {
            converted = img.clone();
        }

        int h = converted.rows();
        int w = converted.cols();
        int c = converted.channels();

        float[] chw = new float[c * h * w];
        byte[] data = new byte[(int) (converted.total() * converted.channels())];
        converted.get(0, 0, data);

        int idx = 0;
        for (int ci = 0; ci < c; ci++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int offset = (y * w + x) * c + ci;
                    float val = (data[offset] & 0xFF) / 255.0f;
                    chw[idx++] = val;
                }
            }
        }
        return chw;
    }

    public static Map<String, OnnxTensor> prepareImgAndMask(Mat image, Mat mask,
                                                            OrtEnvironment env, OrtSession session,
                                                            int padOutToModulo) throws OrtException {
        image = padImgToModulo(image, padOutToModulo);
        mask = padImgToModulo(mask, padOutToModulo);

        int h = image.rows();
        int w = image.cols();

        // image: 3 通道
        float[] imgCHW = getImage(image, false);
        long[] imgShape = new long[]{1, 3, h, w};
        OnnxTensor imgTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imgCHW), imgShape);

        // mask: 1 通道
        float[] maskCHW = getImage(mask, true);
        for (int i = 0; i < maskCHW.length; i++) {
            maskCHW[i] = maskCHW[i] > 0 ? 1.0f : 0.0f;
        }
        long[] maskShape = new long[]{1, 1, h, w};
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(maskCHW), maskShape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("image", imgTensor);
        inputs.put("mask", maskTensor);
        return inputs;
    }

    public static int ceilModulo(int x, int mod) {
        return (x % mod == 0) ? x : ((x / mod + 1) * mod);
    }

    // 缩放图像
    public static Mat scaleImage(Mat img, double factor, int interpolation) {
        Mat dst = new Mat();
        Imgproc.resize(img, dst, new Size(img.cols() * factor, img.rows() * factor), 0, 0, interpolation);
        return dst;
    }

    // 填充到指定倍数
    public static Mat padImgToModulo(Mat img, int mod) {
        int h = img.rows();
        int w = img.cols();
        int outH = ceilModulo(h, mod);
        int outW = ceilModulo(w, mod);

        Mat dst = new Mat();
        Core.copyMakeBorder(img, dst, 0, outH - h, 0, outW - w, Core.BORDER_REFLECT);
        return dst;
    }
}
