package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Map;

public class LamaInpaintDynamicSingleton {

    // ----------------------------
    // 单例缓存模型
    // ----------------------------
    public static class ModelCache {
        private static OrtEnvironment env;
        private static OrtSession session;

        public static void init(String modelPath) throws OrtException {
            if (session == null) {
                env = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                session = env.createSession(modelPath, opts);
                System.out.println("ONNX model loaded.");
            }
        }

        public static OrtSession getSession() {
            return session;
        }

        public static void close() throws OrtException {
            if (session != null) session.close();
            if (env != null) env.close();
        }
    }

    // ----------------------------
    // 工具函数
    // ----------------------------
    public static Mat loadImage(String path) {
        Mat img = Imgcodecs.imread(path, Imgcodecs.IMREAD_COLOR);
        img.convertTo(img, CvType.CV_32FC3, 1.0 / 255.0);
        return img;
    }

    public static Mat loadMask(String path) {
        Mat mask = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
        Imgproc.threshold(mask, mask, 127, 1.0, Imgproc.THRESH_BINARY);
        mask.convertTo(mask, CvType.CV_32FC1);
        return mask;
    }

    public static Mat resizeWithAspect(Mat img, int maxSize) {
        int h = img.rows();
        int w = img.cols();
        double scale = Math.min((double)maxSize / h, (double)maxSize / w);
        if (scale >= 1.0) return img.clone();
        int newW = (int)(w * scale);
        int newH = (int)(h * scale);
        Mat resized = new Mat();
        Imgproc.resize(img, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_CUBIC);
        return resized;
    }

    public static Mat addBorder(Mat img, int divisor) {
        int h = img.rows();
        int w = img.cols();
        int newH = ((h + divisor - 1) / divisor) * divisor;
        int newW = ((w + divisor - 1) / divisor) * divisor;
        int padH = newH - h;
        int padW = newW - w;
        Mat padded = new Mat();
        Core.copyMakeBorder(img, padded, 0, padH, 0, padW, Core.BORDER_REFLECT);
        return padded;
    }

    public static OnnxTensor matToTensor(OrtEnvironment env, Mat img) throws OrtException {
        int h = img.rows();
        int w = img.cols();
        int c = img.channels();

        float[] data = new float[h * w * c];
        img.get(0, 0, data);

        // HWC -> CHW
        float[] chw = new float[data.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    chw[ch * h * w + y * w + x] = data[y * w * c + x * c + ch];
                }
            }
        }

        long[] shape = new long[]{1, c, h, w};
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape);
    }

    public static Mat tensorToMat(OnnxTensor tensor, int outH, int outW) throws OrtException {
        float[][][][] arr = (float[][][][]) tensor.getValue();
        float[][][] outCHW = arr[0];
        int c = outCHW.length;
        Mat out = new Mat(outH, outW, CvType.CV_8UC3);

        byte[] buffer = new byte[outH * outW * c];
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                for (int ch = 0; ch < c; ch++) {
                    float v = outCHW[ch][y][x];
                    buffer[(y * outW + x) * c + ch] = (byte)Math.max(0, Math.min(255, v * 255.0f));
                }
            }
        }
        out.put(0, 0, buffer);
        return out;
    }

    // ----------------------------
    // 核心 inpaint 函数
    // ----------------------------
    public static Mat inpaintONNX(Mat img, Mat mask, int maxSize) throws Exception {
        OrtSession session = ModelCache.getSession();
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        // 缩放 + 添加边界
        Mat imgResized = resizeWithAspect(img, maxSize);
        Mat maskResized = resizeWithAspect(mask, maxSize);
        Mat imgPadded = addBorder(imgResized, 8);
        Mat maskPadded = addBorder(maskResized, 8);

        OnnxTensor imgTensor = matToTensor(env, imgPadded);
        OnnxTensor maskTensor = matToTensor(env, maskPadded);

        Map<String, OnnxTensor> inputs = Map.of("image", imgTensor, "mask", maskTensor);
        OrtSession.Result result = session.run(inputs);

        Mat outMat = tensorToMat((OnnxTensor) result.get(0), imgResized.rows(), imgResized.cols());

        Mat finalOut = new Mat();
        Imgproc.resize(outMat, finalOut, new Size(img.cols(), img.rows()), 0, 0, Imgproc.INTER_CUBIC);

        //Imgcodecs.imwrite(outputPath, finalOut);
        //System.out.println("Inpainted image saved to " + outputPath);

        imgTensor.close();
        maskTensor.close();
        return finalOut;
    }
}
