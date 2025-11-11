package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.*;

public class LamaInference {
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;

    public LamaInference(String path) throws OrtException {
        EnumSet<OrtProvider> providers = OrtEnvironment.getEnvironment().getAvailableProviders();
        System.out.println("Available ONNX Runtime providers:");
        for (OrtProvider provider : providers) {
            System.out.println(" - " + provider.getName());
        }
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = env.createSession(path, options);
    }

    public Mat inpaint(Mat image, Mat mask) throws OrtException {
        //Imgproc.resize(image, image, new Size(512, 512));
        //Imgproc.resize(mask, mask, new Size(512, 512), 0, 0, Imgproc.INTER_NEAREST);

        try (OrtSession.Result results = session.run(prepareImgAndMask(image, mask, 8))) {
            float[][][][] output = (float[][][][]) results.get(0).getValue();
            return convertOutputToMat(output);
        }
    }

    private Mat convertOutputToMat(float[][][][] output) {
        int outC = output[0].length;
        int outH = output[0][0].length;
        int outW = output[0][0][0].length;

        Mat outImg = new Mat(outH, outW, CvType.CV_8UC3);
        byte[] outData = new byte[outH * outW * outC];

        // 假设模型输出在 [0, 1] 范围内，直接乘以255
        int idx = 0;
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                for (int c = 0; c < outC; c++) {
                    float v = output[0][c][y][x];
                    // 直接使用固定范围，避免动态归一化
                    int iv = (int) (v * 255.0f);
                    iv = Math.max(0, Math.min(255, iv));
                    outData[idx++] = (byte) iv;
                }
            }
        }
        outImg.put(0, 0, outData);
        Imgproc.cvtColor(outImg, outImg, Imgproc.COLOR_BGR2RGB);
        return outImg;
    }

    private Map<String, OnnxTensor> prepareImgAndMask(Mat image, Mat mask, int padOutToModulo) throws OrtException {
        image = padImgToModulo(image, padOutToModulo);
        mask = padImgToModulo(mask, padOutToModulo);

        int h = image.rows(), w = image.cols();

        float[] imgCHW = getImage(image, false);
        OnnxTensor imgTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imgCHW), new long[]{1, 3, h, w});

        float[] maskCHW = getImage(mask, true);
        for (int i = 0; i < maskCHW.length; i++) maskCHW[i] = maskCHW[i] > 0 ? 1.0f : 0.0f;
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(maskCHW), new long[]{1, 1, h, w});

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("image", imgTensor);
        inputs.put("mask", maskTensor);
        return inputs;
    }

    private float[] getImage(Mat img, boolean isMask) {
        Mat converted = new Mat();
        if (!isMask && img.channels() == 3) {
            Imgproc.cvtColor(img, converted, Imgproc.COLOR_BGR2RGB);
        } else {
            converted = img.clone();
        }

        int h = converted.rows(), w = converted.cols(), c = converted.channels();
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

    private int ceilModulo(int x, int mod) {
        return (x % mod == 0) ? x : ((x / mod + 1) * mod);
    }

    private Mat padImgToModulo(Mat img, int mod) {
        int h = img.rows(), w = img.cols();
        int outH = ceilModulo(h, mod), outW = ceilModulo(w, mod);
        Mat dst = new Mat();
        Core.copyMakeBorder(img, dst, 0, outH - h, 0, outW - w, Core.BORDER_REFLECT);
        return dst;
    }
}
