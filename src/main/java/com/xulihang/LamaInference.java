package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.*;

public class LamaInference {
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final boolean isMergedInput; // INT8 模型把 image+mask 合并为 4 通道输入
    private boolean needsResize;   // 固定尺寸模型需要 resize
    private int modelH, modelW;    // 模型期望的输入尺寸

    public LamaInference(String path) throws OrtException {
        EnumSet<OrtProvider> providers = OrtEnvironment.getEnvironment().getAvailableProviders();
        System.out.println("Available ONNX Runtime providers:");
        for (OrtProvider provider : providers) {
            System.out.println(" - " + provider.getName());
        }
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = env.createSession(path, options);

        // 检测模型输入格式
        Set<String> inputNames = session.getInputNames();
        isMergedInput = inputNames.size() == 1 && inputNames.contains("input");
        System.out.println("Model input format: " + (isMergedInput ? "merged (4ch)" : "separate (image+mask)"));

        // 检测输入尺寸：通过 toString() 解析，固定正数尺寸才需要 resize
        try {
            Map<String, NodeInfo> inputInfo = session.getInputInfo();
            NodeInfo info = inputInfo.get(isMergedInput ? "input" : "image");
            String infoStr = info.toString();
            System.out.println(infoStr);
            int shapeIdx = infoStr.indexOf("shape=[");
            if (shapeIdx >= 0) {
                int endIdx = infoStr.indexOf("]", shapeIdx);
                String shapePart = infoStr.substring(shapeIdx + 7, endIdx);
                String[] dims = shapePart.trim().split(",\\s*");
                if (dims.length >= 4) {
                    int h = Integer.parseInt(dims[dims.length - 2].trim());
                    int w = Integer.parseInt(dims[dims.length - 1].trim());
                    if (h > 0 && w > 0) {
                        modelH = h;
                        modelW = w;
                        needsResize = true;
                    } else {
                        modelH = -1; modelW = -1; needsResize = false;
                    }
                } else {
                    modelH = -1; modelW = -1; needsResize = false;
                }
            } else {
                modelH = -1; modelW = -1; needsResize = false;
            }
        } catch (Exception e) {
            modelH = -1;
            modelW = -1;
            needsResize = false;
        }
        System.out.println("Model size: " + (needsResize ? (modelW + "x" + modelH + " (fixed)") : "dynamic"));
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }

    public Mat inpaint(Mat image, Mat mask) throws OrtException {
        int origH = image.rows(), origW = image.cols();

        // 在 resize 前保存原图，用于后面的 blend
        Mat origImage = image.clone();
        Mat origMask = mask.clone();

        // 固定尺寸模型需要 resize，动态模型直接 pad
        int targetW = needsResize ? modelW : origW;
        int targetH = needsResize ? modelH : origH;

        if (needsResize) {
            Imgproc.resize(image, image, new Size(targetW, targetH));
            Imgproc.resize(mask, mask, new Size(targetW, targetH), 0, 0, Imgproc.INTER_NEAREST);
        }

        if (isMergedInput) {
            return inpaintMerged(image, mask, origImage, origMask, targetW, targetH, origW, origH);
        } else {
            return inpaintSeparate(image, mask, targetW, targetH, origW, origH);
        }
    }

    private Mat inpaintMerged(Mat image, Mat mask, Mat origImage, Mat origMask,
                               int targetW, int targetH, int origW, int origH) throws OrtException {
        Map<String, OnnxTensor> inputs = prepareMergedInput(image, mask, 64);
        try {
            OrtSession.Result results = session.run(inputs);
            try {
                float[][][][] output = (float[][][][]) results.get(0).getValue();
                Mat outMat = convertOutputToMat(output);

                // 裁切掉 pad，再 resize 回原始尺寸
                Mat cropped = new Mat(outMat, new Rect(0, 0, targetW, targetH));
                Mat resized;
                if (needsResize) {
                    resized = new Mat();
                    Imgproc.resize(cropped, resized, new Size(origW, origH), 0, 0, Imgproc.INTER_CUBIC);
                } else {
                    resized = cropped.clone();
                }

                return blendWithOriginal(origImage, origMask, resized);
            } finally {
                results.close();
            }
        } finally {
            for (OnnxTensor t : inputs.values()) {
                t.close();
            }
        }
    }

    private Mat inpaintSeparate(Mat image, Mat mask, int targetW, int targetH,
                                 int origW, int origH) throws OrtException {
        Map<String, OnnxTensor> inputs = prepareImgAndMask(image, mask, 64);
        try {
            OrtSession.Result results = session.run(inputs);
            try {
                float[][][][] output = (float[][][][]) results.get(0).getValue();
                Mat outMat = convertOutputToMat(output);

                Mat cropped = new Mat(outMat, new Rect(0, 0, targetW, targetH));
                if (needsResize) {
                    Mat resized = new Mat();
                    Imgproc.resize(cropped, resized, new Size(origW, origH), 0, 0, Imgproc.INTER_CUBIC);
                    return resized;
                }
                return cropped.clone();
            } finally {
                results.close();
            }
        } finally {
            for (OnnxTensor t : inputs.values()) {
                t.close();
            }
        }
    }

    /**
     * 只在 mask 区域用 inpainted 结果，其余区域保留原图。
     */
    private Mat blendWithOriginal(Mat original, Mat mask, Mat inpainted) {
        byte[] origData = new byte[(int) (original.total() * original.channels())];
        byte[] inpData = new byte[(int) (inpainted.total() * inpainted.channels())];
        byte[] maskData = new byte[(int) (mask.total() * mask.channels())];
        original.get(0, 0, origData);
        inpainted.get(0, 0, inpData);
        mask.get(0, 0, maskData);

        int h = original.rows(), w = original.cols(), c = original.channels();
        byte[] outData = new byte[h * w * c];
        System.arraycopy(origData, 0, outData, 0, h * w * c);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                int gray = maskData[idx] & 0xFF;
                if (gray > 0) {
                    System.arraycopy(inpData, idx * c, outData, idx * c, c);
                }
            }
        }

        original.put(0, 0, outData);
        return original;
    }

    private Mat convertOutputToMat(float[][][][] output) {
        int outC = output[0].length;
        int outH = output[0][0].length;
        int outW = output[0][0][0].length;

        Mat outImg = new Mat(outH, outW, CvType.CV_8UC3);
        byte[] outData = new byte[outH * outW * outC];

        int idx = 0;
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                for (int c = 0; c < outC; c++) {
                    float v = output[0][c][y][x];
                    int iv = (int) (v * 255.0f);
                    iv = Math.max(0, Math.min(255, iv));
                    outData[idx++] = (byte) iv;
                }
            }
        }
        outImg.put(0, 0, outData);
        return outImg;
    }

    private Map<String, OnnxTensor> prepareMergedInput(Mat image, Mat mask, int padOutToModulo) throws OrtException {
        Mat paddedImg = padImgToModulo(image, padOutToModulo);
        Mat paddedMask = padImgToModulo(mask, padOutToModulo);

        int h = paddedImg.rows(), w = paddedImg.cols();

        float[] imgCHW = getImage(paddedImg, false);
        float[] maskCHW = getImage(paddedMask, true);
        for (int i = 0; i < maskCHW.length; i++) maskCHW[i] = maskCHW[i] > 0 ? 1.0f : 0.0f;

        int pixels = h * w;
        float[] merged = new float[4 * pixels];
        for (int i = 0; i < pixels; i++) {
            float invMask = 1.0f - maskCHW[i];
            merged[0 * pixels + i] = imgCHW[0 * pixels + i] * invMask;
            merged[1 * pixels + i] = imgCHW[1 * pixels + i] * invMask;
            merged[2 * pixels + i] = imgCHW[2 * pixels + i] * invMask;
            merged[3 * pixels + i] = maskCHW[i];
        }

        OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(merged), new long[]{1, 4, h, w});
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input", tensor);
        return inputs;
    }

    private Map<String, OnnxTensor> prepareImgAndMask(Mat image, Mat mask, int padOutToModulo) throws OrtException {
        Mat paddedImg = padImgToModulo(image, padOutToModulo);
        Mat paddedMask = padImgToModulo(mask, padOutToModulo);

        int h = paddedImg.rows(), w = paddedImg.cols();

        float[] imgCHW = getImage(paddedImg, false);
        OnnxTensor imgTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imgCHW), new long[]{1, 3, h, w});

        float[] maskCHW = getImage(paddedMask, true);
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
        converted.release();

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
        int outSize = Math.max(outH, outW);
        Mat dst = new Mat();
        Core.copyMakeBorder(img, dst, 0, outSize - h, 0, outSize - w, Core.BORDER_REFLECT);
        return dst;
    }
}
