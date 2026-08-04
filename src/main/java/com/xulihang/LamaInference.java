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

    public LamaInference(String path) throws OrtException {
        EnumSet<OrtProvider> providers = OrtEnvironment.getEnvironment().getAvailableProviders();
        System.out.println("Available ONNX Runtime providers:");
        for (OrtProvider provider : providers) {
            System.out.println(" - " + provider.getName());
        }
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = env.createSession(path, options);

        // 检测模型输入格式：如果只有一个名为 "input" 的 4 通道输入，则是合并输入格式
        Set<String> inputNames = session.getInputNames();
        isMergedInput = inputNames.size() == 1 && inputNames.contains("input");
        System.out.println("Model input format: " + (isMergedInput ? "merged (4ch)" : "separate (image+mask)"));
    }

    public Mat inpaint(Mat image, Mat mask) throws OrtException {
        // 保存原始图像和 mask（resize 会覆盖输入）
        int origH = image.rows(), origW = image.cols();
        Mat origImage = image.clone();
        Mat origMask = mask.clone();

        // 和 JS 代码一样，先 resize 到 512x512
        Imgproc.resize(image, image, new Size(512, 512));
        Imgproc.resize(mask, mask, new Size(512, 512), 0, 0, Imgproc.INTER_NEAREST);

        Mat output;

        if (isMergedInput) {
            Map<String, OnnxTensor> inputs = prepareMergedInput(image, mask, 64);
            try (OrtSession.Result results = session.run(inputs)) {
                output = convertOutputToMat((float[][][][]) results.get(0).getValue());
            }
        } else {
            Map<String, OnnxTensor> inputs = prepareImgAndMask(image, mask, 64);
            try (OrtSession.Result results = session.run(inputs)) {
                output = convertOutputToMat((float[][][][]) results.get(0).getValue());
            }
        }

        // 裁切回 512x512（去掉 pad），再 resize 回原始尺寸
        Mat cropped = new Mat(output, new Rect(0, 0, 512, 512));
        Mat resized = new Mat();
        Imgproc.resize(cropped, resized, new Size(origW, origH), 0, 0, Imgproc.INTER_CUBIC);

        // 只在 mask 区域使用模型输出，其余保留原图
        return blendWithOriginal(origImage, origMask, resized);
    }

    /**
     * 只在 mask 区域用 inpainted 结果，其余区域保留原图。
     * 对应 JS code: for each pixel, if mask>0 use model output else use original.
     */
    private Mat blendWithOriginal(Mat original, Mat mask, Mat inpainted) {
        Mat result = original.clone();
        byte[] origData = new byte[(int) (original.total() * original.channels())];
        byte[] inpData = new byte[(int) (inpainted.total() * inpainted.channels())];
        byte[] maskData = new byte[(int) (mask.total() * mask.channels())];
        result.get(0, 0, origData);
        inpainted.get(0, 0, inpData);
        mask.get(0, 0, maskData);

        int h = original.rows(), w = original.cols(), c = original.channels();
        byte[] outData = new byte[h * w * c];
        System.arraycopy(origData, 0, outData, 0, h * w * c);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x);
                int gray = maskData[idx] & 0xFF;
                if (gray > 0) {
                    for (int ch = 0; ch < c; ch++) {
                        outData[idx * c + ch] = inpData[idx * c + ch];
                    }
                }
            }
        }

        result.put(0, 0, outData);
        return result;
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
        // 模型输出是 RGB，OpenCV 当 BGR 存储，不做转换即可保持正确
        return outImg;
    }

    private Map<String, OnnxTensor> prepareMergedInput(Mat image, Mat mask, int padOutToModulo) throws OrtException {
        image = padImgToModulo(image, padOutToModulo);
        mask = padImgToModulo(mask, padOutToModulo);

        int h = image.rows(), w = image.cols();

        // 将 image (BGR) 转为 RGB CHW
        float[] imgCHW = getImage(image, false);
        // mask 单通道 CHW
        float[] maskCHW = getImage(mask, true);
        for (int i = 0; i < maskCHW.length; i++) maskCHW[i] = maskCHW[i] > 0 ? 1.0f : 0.0f;

        // 关键：image * (1 - mask)，对应 JS 代码 f[e*d+t] = s[e*d+t] * (1 - u[t])
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
        // 取较大值确保 H==W，big-lama 模型内部 FFC 层要求两者一致
        int outSize = Math.max(outH, outW);
        Mat dst = new Mat();
        Core.copyMakeBorder(img, dst, 0, outSize - h, 0, outSize - w, Core.BORDER_REFLECT);
        return dst;
    }
}
