package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;

public class LamaInference {
    public static final String DEVICE_CPU = "cpu";
    public static final String DEVICE_WEBGPU = "webgpu";
    private static final String WEBGPU_EP = "WebGpuExecutionProvider";
    private static final String WEBGPU_LIB_BASE = "onnxruntime_providers_webgpu";

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final String device;   // 实际使用的后端：cpu / webgpu
    private OrtSession session;
    private final boolean isMergedInput; // INT8 模型把 image+mask 合并为 4 通道输入
    private boolean needsResize;   // 固定尺寸模型需要 resize
    private int modelH, modelW;    // 模型期望的输入尺寸
    private double lastRunMs = -1; // 最近一次 session.run() 耗时（纯推理，不含前后处理）

    /** CPU session。 */
    public LamaInference(String path) throws OrtException {
        this(path, DEVICE_CPU, null);
    }

    /**
     * @param path          ONNX model path
     * @param device        {@link #DEVICE_WEBGPU} 或 {@link #DEVICE_CPU}（null 视为 CPU）；
     *                      WebGPU 不可用时自动回退 CPU
     * @param webgpuLibPath 存放 WebGPU provider 库的目录或库文件本身的完整路径，可为 null。
     *                      自动按平台查找库名（onnxruntime_providers_webgpu.dll / .dylib / .so），
     *                      目录查找顺序：webgpuLibPath → 当前工作目录 → jar 所在目录
     */
    public LamaInference(String path, String device, String webgpuLibPath) throws OrtException {
        // Java 26+ 下 ORT-jar 的 native 加载会打印 System.loadLibrary 的 restricted-method 警告；
        // --enable-native-access=ALL-UNNAMED 由宿主程序传入 JVM 参数解决
        this.device = createSession(path, device, webgpuLibPath);

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

    /** 创建 session，返回实际使用的后端。WebGPU 失败时回退 CPU。 */
    private String createSession(String path, String device, String webgpuLibPath) throws OrtException {
        if (DEVICE_WEBGPU.equalsIgnoreCase(device == null ? DEVICE_CPU : device)) {
            try {
                // WebGPU 不可用（缺库 / 无 GPU 设备）时走到下面的 CPU 分支
                registerWebGpu(findWebGpuLib(webgpuLibPath));
                session = createWebGpuSession(path);
                System.out.println("[lama4j] session created with WebGPU");
                return DEVICE_WEBGPU;
            } catch (Throwable t) {
                // 回退到 CPU
                System.out.println("[lama4j] WebGPU unavailable, falling back to CPU: " + t);
            }
        }
        session = env.createSession(path, new OrtSession.SessionOptions());
        System.out.println("[lama4j] session created with CPU");
        return DEVICE_CPU;
    }

    private OrtSession createWebGpuSession(String path) throws OrtException {
        List<OrtEpDevice> devices = new ArrayList<>();
        for (OrtEpDevice d : env.getEpDevices()) {
            if (WEBGPU_EP.equals(d.getEpName())) devices.add(d);
        }
        if (devices.isEmpty()) throw new IllegalStateException("no " + WEBGPU_EP + " device");

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        // device-dependent EP：不支持的节点自动 fallback 到 CPU
        options.addExecutionProvider(devices, new HashMap<>());
        return env.createSession(path, options);
    }

    /** 按平台拼出 WebGPU provider 库的文件名。 */
    private static String webGpuLibName() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String prefix = os.contains("mac") ? "lib" : "";
        String suffix = os.contains("win") ? ".dll" : os.contains("mac") ? ".dylib" : ".so";
        return prefix + WEBGPU_LIB_BASE + suffix;
    }

    /**
     * 定位 WebGPU provider 库：webgpuLibPath 可直接指向库文件，也可以是目录
     * （再依次尝试当前工作目录、jar 所在目录）。
     */
    private static File findWebGpuLib(String webgpuLibPath) {
        String name = webGpuLibName();
        List<File> dirs = new ArrayList<>();
        if (webgpuLibPath != null && !webgpuLibPath.isEmpty()) {
            File f = new File(webgpuLibPath);
            if (f.isFile()) return f;
            dirs.add(f);
        }
        dirs.add(new File(System.getProperty("user.dir")));
        dirs.add(jarDir());
        for (File dir : dirs) {
            File f = new File(dir, name);
            if (f.isFile()) return f;
        }
        throw new IllegalStateException(name + " not found (searched: " + dirs + ")");
    }

    private static File jarDir() {
        try {
            File f = new File(LamaInference.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            return f.isFile() ? f.getParentFile() : f;
        } catch (Exception e) {
            return new File(System.getProperty("user.dir"));
        }
    }

    /** 首次调用时注册 WebGPU 插件 EP；onnxruntime.dll 本身由宿主程序或 ORT-jar 负责加载。 */
    private static synchronized void registerWebGpu(File dll) throws OrtException {
        System.out.println("[lama4j] found WebGPU provider DLL: " + dll.getAbsolutePath());
        OrtEnvironment.getEnvironment().registerExecutionProviderLibrary("webgpu", dll.getAbsolutePath());
    }

    /** 实际使用的后端："cpu" 或 "webgpu"。 */
    public String getDevice() {
        return device;
    }

    public boolean isWebGpu() {
        return DEVICE_WEBGPU.equals(device);
    }

    /** 最近一次 inpaint 的纯推理耗时（毫秒），尚无记录时为 -1。 */
    public double getLastRunMs() {
        return lastRunMs;
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
            OrtSession.Result results = run(inputs);
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
            OrtSession.Result results = run(inputs);
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

    private OrtSession.Result run(Map<String, OnnxTensor> inputs) throws OrtException {
        long t = System.nanoTime();
        OrtSession.Result results = session.run(inputs);
        lastRunMs = (System.nanoTime() - t) / 1e6;
        return results;
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
        Imgproc.cvtColor(outImg, outImg, Imgproc.COLOR_BGR2RGB);
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
