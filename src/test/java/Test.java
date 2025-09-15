import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import static com.xulihang.LamaInference.prepareImgAndMask;

public class Test {
    public static void main(String[] args) throws OrtException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Hello world!");
        System.out.println("Test");
        // 加载 ONNX 模型
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("lama_fp32.onnx", options);

        // 读取图像和 mask
        Mat image = Imgcodecs.imread("image.jpg");
        Mat mask = Imgcodecs.imread("mask.png", Imgcodecs.IMREAD_GRAYSCALE);

        // 调整大小到 512x512
        Imgproc.resize(image, image, new Size(512, 512));
        Imgproc.resize(mask, mask, new Size(512, 512), 0, 0, Imgproc.INTER_NEAREST);

        System.out.println("Run model...");
        Map<String, OnnxTensor> inputs = prepareImgAndMask(image, mask, env, session, 8);

        // 推理
        OrtSession.Result results = session.run(inputs);
        float[][][][] output = (float[][][][]) results.get(0).getValue();

        // 转换输出 (NCHW -> HWC)
        int outC = output[0].length;
        int outH = output[0][0].length;
        int outW = output[0][0][0].length;

        Mat outImg = new Mat(outH, outW, CvType.CV_8UC3);
        byte[] outData = new byte[outH * outW * outC];

        float minVal = Float.MAX_VALUE;
        float maxVal = -Float.MAX_VALUE;

// 先找最小最大值
        for (int c = 0; c < outC; c++) {
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    float v = output[0][c][y][x];
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }
            }
        }

        int idx = 0;
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                for (int c = 0; c < outC; c++) {
                    float v = output[0][c][y][x];
                    // 归一化到 0–255
                    int iv = (int) ((v - minVal) / (maxVal - minVal) * 255.0f);
                    iv = Math.max(0, Math.min(255, iv));
                    outData[idx++] = (byte) iv;
                }
            }
        }

        outImg.put(0, 0, outData);
        Imgproc.cvtColor(outImg, outImg, Imgproc.COLOR_BGR2RGB);
        Imgcodecs.imwrite("out.jpg", outImg);
        System.out.println("Saved: out.jpg (range " + minVal + " ~ " + maxVal + ")");
    }
}