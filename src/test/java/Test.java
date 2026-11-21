import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.xulihang.LamaInference;
import com.xulihang.LamaInpaintDynamicSingleton;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import static com.xulihang.LamaInpaintDynamicSingleton.inpaintONNX;

public class Test {
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Hello world!");
        System.out.println("Test");
        Test(args.length > 0 ? args[0] : "webgpu", args.length > 1 ? args[1] : "lama_512_int8.onnx");
    }

    private static void Test(String device, String model) throws OrtException{
        // 加载 ONNX 模型：device = "webgpu"（默认）或 "cpu"
        LamaInference lama = new LamaInference(model, device, "src/main/resources/native");


        // 读取图像和 mask
        Mat image = Imgcodecs.imread("cropped.png");
        Mat mask = Imgcodecs.imread("cropped.png-mask.png", Imgcodecs.IMREAD_GRAYSCALE);

        long t = System.nanoTime();
        Mat outImg = lama.inpaint(image,mask);
        System.out.printf("device=%s  total=%.0fms  run=%.0fms%n", lama.getDevice(),
                (System.nanoTime() - t) / 1e6, lama.getLastRunMs());
        Imgcodecs.imwrite("out_" + lama.getDevice() + ".jpg", outImg);
        System.out.println("Saved: out_" + lama.getDevice() + ".jpg");
    }

    private static void Test2() throws Exception {

        // 只初始化一次模型
        LamaInpaintDynamicSingleton.ModelCache.init("model.onnx");
        Mat img = LamaInpaintDynamicSingleton.loadImage("image.jpg");
        Mat mask = LamaInpaintDynamicSingleton.loadMask("mask.png");
        // 调用 inpaint
        Mat out = inpaintONNX(
                img,
                mask,
                512
        );
        Imgcodecs.imwrite("out2.jpg", out);
        // 程序结束时释放模型
        LamaInpaintDynamicSingleton.ModelCache.close();

    }
}