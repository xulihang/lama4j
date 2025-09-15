import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.xulihang.LamaInference;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class Test {
    public static void main(String[] args) throws OrtException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Hello world!");
        System.out.println("Test");
        // 加载 ONNX 模型
        LamaInference lama = new LamaInference("lama_fp32.onnx");


        // 读取图像和 mask
        Mat image = Imgcodecs.imread("image.jpg");
        Mat mask = Imgcodecs.imread("mask.png", Imgcodecs.IMREAD_GRAYSCALE);
        Mat outImg = lama.inpaint(image,mask);
        Imgcodecs.imwrite("out.jpg", outImg);

        Mat outImg2 = lama.inpaint(image,mask);
        Imgcodecs.imwrite("out2.jpg", outImg2);
        System.out.println("Saved: out.jpg");
    }
}