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
        Test2();
    }

    private static void Test() throws OrtException{
        // 加载 ONNX 模型
        LamaInference lama = new LamaInference("lama_fp32.onnx");


        // 读取图像和 mask
        Mat image = Imgcodecs.imread("image.jpg");
        Mat mask = Imgcodecs.imread("mask.png", Imgcodecs.IMREAD_GRAYSCALE);
        for (int i = 0; i < 50; i++) {
            Mat outImg = lama.inpaint(image,mask);
            Imgcodecs.imwrite("out.jpg", outImg);
            System.out.println("Saved: out.jpg");
        }
        Mat outImg2 = lama.inpaint(image,mask);
        Imgcodecs.imwrite("out2.jpg", outImg2);
        System.out.println("Saved: out.jpg");
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