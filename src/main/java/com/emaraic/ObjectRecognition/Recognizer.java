package com.emaraic.ObjectRecognition;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.filechooser.FileNameExtensionFilter;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

// import com.esotericsoftware.tablelayout.swing.Table;

import interfaces.ConfigConstants;


public class Recognizer {


    // private Table table;
//    private JButton predict;
//    private JButton incep;
//    private JButton img;
//    private JFileChooser incepch;
//    private JFileChooser imgch;
//    private JLabel viewer;
//    private JTextField result;
//    private JTextField imgpth;
//    private JTextField modelpth;
    private FileNameExtensionFilter imgfilter = new FileNameExtensionFilter(
            "JPG & JPEG Images", "jpg", "jpeg");
    private String modelpath;
    private String imagepath;
    private boolean modelselected = false;
    private byte[] graphDef;
    private List<String> labels;

    public Recognizer() {
//        
    }

    public void actionPerformed(Mat image) { 
    		Mat zoomImage=	zoomImage(image, 5);
    	       Imgcodecs.imwrite("C:\\testimage\\output-images\\output.jpg", zoomImage);
                try {
                    File file = new File("C:\\testimage\\output-images\\output.jpg");
                    imagepath = file.getAbsolutePath();
                    executeInceptionModel();
                    System.out.println("Image Path: " + imagepath);                    
                    executePredictionModel();
                    
                } catch (Exception ex) {
                    Logger.getLogger(Recognizer.class.getName()).log(Level.SEVERE, null, ex);
                }
                
     } 
        
    
    private void executeInceptionModel() {
    	File file = new File("C:\\inception_dec_2015");;                 
        modelpath = file.getAbsolutePath();                
        System.out.println("Opening: " + file.getAbsolutePath());
        modelselected = true;
        graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
        labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));
    }
    
    private void executePredictionModel() {
    	byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));

        try (Tensor image = Tensor.create(imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int bestLabelIdx = maxIndex(labelProbabilities);
            
            System.out.println(
                    String.format(
                            "BEST MATCH: %s (%.2f%% likely)",
                            labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
        }
    }
    ///
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                    Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {

        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }

        Output resizeBilinear(Output images, Output size) {
            return binaryOp("ResizeBilinear", images, size);
        }

        Output expandDims(Output input, Output dim) {
            return binaryOp("ExpandDims", input, dim);
        }

        Output cast(Output value, DataType dtype) {
            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
        }

        Output decodeJpeg(Output contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .output(0);
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
        }

        private Graph g;
    }
    
    private static Mat zoomImage(Mat sourceImage, int zoomingFactor) {
     	
    	Mat destination = new Mat(sourceImage.rows() * zoomingFactor, sourceImage.cols()*  zoomingFactor, sourceImage.type());  
        
        Imgproc.resize(sourceImage, destination, destination.size(),  zoomingFactor,zoomingFactor,Imgproc.INTER_LANCZOS4);
        
        return destination;
    }
    
    ////////////
    
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws Exception {
    	if(ConfigConstants.IS_WEB_CAM_ENABLED) {
        	Timer timer = new Timer();
        	timer.schedule(new Recognizer().new SpotDetectorRun(), 0, ConfigConstants.CAM_CAPTURE_DELAY);
        }else {
             Mat image = Imgcodecs.imread(ConfigConstants.TEST_FILE, Imgcodecs.IMREAD_COLOR);
             new Recognizer().actionPerformed(image);
        }	    
    	
    }
    
    
    class SpotDetectorRun extends TimerTask {
    	
    	private Process vlcPlayer;
    	private String productId;
    	
    	public void run() {
    	       startWebFeed();
    	}
    	
    	
    	private void startWebFeed() {
    		Mat webcam_image=new Mat();  
    		
    		VideoCapture capture =new VideoCapture(0);
    		
    		if(ConfigConstants.IS_FULL_HD_ENABLED) {
    			capture.set(Videoio.CAP_PROP_FRAME_WIDTH, ConfigConstants.CAM_FULL_HD_WIDTH);
    			capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, ConfigConstants.CAM_FULL_HD_HEIGHT);
    		}else {
    			capture.set(Videoio.CAP_PROP_FRAME_WIDTH, ConfigConstants.CAM_HD_WIDTH);
    			capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, ConfigConstants.CAM_HD_HEIGHT);
    		}
    		
    		capture.set(Videoio.CV_CAP_PROP_ZOOM, ConfigConstants.CAM_ZOOM_LEVEL);
    		
    		
    		System.err.println(capture.get(3)+" cam set:"+capture.get(4));
    		
    		if( capture.isOpened())  { 
    			try {
    		        capture.read(webcam_image);
    		        
    		        if( !webcam_image.empty() ) {    		        	
    		        	//new Recognizer().actionPerformed(webcam_image);
    		        	//runCircleDetector(webcam_image);
    		        	
    		        }  
    		        else {   
    		            System.out.println(" --(!) No captured frame -- Break!");   
    		            //break;   
    		        } 
    		
    			} catch (Exception e) {
    					e.printStackTrace();
    				}
    		}  
    		capture.release();
    	}
    }
    

//    public static void main(String[] args) {
//        SwingUtilities.invokeLater(new Runnable() {
//            public void run() {
//                new Recognizer().setVisible(true);
//
//            }
//        });
//    }

}
