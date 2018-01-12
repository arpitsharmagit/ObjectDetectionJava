package interfaces;

//import org.opencv.core.Scalar;

public interface ConfigConstants {
	
	//String TEST_FILE = "C:\\\\testimage\\input-images\\shapes2-web.jpg";
	String TEST_FILE = "C:\\\\testimage\\input-images\\smallImage50.png";
	//String TEST_FILE = "C:\\\\testimage\\input-images\\shapes2.jpg";
	
	String ZOOM_FILE_LOCATION = "C:\\\\testimage\\\\zoom-test-image\\\\";
	String ZOOM_FILE_NAME_PREFIX = "output-zoom-image-X";
	String ZOOM_PROMOTION_EXTENSION = ".png";
	
	String VIDEO_FILE_LOCATION = "C:\\\\testimage\\\\video\\\\";
	String VIDEO_FILE_NAME_PREFIX = "Video-";
	String VIDEO_PROMOTION_EXTENSION = ".mp4";
	
	String PROCESSED_IMAGE_FILE_LOCATION = "C:\\\\testimage\\\\QR-PROCESS-IMAGES\\\\";
	String PROCESSED_IMAGE_FILE_NAME_PREFIX = "output-crop-test-";
	String PROCESSED_IMAGE_EXTENSION = ".png";
	
	String OUTPUT_IMAGE_LOWER_RANGE = "C:\\testimage\\output-images\\output-state1.png";
	String OUTPUT_IMAGE_UPPER_RANGE = "C:\\testimage\\output-images\\output-state2.png";
	String OUTPUT_IMAGE_WEIGHTED_RANGE = "C:\\testimage\\output-images\\output-state3.png";
	String OUTPUT_IMAGE_SPOT_DETECTED = "C:\\testimage\\output-images\\output-state4.png";
	
//	Scalar COLOR_LOWER_RANGE_START = new Scalar(0,150,150);
//	Scalar COLOR_LOWER_RANGE_END = new Scalar(10,255,255);
//	Scalar COLOR_UPPER_RANGE_START = new Scalar(160, 100, 100);
//	Scalar COLOR_UPPER_RANGE_END = new Scalar(179, 255, 255);
	
	int ZOOM_FACTOR = 1;
	int ZOOM_FACTOR_CUT_OFF = 5;
	
	double CAM_ZOOM_LEVEL = 4.0;
	double CAM_FULL_HD_WIDTH = 1920;
	double CAM_FULL_HD_HEIGHT = 1080;
	double CAM_HD_WIDTH = 1280;
	double CAM_HD_HEIGHT = 720;
	long CAM_CAPTURE_DELAY = 5000;
	
	double ENHANCE_IMAGE_ALPHA = 0.5;
	double ENHANCE_IMAGE_BETA = 10;
	
	double REDRAW_WHITE = 255;
	double REDRAW_BLACK = 0;
	double REDRAW_RED_CUT_OFF = 0;
	double REDRAW_GREEN_CUT_OFF = 0;
	double REDRAW_BLUE_CUT_OFF = 0;
	
	int SHAPE_IDENTIFIER = 4;
	boolean EXACT_SHAPE_IDENTIFIER = false;

	boolean IS_FULL_HD_ENABLED = true;
	boolean IS_WEB_CAM_ENABLED = true;
}
