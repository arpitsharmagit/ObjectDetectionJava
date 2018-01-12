package com.gl;
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


import java.util.List;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import javax.imageio.ImageIO;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import interfaces.ConfigConstants;

public class Main implements Runnable {
	final int INTERVAL = 1000;	
	
    private static final int INPUT_SIZE = 300;
    private static final String modelDir = "D:\\Projects\\models\\ssd_mobilenet_v1_android_export.pb";
    private static final String labelDir = "D:\\Projects\\models\\coco_labels_list.txt";
    private static final String sampleImageFile = "D:\\Projects\\models\\400.jpg"; 
    private static final int cropSize = 300;
    
    
    private static TensorFlowDetector detector;
    public Main(){    	
    }
    public static void main(String[] args) {
    	Main gs = new Main();
        Thread th = new Thread(gs);
        th.start();
    }
    
    private static Vector<String> getLabels(String path){
    	 Vector<String> lbls = new Vector<String>();
    	 try {
    	  BufferedReader br = new BufferedReader(new FileReader(path));			
    	    String line;
    	    while ((line = br.readLine()) != null) {
    	    	lbls.add(line);
    	    }
    	    br.close();
    	    return lbls;
    	    } catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				return lbls;
			}
    }
    private void initialize (){
    	Vector<String> labels = getLabels(labelDir);
    	detector = new TensorFlowDetector(modelDir, labels, INPUT_SIZE);    	
    }
    private static BufferedImage Mat2BufferedImage(Mat matrix) {        
    	   MatOfByte mob=new MatOfByte();
    	   Imgcodecs.imencode(".png", matrix, mob);
    	   byte ba[]=mob.toArray();

    	   BufferedImage bi = null;
    	try {
    	bi = ImageIO.read(new ByteArrayInputStream(ba));
    	} catch (IOException e) {
    	// TODO Auto-generated catch block
    	e.printStackTrace();
    	}
    	   return bi;
    	}
    public void run() {
    		
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
		
        int i = 0;    	    	  
		try {
			initialize();
			Mat webcam_image=new Mat();  	
			while (true) { 
				if(capture.isOpened()){
					capture.read(webcam_image);
					if( !webcam_image.empty() ) {    		        	
	                    
	                	BufferedImage finalImage =Mat2BufferedImage(webcam_image);     	                	
	                	BufferedImage cropImage = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
	                	cropImage.getGraphics().drawImage(finalImage, 0, 0, null); 
	                    ImageIO.write(cropImage, "jpg", new File((i++) + "-aa.jpg"));
	                    
	                    // recognizeImage
	                    List<Classifier.Recognition> recognitions = detector.recognizeImage(cropImage);
	                    System.out.println("Result length " + recognitions.size());
	                    for (Classifier.Recognition recognition : recognitions) {
	                        RectF rectF = recognition.getLocation();
	                        System.out.println(recognition.getTitle() + " " + recognition.getConfidence() + ", " +
	                                (int) rectF.left + " " + (int) rectF.top + " " + (int) rectF.right + " " + ((int) rectF.bottom));
	                       
	                    }			        				        
			        } 
				}				
           
                //save
                
                Thread.sleep(INTERVAL);

            }	
			
        	           
//            Graphics2D graphics = convertedImg.createGraphics();
//
//            Stroke stroke = graphics.getStroke();
//            graphics.setStroke(new BasicStroke(3));
//            graphics.setColor(Color.green);
//            graphics.drawRoundRect((int) rectF.left, (int) rectF.top, (int) rectF.right, (int) rectF.bottom, 5, 5);
//            graphics.setStroke(stroke);
//
//            graphics.dispose();
//            ImageIcon icon=new ImageIcon(convertedImg);
//            JFrame frame=new JFrame();
//            frame.setLayout(new FlowLayout());
//            frame.setSize(convertedImg.getWidth(),convertedImg.getHeight());
//            JLabel lbl=new JLabel();
//            frame.setTitle("Java (Win/Ubuntu), Tensorflow & Yolo");
//            lbl.setIcon(icon);
//            frame.add(lbl);
//            frame.setVisible(true);
//            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        } catch (Exception e) {
            e.printStackTrace();
            capture.release();
        }
    	
    }    
    
    

}