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


import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.bytedeco.javacv.*;

import static org.bytedeco.javacpp.opencv_core.IplImage;
import static org.bytedeco.javacpp.opencv_core.cvFlip;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;

public class Main implements Runnable {
	final int INTERVAL = 100;
	CanvasFrame canvas = new CanvasFrame("Web Cam");
	
    private static final int INPUT_SIZE = 300;
    private static final String modelDir = "D:\\Projects\\models\\ssd_mobilenet_v1_android_export.pb";
    private static final String labelDir = "D:\\Projects\\models\\coco_labels_list.txt";
    private static final String sampleImageFile = "D:\\Projects\\models\\400.jpg"; 
    private static final int cropSize = 300;
    
    
    private static TensorFlowDetector detector;
    public Main(){
    	canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
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
    public void run() {
    	FrameGrabber grabber = new VideoInputFrameGrabber(0);
    	OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    	Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        
        
        int i = 0;    	    	  
		try {
			initialize();
			grabber.setImageWidth(300);
            grabber.setImageHeight(300);
			grabber.start();
			
			while (true) {                
				Frame frame = grabber.grab();
				                 
                if (frame != null)
                {
                	IplImage img = converter.convert(frame);
                    cvFlip(img, img, 1);
                    Frame flippedFrame = converter.convert(img);
                    
                	BufferedImage finalImage =paintConverter.getBufferedImage(flippedFrame,1);                	                	
                	BufferedImage cropImage = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
                	cropImage.getGraphics().drawImage(finalImage, 0, 0, null); 
                    ImageIO.write(cropImage, "jpg", new File("abc.png"));
                    canvas.showImage(cropImage);
                    cvSaveImage((i++) + "-aa.jpg", img);
                    
                    // recognizeImage
                    List<Classifier.Recognition> recognitions = detector.recognizeImage(cropImage);
                    System.out.println("Result length " + recognitions.size());
                    for (Classifier.Recognition recognition : recognitions) {
                        RectF rectF = recognition.getLocation();
                        System.out.println(recognition.getTitle() + " " + recognition.getConfidence() + ", " +
                                (int) rectF.left + " " + (int) rectF.top + " " + (int) rectF.right + " " + ((int) rectF.bottom));
                       
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
        }
    	
    }
}