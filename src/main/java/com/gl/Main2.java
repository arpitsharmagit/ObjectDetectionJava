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
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.Graph;
import org.tensorflow.Operation;

public class Main2 implements Classifier {

    private static final int BLOCK_SIZE = 32;
    private static final int MAX_RESULTS = 3;
    private static final int NUM_CLASSES = 20;
    private static final int NUM_BOXES_PER_BLOCK = 5;
    private static final int INPUT_SIZE = 416;
    private static final String inputName = "input";
    private static final String outputName = "output";

    // Pre-allocated buffers.
    private static int[] intValues;
    private static float[] floatValues;
    private static String[] outputNames;

    // yolo 2
    private static final double[] ANCHORS = { 1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071 };

    // tiny yolo
    //private static final double[] ANCHORS = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };

    private static final String[] LABELS = {
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
    };

    private static TensorFlowInferenceInterface inferenceInterface;

    public static void main(String[] args) {

        //String modelDir = "/home/user/JavaProjects/TensorFlowJavaProject"; // Ubuntu
        String modelAndTestImagesDir = "D:\\Projects\\models"; // Windows
        String imageFile = "C:\\temp\\GlobalLogic Customer Gifts\\IMG_4848.jpg"; // 416x416 test image

        outputNames = outputName.split(",");
        floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];

        // yolo 2 voc
        inferenceInterface = new TensorFlowInferenceInterface(Paths.get(modelAndTestImagesDir, "ssd_mobilenet_v1_android_export"));

        // tiny yolo voc
        //inferenceInterface = new TensorFlowInferenceInterface(Paths.get(modelAndTestImagesDir, "graph-tiny-yolo-voc.pb"));

        BufferedImage img;

        try {
            img = ImageIO.read(new File(imageFile));

            BufferedImage convertedImg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            convertedImg.getGraphics().drawImage(img, 0, 0, null);

            intValues = ((DataBufferInt) convertedImg.getRaster().getDataBuffer()).getData() ;

            List<Classifier.Recognition> recognitions = recognizeImage();

            System.out.println("Result length " + recognitions.size());

            Graphics2D graphics = convertedImg.createGraphics();

            for (Recognition recognition : recognitions) {
                RectF rectF = recognition.getLocation();
                System.out.println(recognition.getTitle() + " " + recognition.getConfidence() + ", " +
                        (int) rectF.left + " " + (int) rectF.top + " " + (int) rectF.right + " " + ((int) rectF.bottom));
                Stroke stroke = graphics.getStroke();
                graphics.setStroke(new BasicStroke(3));
                graphics.setColor(Color.green);
                graphics.drawRoundRect((int) rectF.left, (int) rectF.top, (int) rectF.right, (int) rectF.bottom, 5, 5);
                graphics.setStroke(stroke);
            }

            graphics.dispose();
            ImageIcon icon=new ImageIcon(convertedImg);
            JFrame frame=new JFrame();
            frame.setLayout(new FlowLayout());
            frame.setSize(convertedImg.getWidth(),convertedImg.getHeight());
            JLabel lbl=new JLabel();
            frame.setTitle("Java (Win/Ubuntu), Tensorflow & Yolo");
            lbl.setIcon(icon);
            frame.add(lbl);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private static List<Classifier.Recognition> recognizeImage() {

        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
        }
        inferenceInterface.feed(inputName, floatValues, 1, INPUT_SIZE, INPUT_SIZE, 3);

        inferenceInterface.run(outputNames, false);

        final int gridWidth = INPUT_SIZE / BLOCK_SIZE;
        final int gridHeight = INPUT_SIZE / BLOCK_SIZE;

        final float[] output = new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];

        inferenceInterface.fetch(outputNames[0], output);

        // Find the best detections.
        final PriorityQueue<Classifier.Recognition> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Classifier.Recognition>() {
                            @Override
                            public int compare(final Classifier.Recognition lhs, final Classifier.Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    final float xPos = (x + expit(output[offset + 0])) * BLOCK_SIZE;
                    final float yPos = (y + expit(output[offset + 1])) * BLOCK_SIZE;

                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * BLOCK_SIZE;
                    final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * BLOCK_SIZE;

                    final RectF rect =
                            new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(INPUT_SIZE - 1, xPos + w / 2),
                                    Math.min(INPUT_SIZE - 1, yPos + h / 2));

                    final float confidence = expit(output[offset + 4]);

                    String detectedClass = "-1";
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = output[offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = String.format("%d", c);
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.01) {
                        pq.add(new Classifier.Recognition(detectedClass, LABELS[Integer.parseInt(detectedClass)], confidenceInClass, rect));
                    }
                }
            }
        }

        final ArrayList<Classifier.Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;

    }

    private static float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private static void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    public void close() {
        inferenceInterface.close();
    }
}