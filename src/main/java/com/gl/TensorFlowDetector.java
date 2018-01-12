package com.gl;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import javax.imageio.ImageIO;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import java.nio.file.Paths;

public class TensorFlowDetector implements Classifier {	  

	  // Only return this many results.
	  private static final int MAX_RESULTS = 100;

	  // Config values.
	  private String inputName;


	  // Pre-allocated buffers.
	  private int[] intValues;
	  private byte[] byteValues;
	  private float[] outputLocations;
	  private float[] outputScores;
	  private float[] outputClasses;
	  private float[] outputNumDetections;
	  private String[] outputNames;

	  private boolean logStats = false;

	  private TensorFlowInferenceInterface inferenceInterface;
	  private Vector<String> labels;
	  private int inputSize;

	  public TensorFlowDetector(final String modelFilepath,
		      final Vector<String> labelsParam,
		      final int inputSizeParam) {
		  inputSize =inputSizeParam;
		  labels = labelsParam;
		  
		  inferenceInterface = new TensorFlowInferenceInterface(Paths.get(modelFilepath));

		    final Graph g = inferenceInterface.graph();

		    inputName = "image_tensor";
		    // The inputName node has a shape of [N, H, W, C], where
		    // N is the batch size
		    // H = W are the height and width
		    // C is the number of channels (3 for our purposes - RGB)
		    final Operation inputOp = g.operation(inputName);
		    if (inputOp == null) {
		      throw new RuntimeException("Failed to find input Node '" + inputName + "'");
		    }
		    // The outputScoresName node has a shape of [N, NumLocations], where N
		    // is the batch size.
		    final Operation outputOp1 = g.operation("detection_scores");
		    if (outputOp1 == null) {
		      throw new RuntimeException("Failed to find output Node 'detection_scores'");
		    }
		    final Operation outputOp2 = g.operation("detection_boxes");
		    if (outputOp2 == null) {
		      throw new RuntimeException("Failed to find output Node 'detection_boxes'");
		    }
		    final Operation outputOp3 = g.operation("detection_classes");
		    if (outputOp3 == null) {
		      throw new RuntimeException("Failed to find output Node 'detection_classes'");
		    }

		    // Pre-allocate buffers.
		    outputNames = new String[] {"detection_boxes", "detection_scores",
		                                  "detection_classes", "num_detections"};
		    intValues = new int[inputSize * inputSize];
		    byteValues = new byte[inputSize * inputSize * 3];
		    outputScores = new float[MAX_RESULTS];
		    outputLocations = new float[MAX_RESULTS * 4];
		    outputClasses = new float[MAX_RESULTS];
		    outputNumDetections = new float[1];
		  
	  }

	  public List<Recognition> recognizeImage(final BufferedImage img ) throws IOException {
	    // Log this method so that it can be analyzed with systrace.
		  
          BufferedImage convertedImg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
          convertedImg.getGraphics().drawImage(img, 0, 0, null);

          intValues = ((DataBufferInt) convertedImg.getRaster().getDataBuffer()).getData() ;

	    // Preprocess the image data from 0-255 int to normalized float based
	    // on the provided parameters.
          //bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

	    for (int i = 0; i < intValues.length; ++i) {
	      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
	      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
	      byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
	    }


	    // Copy the input data into TensorFlow.

	    inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);


	    // Run the inference call.

	    inferenceInterface.run(outputNames, logStats);


	    // Copy the output Tensor back into the output array.

	    outputLocations = new float[MAX_RESULTS * 4];
	    outputScores = new float[MAX_RESULTS];
	    outputClasses = new float[MAX_RESULTS];
	    outputNumDetections = new float[1];
	    inferenceInterface.fetch(outputNames[0], outputLocations);
	    inferenceInterface.fetch(outputNames[1], outputScores);
	    inferenceInterface.fetch(outputNames[2], outputClasses);
	    inferenceInterface.fetch(outputNames[3], outputNumDetections);


	    // Find the best detections.
	    final PriorityQueue<Recognition> pq =
	        new PriorityQueue<Recognition>(
	            1,
	            new Comparator<Recognition>() {
	              @Override
	              public int compare(final Recognition lhs, final Recognition rhs) {
	                // Intentionally reversed to put high confidence at the head of the queue.
	                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
	              }
	            });

	    // Scale them back to the input size.
	    for (int i = 0; i < outputScores.length; ++i) {
	      final RectF detection =
	          new RectF(
	              outputLocations[4 * i + 1] * inputSize,
	              outputLocations[4 * i] * inputSize,
	              outputLocations[4 * i + 3] * inputSize,
	              outputLocations[4 * i + 2] * inputSize);
	      pq.add(
	          new Classifier.Recognition("" + i, labels.get((int) outputClasses[i]), outputScores[i], detection));
	    }

	    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
	    for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
	      recognitions.add(pq.poll());
	    }
	    return recognitions;
	  }


	  @Override
	  public void close() {
	    inferenceInterface.close();
	  }
	}

