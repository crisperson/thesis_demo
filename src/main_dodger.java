import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

import org.apache.commons.math.MathException;
import org.apache.commons.math.special.Erf;

import com.googlecode.charts4j.AxisLabels;
import com.googlecode.charts4j.AxisLabelsFactory;
import com.googlecode.charts4j.AxisStyle;
import com.googlecode.charts4j.AxisTextAlignment;
import com.googlecode.charts4j.BarChart;
import com.googlecode.charts4j.BarChartPlot;
import com.googlecode.charts4j.Color;
import com.googlecode.charts4j.Data;
import com.googlecode.charts4j.DataUtil;
import com.googlecode.charts4j.Fills;
import com.googlecode.charts4j.GCharts;
import com.googlecode.charts4j.Line;
import com.googlecode.charts4j.LineChart;
import com.googlecode.charts4j.LineStyle;
import com.googlecode.charts4j.LinearGradientFill;
import com.googlecode.charts4j.Plots;
import com.googlecode.charts4j.Shape;
import static com.googlecode.charts4j.Color.*;

///

/////

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;

public class main_dodger {

	/**
	 * @param args
	 */
	static //Calendar var so we don't have to keep recreating it

	Calendar c = Calendar.getInstance();
	
	
	// Number of previous measures in linear predictor
	static int q = 30;
	// Weights in linear predictor
	static int[] weights;
	// Constant in the linear predictor
	static float b = 0.01f;
	static float linearLearningRate = 0.01f;
	static float linearMomentumFactor = 0.01f;

	// Sigma and Mu for the Real Time Predictor
	static double mu;
	static double sigma;
	static double epsilon;

	// Sigmua and Mu for the Context aware Predictor
	static int numberOfFeatures;
	static int k; // number of clusters
	static double[][] mu_array;
	static double[][] sigma_array;

	static int size;

	// Number of simulation iterations
	static int iterations = 1000;

	// K-Means Cluster
	static SimpleKMeans kMeans = new SimpleKMeans();

	public static void main(String[] args) {

		//String dateTest = "21/12/2012 23:59";
		//System.out.println(getTimeOfDay(dateTest));
		
		//if(true)
		//	return;
		
		ArrayList<Double> singleData;
		List<String[]> bulkData;

		ArrayList<String[]> anomalyArray = new ArrayList<String[]>();
		ArrayList<String[]> contextAnomalyArray = new ArrayList<String[]>();

		weights = new int[q - 1];
		mu = sigma = 0.0;
		epsilon = 0.0000001;

		numberOfFeatures = 4;
		k = 4; // number of clusters, should be empirically set

		mu_array = new double[k][numberOfFeatures];
		sigma_array = new double[k][numberOfFeatures];

		long startTimeCSV = System.nanoTime();
		
		//singleData = readCSV("electricity_byminute_attributenames", 1);//?????
		singleData = readCSV("dodgers", 2);
		long endTimeCSV = System.nanoTime();
		System.out.println("Running time of readCSV(): "
				+ (endTimeCSV - startTimeCSV) + "ns");
		
		// Pre-processing; break up input value as test and training
		// where the "training" set in this case is mimicing real-time sensor
		// inputs
		int eightyPercent = (int) (singleData.size() * .85);
		List<Double> singleDataTrain = singleData.subList(0, eightyPercent);
		List<Double> singleDataEvaluate = singleData.subList(eightyPercent,
				singleData.size());

		// System.out.println(singleData.size());
		// System.out.println(eightyPercent);
		// System.out.println(singleDataTrain.size());
		// System.out.println(singleDataEvaluate.size());

		long startTimeBULK = System.nanoTime();
		
		//bulkData = readCSVBulk("electricity_byminute_attributenames");
		bulkData = readCSVBulk("dodgers");
		
		long endTimeBULK = System.nanoTime();
		System.out.println("Running time of readCSVBULK(): "
				+ (endTimeBULK - startTimeBULK) + "ns");
		
		// Pre-processing; break up input values as test and training
		// where the "training" set in this case is mimicing real-time sensor
		// inputs
		eightyPercent = (int) (bulkData.size() * .85);
		List<String[]> bulkDataTrain = bulkData.subList(0, eightyPercent);
		List<String[]> bulkDataEvaluate = bulkData.subList(eightyPercent,
				singleData.size());

		// System.out.println(bulkData.size());
		// System.out.println(eightyPercent);
		// System.out.println(bulkDataTrain.size());
		// System.out.println(bulkDataEvaluate.size());
		
		// TODO: Generate clusters offline on all data
		long startTimeBuildClusters = System.nanoTime();
		
		//Instances arffData = buildClusters(k,
			//	"electricity_byminute_attributenames");
		Instances arffData = buildClusters(k,
				"dodgers");
		
		long endTimeBuildClusters = System.nanoTime();
		System.out.println("Running time of readCSVBULK(): "
				+ (endTimeBuildClusters - startTimeBuildClusters) + "ns");

		// TODO: Generate predictive model based on these clusters using all
		// data
		long startTimeBuildContextModel = System.nanoTime();
		// buildContextModels(bulkData, size, k);
		// should bulkDataTrain.size() be size?
		buildContextModels(bulkDataTrain, bulkDataTrain.size(), k);
		long endTimeBuildContextModel = System.nanoTime();
		System.out.println("Running time of buildContextModels(): "
				+ (endTimeBuildContextModel - startTimeBuildContextModel)
				+ "ns");

		long startTimeBuildRealModel = System.nanoTime();
		// TODO: Build real-time model
		// buildRealTimeModel(singleData, size);
		// should singleDataTrain.size() be size?
		
		buildRealTimeModel(singleDataTrain, singleDataTrain.size());
		long endTimeBuildRealModel = System.nanoTime();
		System.out.println("Running time of buildRealTimeModel(): "
				+ (endTimeBuildRealModel - startTimeBuildRealModel) + "ns");

		iterations = bulkDataEvaluate.size();
		//System.out.println(iterations);
		// Begin while
		// TODO: Iterate through real-time
		while (iterations-- > 0) {

			// Calculate if it is anomalous with the context free predictor
			// double value = 8.1; //test value, should always be anomalous
			// String value[] = { "18/11/2013 00:00", "8.1", "0.1", "0.2", "0.3"
			// };
			String value[] = bulkDataEvaluate.get(bulkDataEvaluate.size() - iterations-1);
//			String value[] = bulkDataEvaluate.get(1000 - iterations);

			long startTime = System.nanoTime();
			boolean isAnomalous = checkValueWithRealTime(Double
					.parseDouble(value[2]));
			if (isAnomalous ) {
				anomalyArray.add(value);
			}
			long endTime = System.nanoTime();
			 //System.out.println("Running time of realTimeCheckValue() iteration #"+(1000-iterations)+": "
			 //+ (endTime-startTime) + "ns");

			// TODO: Update real-time predictor parameters (not used here)
			updateRealTimeModel();
		}
		// End while

		// TODO: Check the anomalies based on the context predictor
		for (int a = 0; a < anomalyArray.size(); a++) {

			long startTime = System.nanoTime();
			String[] v = anomalyArray.get(a);
			
			//int cluster = findCluster(v, "electricity_byminute_attributenames",
				//	arffData);
			int cluster = findCluster(v, "dodger",
					arffData);
			
			
			// System.out.println(a + "/" + anomalyArray.size() + " : " +
			// cluster);
			
			long endTime = System.nanoTime();
			// System.out.println("Running time of findCluster(): " +
			// (endTime-startTime) + "ns");

			long startTime2 = System.nanoTime();
			
			if (checkAnomalyWithContext(v, cluster)) {
				contextAnomalyArray.add(v);
			}
			
			//if(checkAnomalyWithSimpleContext(v)) {
				//contextAnomalyArray.add(v);
			//}
			
			long endTime2 = System.nanoTime();
			// System.out.println("Running time of checkAnomalyWithContext(): "
			// + (endTime2-startTime2) + "ns");

		}

		// Now we have two arrays: contextAnomalyArray and anomalyArray
		// contextArray has the anomalies that are both contextual and point
		// anomalies
		// anomalyArray only has point anomalies
		// always anomalyArray >= contextArray
		// TODO: Compare results to see if any are seen as non-anomalous

		graphResults(bulkDataEvaluate);

		int[] anomalies = { anomalyArray.size(), 0, 0 };
		int[] contextuals = { contextAnomalyArray.size(), 0, 0 };
		graphBarResults(anomalies, contextuals);

		// Need to run simulation for a few other datasets but almost there.

		System.out.println("Number of point anomalies: " + anomalyArray.size());
		System.out.println("Number of contextual anomalies: "
				+ contextAnomalyArray.size());
		System.out.println("Number of point anomalies cleared: " + (anomalyArray.size() - contextAnomalyArray.size()));

		System.out.println("Simulation success.");
		
		//Print anomalous values
//		for(int i=0; i<anomalyArray.size(); i++)
//		{
//			for(int j=0; j<anomalyArray.get(i).length; j++)
//				System.out.print(anomalyArray.get(i)[j] + " ");
//			System.out.println();
//		}
//		
		//printArray(anomalyArray);
		//System.out.println("--");
		//printArray(contextAnomalyArray);
		
//		for(int i=0; i<contextAnomalyArray.size(); i++)
//		{
//			for(int j=0; j<contextAnomalyArray.get(i).length; j++)
//				System.out.print(contextAnomalyArray.get(i)[j] + " ");
//			System.out.println();
//		}
	}

	public static void printArray(ArrayList<String[]> s)
	{
		for(int i=0; i<s.size(); i++)
		{
			for(int j=0; j<s.get(i).length; j++)
				System.out.print(s.get(i)[j] + " ");
			System.out.println();
		}
	}
	
	// ---------------------------------------------------------------------
	// REAL-TIME POINT DETECTION
	// ---------------------------------------------------------------------
	// Build real-time model
	public static void buildRealTimeModel(List<Double> singleDataTrain, int m) {
		
		double intermediate = 0;

		for (int i = 0; i < m; i++) {
			intermediate += singleDataTrain.get(i);
		}

		mu = (1 / (double) m) * intermediate;

		intermediate = 0;
		
		for (int i = 0; i < m; i++) {
//intermediate += Math.pow((singleDataTrain.get(i) - mu), 2);
			intermediate += Math.abs((singleDataTrain.get(i) - mu));
		}
	//	System.out.println(intermediate);
		sigma = (1 / (double) m) * intermediate;
		
		//System.out.println("mu"+mu + " " + "sigma" + sigma);

	}

	// Check the predicted value of the model with the actual value
	public static boolean checkValueWithRealTime(double value) {

		double predictedValue = 0.0;

		predictedValue = (1 / (Math.sqrt(2 * Math.PI * sigma)))
				* Math.exp(-1 * (Math.pow((value - sigma), 2) / (2 * sigma)));
	
		if (predictedValue < epsilon)
			return true;
		return false;

	}

	// ---------------------------------------------------------------------
	// CONTEXT-AWARE ANOMALY DETECTION
	// ---------------------------------------------------------------------
	// Build context-aware clusters
	// i.e. use the meta data associated with the sensors to build "similar"
	// sensors
	public static Instances buildClusters(int k, String location) {
		try {
			// load CSV
			CSVLoader loader = new CSVLoader();
			loader.setSource(new File(location + ".csv"));
			Instances data = loader.getDataSet();
			
			// save ARFF
			ArffSaver saver = new ArffSaver();
			saver.setInstances(data);
			saver.setFile(new File(location + ".arff"));
			saver.writeBatch();

			BufferedReader reader = new BufferedReader(new FileReader(location
					+ ".arff"));
			Instances arffData = new Instances(reader);

			int eightyPercent = (int) (arffData.numInstances() * .85);
			Instances arffDataPercentage = new Instances(arffData, 0,
					eightyPercent);

			kMeans.setSeed(10);
			kMeans.setPreserveInstancesOrder(true);
			kMeans.setNumClusters(k);
			// kMeans.buildClusterer(arffData);
			kMeans.buildClusterer(arffDataPercentage);
			
			Instances instances = kMeans.getClusterCentroids();
			for ( int i = 0; i < instances.numInstances(); i++ ) {
			    // for each cluster center
			    Instance inst = instances.instance( i );
			    // as you mentioned, you only had 1 attribute
			    // but you can iterate through the different attributes
			    double value = inst.value( 0 );
			    System.out.println( "Value for centroid " + i + ": " + value );
			    value = inst.value( 1 );
			    System.out.println( "Value for centroid " + i + ": " + value );
			    value = inst.value( 2 );
			  //  System.out.println( "Value for centroid " + i + ": " + value );
			  //  value = inst.value( 3 );
			  //  System.out.println( "Value for centroid " + i + ": " + value );
			  //  value = inst.value( 4 );
			  //  System.out.println( "Value for centroid " + i + ": " + value );
			  //  value = inst.value( 5 );
			 //   System.out.println( "Value for centroid " + i + ": " + value );
			}
			return arffData;

			// Evaluation

			// ClusterEvaluation eval = new ClusterEvaluation();
			// eval.setClusterer(kMeans);
			// eval.evaluateClusterer(arffData);

			// Instances centroids = kMeans.getClusterCentroids();
			// for ( int i = 0; i < centroids.numInstances(); i++ ) {
			// // for each cluster center
			// Instance inst = centroids.instance( i );
			// // as you mentioned, you only had 1 attribute
			// // but you can iterate through the different attributes
			// double value = inst.value( 0 );
			// System.out.println( "Value for centroid " + i + ": " + value );
			// }
			//
			// int[] numberOfInstances = kMeans.getClusterSizes();
			//
			// for(int i=0; i<numberOfInstances.length; i++)
			// {
			// System.out.println("Instances in cluster " + i + " : " +
			// numberOfInstances[i]);
			// }

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return null;

	}

	// Find the cluster of a new value, so we can evaluate the appropriate
	// context anomaly predictor
	public static int findCluster(String[] data, String location,
			Instances dataset) {
		int cluster = -1;
		try {

			// // Create empty instance with three attribute values
			// Instance inst = new Instance(4);
			//
			// BufferedReader reader = new BufferedReader(
			// new FileReader(location+".arff"));
			// Instances arffData = new Instances(reader);
			//
			// // Set instance's dataset to be the dataset "race"
			// inst.setDataset(arffData);
			//
			// // Set instance's values for the attributes "length", "weight",
			// and "position"
			// //inst.setValue(new Attribute("READING_DATE"),
			// data[0].toString());
			// inst.setValue(new Attribute("1st Fl HVAC Admin-Total kWh"),
			// Double.parseDouble(data[1]));
			// inst.setValue(new Attribute("1st Fl HVAC R&D-Total kWh"),
			// Double.parseDouble(data[2]));
			// inst.setValue(new Attribute("2nd Fl HVAC Exec-Total kWh"),
			// Double.parseDouble(data[3]));
			// inst.setValue(new Attribute("2nd Fl HVAC Forge-Total kWh"),
			// Double.parseDouble(data[4]));

			long startTime1 = System.nanoTime();

			// BufferedReader reader = new BufferedReader(new
			// FileReader(location
			// + ".arff"));
			// Instances dataset = new Instances(reader);

			long endTime1 = System.nanoTime();
			// System.out.println("Running tmie of BufferedReader() call: " +
			// (endTime1-startTime1) + "ns");

			Instance xyz = new Instance(dataset.numAttributes());
			xyz.setDataset(dataset);
			xyz.setValue(dataset.attribute(0), Integer.parseInt(data[0]));
			xyz.setValue(dataset.attribute(1), Double.parseDouble(data[1]));
			xyz.setValue(dataset.attribute(2), Double.parseDouble(data[2]));
			xyz.setValue(dataset.attribute(3), Double.parseDouble(data[3]));
		//	xyz.setValue(dataset.attribute(4), Double.parseDouble(data[4]));
		//	xyz.setValue(dataset.attribute(5), Double.parseDouble(data[5]));
			
			//ClusterEvaluation eval = new ClusterEvaluation();
			//eval.setClusterer(kMeans);

			long startTime = System.nanoTime();

			cluster = kMeans.clusterInstance(xyz);
			
			//ALWAYS FINDS INSTANCE AS CLUSTER 1???
			//System.out.println(cluster);
		
			long endTime = System.nanoTime();
			// System.out.println("Running time of clusterInstance(): " +
			// (endTime-startTime) + "ns");
			
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return cluster;
	}

	// Build context-aware predictor based on clusters
	// Just a Gaussian predictor, can look at changing this
	public static void buildContextModels(List<String[]> data, int m, int k) {
		// k=number of predictors (i.e. number of clusters)

		// shouldn't JUST BE data here, need to build different clusters for
		// different sets of the input
		// need to use the k'th subset of the data here
		ArrayList<ArrayList<String[]>> dataIn = new ArrayList<ArrayList<String[]>>();
		for (int z = 0; z < k; z++) {
			ArrayList<String[]> temp = new ArrayList<String[]>();
			dataIn.add(temp);
		}

		try {
			int[] assignments = kMeans.getAssignments();

			int z = 0;
			for (int clusterNum : assignments) {
				if(z<m) {
					dataIn.get(clusterNum).add(data.get(z));
				}
				//if(clusterNum == 0)
				//	System.out.println(data.get(z)[1] + " " +  data.get(z)[2] + " " +  data.get(z)[3] + " " + data.get(z)[4]);
				
				// System.out.printf("Instance %d -> Cluster %d", z,
				// clusterNum);
				z++;
			}

			// for(z=0; z<k; z++)
			// {
			// System.out.println("DataIn for cluster " + z + " size is: " +
			// dataIn.get(z).size());
			// }

			for (int a = 0; a < k; a++) {
				double intermediate = 0;

				for (int j = 1; j < numberOfFeatures; j++) {
					
					intermediate = 0;
					
					// for (int i = 0; i < m; i++) {
					for (int i = 0; i < dataIn.get(a).size(); i++) {
						
						// intermediate += Double.parseDouble(data.get(i)[j]);
						intermediate += Double
								.parseDouble(dataIn.get(a).get(i)[j]);
					}
					
					mu_array[a][j] = (1 / (double) dataIn.get(a).size()) * intermediate;
				
				}

				for (int j = 1; j < numberOfFeatures; j++) {
					intermediate = 0;
					// for (int i = 0; i < m; i++) {
					for (int i = 0; i < dataIn.get(a).size(); i++) {
						intermediate += Math
								// .pow((Double.parseDouble(data.get(i)[j]) -
								// mu_array[a][j]),
								.pow((Double
										.parseDouble(dataIn.get(a).get(i)[j]) - mu_array[a][j]),
										2);
					}
					sigma_array[a][j] = (1 / (double) dataIn.get(a).size()) * intermediate;
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static void buildContextModelsDeprecated(List<String[]> data, int m,
			int k) {
		// k=number of predictors (i.e. number of clusters)
		try {

			for (int a = 0; a < k; a++) {
				double intermediate = 0;

				for (int j = 1; j < numberOfFeatures; j++) {
					intermediate = 0;
					for (int i = 0; i < m; i++) {
						intermediate += Double.parseDouble(data.get(i)[j]);
					}
					mu_array[a][j] = (1 / (double) m) * intermediate;
				}

				for (int j = 1; j < numberOfFeatures; j++) {
					intermediate = 0;
					for (int i = 0; i < m; i++) {
						intermediate += Math
								.pow((Double.parseDouble(data.get(i)[j]) - mu_array[a][j]),
										2);
					}
					sigma_array[a][j] = (1 / (double) m) * intermediate;
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	// Check the predicted anomalous value with the contextual predictor's
	// predicted value
	// Evaluating the Gaussian predictor
	public static boolean checkAnomalyWithContext(String[] value, int cluster) {
		double predictedValue = 1.0;
		
		//System.out.println(value[0] + " with value " + value[1]);
		
		for (int j = 1; j < numberOfFeatures; j++) {

			// System.out.println(sigma_array[cluster][j]);

			predictedValue *= (1 / (Math.sqrt(2 * Math.PI
					* sigma_array[cluster][j])))
					* Math.exp(-1
							* (Math.pow(
									(Double.parseDouble(value[j]) - sigma_array[cluster][j]),
									2) / (2 * sigma_array[cluster][j])));

		}
		
		// System.out.println("Checking contextual anomaly");
		if (predictedValue < epsilon) {
			return true;

		}
		return false;
	}

	public static boolean checkAnomalyWithSimpleContext(String[] value) {
		
		//compute distance between it and other values
	//	System.out.println(value[0] + " with value " + value[1]);

//		double distance = 0.0;
//		
//		for(int i=2; i<value.length; i++)
//		{
//			distance += Math.pow( (Double.parseDouble(value[1]) - Double.parseDouble(value[i])), 2);
//		}
//			
//		distance = Math.sqrt(distance);

		//System.out.print(distance + " >? " + sigma);
		

		double sum = 0;
		//-1 to not include that last column
	    for (int i = 2; i < value.length-1; i++) {
	        sum += Double.parseDouble(value[i]);
	    }
	    double mean = sum / (value.length-2);
	    
	    double temp = 0;
        for(int i=2; i<value.length-1; i++) {
            temp += (mean-Double.parseDouble(value[i]))*(mean-Double.parseDouble(value[i]));
        }
        double stdev = Math.sqrt(temp/(value.length-2));
	    
		double zscore = (Double.parseDouble(value[1]) - mean)/stdev;
		
		
		 zscore = zscore / Math.sqrt(2.0);
	        double lPvalue = 0.0;
	        try {
	            lPvalue = Erf.erf(zscore);
	        } catch (MathException e) {
	            e.printStackTrace();
	        }

		//System.out.println(zscore + " " + lPvalue);

		//still anomalies (i.e. CONTEXTUALLY ANOMALOUS)
		if(lPvalue > 0.8)
			return true;
		
		//no longer an anomaly (i.e. NOT CONTEXTUALLY ANOMALOUS)
		return false;
		
//		
//		if(distance > sigma*5) {
//		
//			System.out.println(" true");
//			return true;
//		}
//
//		System.out.println(" false");
//		return false;

	}
	
	public static void graphBarResults(int[] anomalies, int[] contextuals) {

		// Defining data plots.
		BarChartPlot team1 = Plots.newBarChartPlot(
				Data.newData(anomalies[0]),
				//Data.newData(anomalies[0], anomalies[1], anomalies[2]),
					BLUEVIOLET, "Point Anomalies");
		BarChartPlot team2 = Plots.newBarChartPlot(
				Data.newData(anomalies[0]-contextuals[0]),
				//Data.newData(contextuals[0], contextuals[1], contextuals[2]),
				ORANGERED, "Contextual Anomalies");
		
		// BarChartPlot team3 = Plots.newBarChartPlot(Data.newData(10, 20, 30),
		// LIMEGREEN, "Team C");

		// Instantiating chart.
		BarChart chart = GCharts.newBarChart(team1, team2);

		// Defining axis info and styles
		AxisStyle axisStyle = AxisStyle.newAxisStyle(BLACK, 13,
				AxisTextAlignment.CENTER);
		AxisLabels score = AxisLabelsFactory.newAxisLabels("Occurrences", 50.0);
		score.setAxisStyle(axisStyle);
		AxisLabels year = AxisLabelsFactory.newAxisLabels("Dataset", 50.0);
		year.setAxisStyle(axisStyle);

		// Adding axis info to chart.
		//chart.addXAxisLabels(AxisLabelsFactory.newAxisLabels("Powersmiths",
				//"UCI 1", "UCI 2"));
		chart.addXAxisLabels(AxisLabelsFactory.newAxisLabels("Powersmiths"));
		chart.addYAxisLabels(AxisLabelsFactory.newNumericRangeAxisLabels(0, 60));
		chart.addYAxisLabels(score);
		chart.addXAxisLabels(year);

		chart.setSize(600, 450);
		chart.setBarWidth(350);
		chart.setSpaceWithinGroupsOfBars(20);
		chart.setDataStacked(true);
		chart.setTitle("Point vs. Contextual Anomalies", BLACK, 16);
		chart.setGrid(100, 10, 3, 2);
		chart.setBackgroundFill(Fills.newSolidFill(ALICEBLUE));
		LinearGradientFill fill = Fills.newLinearGradientFill(0, LAVENDER, 100);
		fill.addColorAndOffset(WHITE, 0);
		chart.setAreaFill(fill);
		String url = chart.toURLString();

		System.out.println(url);
	}

	// TODO:
	public static void graphResults(List<String[]> data) {

		// Defining lines
		final int NUM_POINTS = 500;
		final double[] competition = new double[NUM_POINTS];
		final double[] mywebsite = new double[NUM_POINTS];
		for (int i = 0; i < NUM_POINTS; i++) {
			competition[i] = Double.parseDouble(data.get(i)[1]);
			mywebsite[i] = Double.parseDouble(data.get(i)[1] + 1);

			// competition[i] = 100-(Math.cos(30*i*Math.PI/180)*10 + 50)*i/20;
			// mywebsite[i] = (Math.cos(30*i*Math.PI/180)*10 + 50)*i/20;
		}

		Data data1 = DataUtil.scale(competition);
		// Data data2 = DataUtil.scale(mywebsite);

		Line line1 = Plots.newLine(data1, Color.newColor("CA3D05"));
		// Line line1 = Plots.newLine(data1, Color.newColor("CA3D05"), "Data");
		line1.setLineStyle(LineStyle.newLineStyle(3, 1, 0));
		line1.addShapeMarkers(Shape.DIAMOND, Color.newColor("CA3D05"), 12);
		line1.addShapeMarkers(Shape.DIAMOND, Color.WHITE, 8);
		// Line line2 = Plots.newLine(data2, SKYBLUE, "data");
		// line2.setLineStyle(LineStyle.newLineStyle(3, 1, 0));
		// line2.addShapeMarkers(Shape.DIAMOND, SKYBLUE, 12);
		// line2.addShapeMarkers(Shape.DIAMOND, Color.WHITE, 8);

		// Defining chart.
		LineChart chart = GCharts.newLineChart(line1);
		chart.setSize(600, 450);
		chart.setTitle("Powersmiths Point Anomalies", WHITE, 12);
		chart.addHorizontalRangeMarker(0, 10, Color.newColor(RED, 30));
		chart.addHorizontalRangeMarker(80, 100, Color.newColor(RED, 30));
		// chart.addVerticalRangeMarker(70, 90, Color.newColor(GREEN, 30));
		chart.setGrid(0.01, 0.01, 1, 1);

		// Defining axis info and styles
		AxisStyle axisStyle = AxisStyle.newAxisStyle(WHITE, 12,
				AxisTextAlignment.CENTER);
		AxisLabels xAxis3 = AxisLabelsFactory.newAxisLabels("Reading Instance",
				50.0);
		xAxis3.setAxisStyle(AxisStyle.newAxisStyle(WHITE, 12,
				AxisTextAlignment.CENTER));
		AxisLabels yAxis = AxisLabelsFactory.newAxisLabels("", "0.2", "0.4",
				"0.6", "0.8", "1.0");
		yAxis.setAxisStyle(axisStyle);
		AxisLabels yAxis2 = AxisLabelsFactory.newAxisLabels("Reading Value",
				50.0);
		yAxis2.setAxisStyle(AxisStyle.newAxisStyle(WHITE, 12,
				AxisTextAlignment.CENTER));
		yAxis2.setAxisStyle(axisStyle);

		// Adding axis info to chart
		chart.addXAxisLabels(xAxis3);
		chart.addYAxisLabels(yAxis);
		chart.addYAxisLabels(yAxis2);

		// Defining background and chart fills.
		chart.setBackgroundFill(Fills.newSolidFill(Color.newColor("1F1D1D")));
		LinearGradientFill fill = Fills.newLinearGradientFill(0,
				Color.newColor("363433"), 100);
		fill.addColorAndOffset(Color.newColor("2E2B2A"), 0);
		chart.setAreaFill(fill);
		String url = chart.toURLString();
		System.out.println(url);
	}

	// TODO: NOT USED CURRENTLY: Update parameters of real-time model
	public static void updateRealTimeModel() {

	}

	// TODO: NOT USED CURRENTLY
	public static void runSimulation() {

	}

	public static ArrayList<Double> readCSV(String location, int columnNumber) {
		CSVReader reader;
		List<String[]> myEntries = null;
		ArrayList<Double> valueColumn = new ArrayList<Double>();
		try {
			// Need these lines for a non-dynamic array
			// reader = new CSVReader(new FileReader(location + ".csv"));
			// myEntries = reader.readAll();

			int i = 0;
			String[] nextLine;
			reader = new CSVReader(new FileReader(location + ".csv"));
			reader.readNext();
			//reader.readNext();

			while ((nextLine = reader.readNext()) != null) {
				try {
					valueColumn.add(Double.parseDouble(nextLine[columnNumber]));
				} catch (Exception e) {
					valueColumn.add(0.0);
				}
				i++;
			}
			size = i;

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return valueColumn;

	}

	public static List<String[]> readCSVBulk(String location) {
		CSVReader reader;
		List<String[]> myEntries = null;

		List<String[]> titles = null;
		
		try {
			reader = new CSVReader(new FileReader(location + ".csv"));
			myEntries = reader.readAll();
			
			titles = myEntries.subList(0, 1);		
			
			if(titles.get(0).length < 4) {
			//if(titles.get(0).length < 6) {
				List<String> temp = new ArrayList<String>(Arrays.asList(titles.get(0)));
				temp.add("TimeOfDay");			
				titles.set(0, temp.toArray(new String[temp.size()]));
 			}
			
			myEntries = myEntries.subList(1, myEntries.size());
			
			for (int i = 0; i < myEntries.size(); i++) {
				
				for (int j = 0; j < myEntries.get(i).length; j++) {
					
					
					if (myEntries.get(i)[j] == null
							|| myEntries.get(i)[j].equals(""))
						myEntries.get(i)[j] = "0.0";
				}

				//if(myEntries.get(i).length < 6) {
				if(myEntries.get(i).length < 4) {
					List<String> temp = new ArrayList<String>(Arrays.asList(myEntries.get(i)));
					temp.add( getTimeOfDay(myEntries.get(i)[1]) + "" );
					myEntries.set(i, temp.toArray(new String[temp.size()]));
				}
				myEntries.get(i)[0] = getDayOfWeek(myEntries.get(i)[0]) + "";
			}
			
			for(int i=0; i<myEntries.size(); i++)
			{
				for(int j=0; j<myEntries.get(i).length; j++)
				{
					if(j==1)
					{
						myEntries.get(i)[j] = convertToDecimal(myEntries.get(i)[j])+"";
					}
				}
			}
			
			reader.close();
		
			//pre-processing step for powersmiths
			//update first column dates to be day of week
			CSVWriter writer = new CSVWriter(new FileWriter(location + ".csv"));
			writer.writeNext(titles.get(0));
			for(int i=0; i<myEntries.size(); i++)
				writer.writeNext(myEntries.get(i));
			writer.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return myEntries;
	}
	
	public static int convertToDecimal(String s)
	{
		if(!s.contains(":")) return Integer.parseInt(s);
		int value=-1;
		
		String[] split = s.split( ":");
		value = (Integer.parseInt(split[0]) * 60) + (Integer.parseInt(split[1]) * 60);
		
		return value;
	}
	
	public static int getDayOfWeek(String value)
	{
		if(value.length() < 9 || value == null)
		{
			return Integer.parseInt(value);
		}
		
		try {
			c.setTime(new SimpleDateFormat("dd/MM/yyyy").parse(value));

			return c.get(Calendar.DAY_OF_WEEK);
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return -1;
		}
	}
	public static int getTimeOfDay(String value)
	{
		if(value.length() < 2 || value == null) 
			return Integer.parseInt(value);
		
		SimpleDateFormat parser = new SimpleDateFormat("hh:mm");
		try {
			c.setTime(parser.parse(value));
			int temp = c.get(Calendar.HOUR_OF_DAY);
			if(temp >= 0 && temp <= 8)
				return 0;		//morning
			else if(temp > 8 && temp <= 16)
				return 1;		//midday
			else
				return 2;		//night
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return -1;
		}
	}
}

// preprocess TIME of day as well