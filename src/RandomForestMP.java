import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.RandomForest;
import org.codehaus.janino.Java;


import java.util.HashMap;
import java.util.regex.Pattern;

public final class RandomForestMP {

    private static class ParseFeature implements Function<String, Vector> {
        private static final Pattern SPACE = Pattern.compile(",");

        public Vector call(String line) {
            String[] token = SPACE.split(line);
            double[] point = new double[token.length-1];
            for (int i = 0; i < token.length - 1; ++i) {
                point[i-1] = Double.parseDouble(token[i]);
            }
            return Vectors.dense(point);
        }
    }

    private static class ParseLabel implements Function<String, Integer> {
        private static final Pattern SPACE = Pattern.compile(",");

        public Integer call(String line) {
            String[] tok = SPACE.split(line);
            return Integer.parseInt(tok[tok.length-1]);
        }
    }

    private static class ParseData implements Function<String, LabeledPoint> {
        private static final Pattern SPACE = Pattern.compile(",");

        public LabeledPoint call(String line) {
            String[] token = SPACE.split(line);
            double[] point = new double[token.length-1];
            for (int i = 0; i < token.length - 1; ++i) {
                point[i-1] = Double.parseDouble(token[i]);
            }
            Vector vector = Vectors.dense(point);
            Integer label = Integer.parseInt(token[token.length-1]);
            return new LabeledPoint(label, vector);
        }
    }


    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                    "Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        final RandomForestModel model;

        Integer numClasses = 2;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

		// TODO
        JavaRDD<String> training_data = sc.textFile(training_data_path);
        // JavaRDD<Vector> training_x = training_data.map(new ParseFeature());
        // JavaRDD<Integer> training_y = training_data.map(new ParseLabel());
        JavaRDD<LabeledPoint> training = training_data.map(new ParseData());

        model = RandomForest.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);

        JavaRDD<String> test_data = sc.textFile(test_data_path);
        JavaRDD<Vector> test_x = test_data.map(new ParseFeature());
        JavaRDD<Integer> test_y = test_data.map(new ParseLabel());
        JavaRDD<LabeledPoint> test = training_data.map(new ParseData());

        JavaRDD<LabeledPoint> results = test_x.map(new Function<Vector, LabeledPoint>() {
            public LabeledPoint call(Vector points) {
                return new LabeledPoint(model.predict(points), points);
            }
        });

        results.saveAsTextFile(results_path);

        sc.stop();
    }

}
