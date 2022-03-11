/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetesnn;

import java.util.Random;

/**
 *
 * @author Wind10
 */
public class BP{

    protected double[][] output; //output (value) setiap neuron
    protected double[][][] weights;
    protected double[][] bias;
    protected double[][] error_signal; //delta setiap neuron hidden dan output
    protected double[][] output_derivative; //turunan pertama fungsi aktivasi
    protected int[] class_train; //hasil klasifikasi train
    protected int[] class_test; //hasil klasifikasi test
    protected double[] MSEarr;
    
    public final int[] NETWORK_LAYER_SIZES; //layer
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;
    
    protected double initialMSE;
    protected double finalMSE;

    public BP(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            if (i > 0) { //inisialisasi bobot dan bias
                this.bias[i] = BP.createRandomArray(NETWORK_LAYER_SIZES[i], 0, 0);
                this.weights[i] = BP.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -0.5, 0.5);
            }
        }
    }
    
    public static double[] createRandomArray(int size, double lower_bound, double upper_bound){
        if(size < 1){
            return null;
        }
        double[] ar = new double[size];
        
        for(int i = 0; i < size; i++){
            ar[i] = randomValue(lower_bound,upper_bound);
        }
        return ar;
    }

    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound){
        if(sizeX < 1 || sizeY < 1){
            return null;
        }
        double[][] ar = new double[sizeX][sizeY];
        for(int i = 0; i < sizeX; i++){
            ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }
        return ar;
    }

    public static double randomValue(double lower_bound, double upper_bound){
        //return Math.random()*(upper_bound-lower_bound) + lower_bound;
        return new Random().nextDouble() * (upper_bound-lower_bound) + lower_bound;
    }   

    public int[] getClass_train() {
        return class_train;
    }

    public int[] getClass_test() {
        return class_test;
    }

    public double[][][] getWeights() {
        return weights;
    }
    
    protected double[] forward(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        this.output[0] = input; //pass input data
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) { //setiap layer
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) { //setiap neuron pada layer

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) { //setiap neuraon pada layer sebelumnya
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                this.output[layer][neuron] = 1d / (1 + Math.exp(-sum));
                this.output_derivative[layer][neuron] = this.output[layer][neuron] * (1 - this.output[layer][neuron]);
            }
        }
        return this.output[NETWORK_SIZE - 1]; //return neuron output
    }
    
    protected int thresholdOutput() {
        int result;
        
        if(this.output[NETWORK_SIZE-1][0] >= 0.5)
            return result = 1;
        else
            return result = 0;
        
    }
    
    protected void backpropError(double[] target) {
        for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[this.NETWORK_SIZE - 1]; neuron++) { //setiap neuron pada layer output
            this.error_signal[this.NETWORK_SIZE - 1][neuron] = (this.output[this.NETWORK_SIZE - 1][neuron] - target[neuron])
                    * this.output_derivative[this.NETWORK_SIZE - 1][neuron];
        }
        for (int layer = this.NETWORK_SIZE - 2; layer > 0; layer--) { //setiap layer hidden
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) { //setiap neuron pada layer hidden
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < this.NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) { //setiap neuron pada layer selanjutnya
                    sum += this.weights[layer + 1][nextNeuron][neuron] * this.error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * this.output_derivative[layer][neuron];
            }
        }
    }

    private void updateWeights(double alpha) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) { //setiap layer
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) { //setiap neuron pada layer

                double delta = -alpha * error_signal[layer][neuron];

                bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) { //setiap neuron pada layer sebelumnya
                    weights[layer][neuron][prevNeuron] += delta * output[layer - 1][prevNeuron];
                }
            }
        }
    }
    
    public void test(DataSet set, ConfusionMatrix cf) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        
        class_test = new int[set.size()];
        
        for (int data=0; data<set.size(); data++) {
            forward(set.getInput(data));
            class_test[data] = thresholdOutput();
            cf.setConfusionMatrix(class_test[data], set.getOutput(data)[0]);
        }
        cf.computePerformance();
        System.out.println("\nTEST");
        System.out.println("Akurasi = "+cf.getAccuracy());
        System.out.println("Sensitifitas = "+cf.getSensitivity());
        System.out.println("Spesifisitas = "+cf.getSpecificity());
        System.out.println("Precision = "+cf.getPrecision());
        System.out.println("FMeasure = "+cf.getFmeasure());
    }
 
    public void trainBP(DataSet set, double learning_rate, int maxIterations, double MSEt) { //train stochastic
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        //MSEarr = new double[maxIterations];
        class_train = new int[set.size()];
        
        int i = 1;
        double MSE = 1;
        
        printWeights();
        while(MSEt <= MSE && i <= maxIterations) {
            for (int data = 0; data < set.size(); data++) {
                train(set.getInput(data), set.getOutput(data), learning_rate);
            }
            MSE = MSE(set);
            if(i==1)
                initialMSE = MSE;
            //MSEarr[i-1] = MSE;
            
            System.out.println("Iterasi : "+ i +" || MSE : "+ MSE);
            i++;
        }
        finalMSE = MSE;
        //writeMSE(MSEarr);
        System.out.println("\nTRAIN");
        System.out.println("Learning rate : " + learning_rate);
        System.out.println("Initial MSE : " + initialMSE);
        System.out.println("Final MSE : " + finalMSE);
        System.out.println("Iterasi : " + (i-1));
        
        ConfusionMatrix cf = new ConfusionMatrix();
        for (int data = 0; data < set.size(); data++) {
            forward(set.getInput(data));
            class_train[data] = thresholdOutput();
            cf.setConfusionMatrix(class_train[data], set.getOutput(data)[0]);
        }
        cf.computePerformance();
        System.out.println("Akurasi = "+cf.getAccuracy());
        
    }

    private void train(double[] input, double[] target, double learning_rate) { //train
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return;
        }
        forward(input);
        backpropError(target);
        updateWeights(learning_rate);
    }

    private double SE(double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return 0;
        }
        forward(input);
        double v = 0;
        
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - output[NETWORK_SIZE - 1][i]) * (target[i] - output[NETWORK_SIZE - 1][i]);
        }
        return v / target.length;
    }

    private double MSE(DataSet set) {
        double v = 0;
        for (int i = 0; i < set.size(); i++) {
            v += SE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }
    
    public void printWeights() {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                System.out.println("Bias || layer | neuron || = ");
                System.out.println(bias[layer][neuron]);
                System.out.println("Bobot || layer | neuron | prevNeuron || = ");
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    System.out.println(weights[layer][neuron][prevNeuron]);
                }
            }
        }
    }
       
    public void printOutputNeuron() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
                System.out.println("Layer : " + layer + " Neuron : " + neuron + " = " + output[layer][neuron]);
            }
        }
    }
    
    /*
    public void writeMSE(double[] MSE) {
        try {
            File f = new File("MSEbp.txt");
            if(!f.exists()){
                System.out.println("Make new file");
                f.createNewFile();
            }
            
            FileWriter fw = new FileWriter(f, true);
            for(int i = 0; i < MSE.length; i++) {
                fw.write(Double.toString(MSE[i])+"\n");
            }
            fw.close();
        } catch (IOException ex) {
            Logger.getLogger(ConfusionMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }
    }*/
    
        /*
    public void test2(DataSet set, Network net, boolean isTest) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        
        if(isTest)
            class_test = new int[set.size()];
        else
            class_train = new int[set.size()];
        
        for (int data=0; data<set.size(); data++) {
            forward(set.getInput(data));
            if(isTest){
                class_test[data] = thresholdOutput();
                net.setConfusionMatrix(class_test[data], set.getOutput(data)[0]);
            }
            else {
                class_train[data] = thresholdOutput();
                net.setConfusionMatrix(class_train[data], set.getOutput(data)[0]);
            }
        }
        net.computePerformance();
        System.out.println("Akurasi = "+net.getAccuracy());
        System.out.println("Sensitifitas = "+net.getSensitivity());
        System.out.println("Spesifisitas = "+net.getSpecificity());
        System.out.println("Precision = "+net.getPrecision());
        System.out.println("FMeasure = "+net.getFmeasure());
    }*/
/*    
    public void trainBatch(DataSet set, int epoch, int batch_size, double learning_rate) { //train batch
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        for (int i = 0; i < epoch; i++) {
            DataSet batch = set.extractBatch(batch_size);
            for (int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), learning_rate);
            }
            System.out.println(MSE(batch));
        }
    }
    */ 
    /*
    public double SSE(DataSet set) { //hitung SSE dengan forward prop
        double v = 0;
        for (int i = 0; i < set.size(); i++) {
            v += SE(set.getInput(i), set.getOutput(i));
        }
        return v / 2d;
    }
    */
    
    /*
    public void saveModel(){
        String layer = Arrays.toString(NETWORK_LAYER_SIZES);
        String weights = Arrays.deepToString(this.weights);
        String bias = Arrays.deepToString(this.bias);
        String MSE = Double.toString(getFinalMSE());
        
        try {
            List<String> lines = Arrays.asList(layer, weights, bias, MSE);
            Path file = Paths.get("models.txt");
            Files.write(file, lines, StandardCharsets.UTF_8);
        } catch (IOException ex) {
            Logger.getLogger(BP.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    public void loadModel(){
        try {
            List list = readByJava8("models.txt");
            list.forEach(System.out::println);
            Arrays.asList(list);
            int[] a = (int[]) list.get(1);
            System.out.println(Arrays.toString(a));
        } catch (Exception e){
            e.printStackTrace();
        }
        
    }
    
    private static List readByJava8(String fileName) throws IOException {
        List<String> result;
        try (Stream<String> lines = Files.lines(Paths.get(fileName))) {
            result = lines.collect(Collectors.toList());
        }
        return result;

    }
    
    public static void main(String[] args) {
        BP bp = new BP(8,5,1);
        
        bp.loadModel();
    }
    */
}
