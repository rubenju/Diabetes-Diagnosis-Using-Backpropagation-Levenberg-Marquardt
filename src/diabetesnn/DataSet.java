/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetesnn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;

/**
 *
 * @author Wind10
 */
public class DataSet {

    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    
    private ArrayList<double[][]> data = new ArrayList<double[][]>();
    private ArrayList<double[][]> data2 = new ArrayList<double[][]>();

    public DataSet(int INPUT_SIZE, int OUTPUT_SIZE) {
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
    }
    
    public void readCSV() {
        //String csvFile = "diabetes-m-mm.csv";
        String csvFile = "diabetes.csv";
        //String csvFile = "diabetes-m-mm-sample.csv";
        
        BufferedReader br = null;
        String line = "";
        boolean first = false;
        int row = 0;

        try {
            br = new BufferedReader(new FileReader(csvFile));
            while ((line = br.readLine()) != null) {
                // use comma as separator
                double[] x = new double[INPUT_SIZE];
                double[] y = new double[OUTPUT_SIZE];
                String[] input = line.split(",");
                if (first) {
                    for (int i = 0; i < input.length; i++) {
                        if (i != input.length - 1) {
                            x[i] = Double.parseDouble(input[i]);
                        } else {
                            y[0] = Double.parseDouble(input[i]);
                            addData(x, y);
                        }
                    }
                    row++;
                }
                first = true;
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public void copyList() {
        for (double[][] r : data) {
            data2.add(r);
        }
    }

    public void addData(double[] in, double[] expected) {
        if (in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE) {
            return;
        }
        data.add(new double[][]{in, expected});
    }

    public DataSet createTrainData(float trainData) {
        int num = data.size();
        int train = Math.round(num * trainData);

        DataSet set = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
        //System.out.println(train);
        for (int i = 0; i < train; i++) {
            set.addData(this.getInput(i), this.getOutput(i));
        }

        return set;
    }

    public DataSet createTestData(float testData) {
        int num = data.size();
        int train = Math.round(num * (1 - testData));
        //int test = Math.round(num * testData);

        DataSet set = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
        //System.out.println(train);
        //System.out.println(test);
        for (int i = train; i < num; i++) {
            set.addData(this.getInput(i), this.getOutput(i));
        }

        return set;
    }

    public void shuffle() {
        Collections.shuffle(data);
    }

    public String toString() {
        String s = "TrainSet [" + INPUT_SIZE + " ; " + OUTPUT_SIZE + "]\n";
        int index = 0;
        for (double[][] r : data) {
            s += index + ":   " + Arrays.toString(r[0]) + "  >-||-<  " + Arrays.toString(r[1]) + "\n";
            index++;
        }
        return s;
    }

    public int size() {
        return data.size();
    }

    public double[] getInput(int index) {
        if (index >= 0 && index < size()) {
            return data.get(index)[0];
        } else {
            return null;
        }
    }

    public double[] getOutput(int index) {
        if (index >= 0 && index < size()) {
            return data.get(index)[1];
        } else {
            return null;
        }
    }
    
    public void handleMissVal() {
        double[] median0 = new double[getInput(0).length];
        double[] median1 = new double[getInput(0).length];
        
        for (int j = 1; j < getInput(0).length; j++) {
            ArrayList<Double> col0 = new ArrayList<>();
            ArrayList<Double> col1 = new ArrayList<>();
            for (int i = 0; i < data.size(); i++) {
                if (getInput(i)[j] != 0.0) {
                    if (getOutput(i)[0] == 0.0) {
                        col0.add(getInput(i)[j]);
                    } else {
                        col1.add(getInput(i)[j]);
                    }
                }
            }
            median0[j] = computeMedian(col0);
            median1[j] = computeMedian(col1);
        }
        
        //System.out.println("0 :"+Arrays.toString(median0));
        //System.out.println("1 :"+Arrays.toString(median1));
        
        for (int i = 0; i < data.size(); i++) {
            double[] input = new double[INPUT_SIZE];
            double output = getOutput(i)[0];
            for (int j = 0; j < getInput(0).length; j++) {
                input[j] = getInput(i)[j];
                if (j!=0) {
                    if (output == 0.0 && input[j] == 0.0) {
                    input[j] = median0[j];
                    }
                    if (output == 1.0 && input[j] == 0.0) {
                    input[j] = median1[j];
                    }
                }
            }
            data.set(i, new double[][]{input, getOutput(i)});
        }
    }
    
    private double computeMedian(ArrayList col) {
        double median;
        
        Collections.sort(col);
        if(col.size() % 2 != 0) {
            int index = (col.size() + 1)/2;
            median = (double) col.get(index-1);
        }
        else {
            int index = col.size()/2;
            int index2 = col.size()/2 + 1;
            median = 0.5*((double) col.get(index-1) + (double)col.get(index2-1));
        }
        
        return median;
    }
    
    private double max(int col) {
        double max = 0;

        for (int i = 0; i < data.size(); i++) {
            double[] input = getInput(i);
            double[] output = getOutput(i);
            if (col < input.length) {
                if (input[col] > max) {
                    max = input[col];
                }
            } else if (output[col - input.length] > max) {
                max = output[col - input.length];
            }
        }
        //System.out.println(max);
        return max;
    }

    private double min(int col) {
        double min = 99999;

        for (int i = 0; i < data.size(); i++) {
            double[] input = getInput(i);
            double[] output = getOutput(i);
            if (col != input.length) {
                if (min > input[col]) {
                    min = input[col];
                }
            } else if (min > output[col - input.length]) {
                min = output[col - input.length];
            }
        }
        //System.out.println(max);
        return min;
    }

    public void minMaxNorm() {
        copyList();
        String s;
        int newmax = 1;
        int newmin = 0;
        double[] max = new double[INPUT_SIZE + OUTPUT_SIZE];
        double[] min = new double[INPUT_SIZE + OUTPUT_SIZE];
        
        int total_col = data.get(0)[0].length + data.get(0)[1].length;
        
        for (int col = 0; col < total_col; col++) {
            max[col] = max(col);
            min[col] = min(col);
        }

        for (int i = 0; i < data.size(); i++) {
            double[] x = new double[INPUT_SIZE];
            double[] y = new double[OUTPUT_SIZE];
            for (int j = 0; j < total_col; j++) {
                if (j != total_col-1) {
                    x[j] = ((getInput(i)[j] - min[j]) * (newmax - newmin)) / ((max[j] - min[j]) + newmin);
                } else {
                    y[0] = ((getOutput(i)[0] - min[j]) * (newmax - newmin)) / ((max[j] - min[j]) + newmin);
                }
                /*
                if (j == 8) {
                    y[0] = ((getOutput(i)[0] - min[j]) * (newmax - newmin)) / ((max[j] - min[j]) + newmin);
                    
                } else {
                    x[j] = ((getInput(i)[j] - min[j]) * (newmax - newmin)) / ((max[j] - min[j]) + newmin);
                }*/
            }
            data.set(i, new double[][]{x, y});
        }
    }
    
    /*
    public DataSet extractBatch(int size) {
        if (size > 0 && size <= this.size()) {
            DataSet set = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = NetworkTools.randomValues(0, this.size() - 1, size);
            for (Integer i : ids) {
                set.addData(this.getInput(i), this.getOutput(i));
            }
            return set;
        } else {
            return this;
        }
    }

    public DataSet extractData(int size) {
        if (size > 0 && size <= this.size()) {
            DataSet set = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = NetworkTools.randomValues(0, this.size() - 1, size);
            for (Integer i : ids) {
                set.addData(this.getInput(i), this.getOutput(i));
            }
            return set;
        } else {
            return this;
        }
    }
*/
    
    /*
    public DataSet[] kFoldCrossValidation(int k) {
        DataSet[] sets = new DataSet[k];
        
        int jdata = data.size();
        int fold = jdata/k;
        int mfold = jdata%k;
        int x = k-mfold;
        
        int[] num = new int[k];
        for (int i = 0; i < k; i++) {
            if(i<=x-1)
                num[i] = fold;
            
            if(mfold!=0 && i>x-1) {
                num[i] = fold+1;
            }
        }

        int first = 0, last = 0;
        for (int i = 0; i < k; i++) {
            DataSet set = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
            if(i==0) {
                first = 0;
                last = num[i]-1;
            }
            else {
                first = last+1;
                last = last + num[i];
            }
            
            for (int j = first; j <= last; j++) {
                set.addData(this.getInput(j), this.getOutput(j));
            }
            
            sets[i] = set;
        }
        
        return sets;
    }
    
    public DataSet[] getTrainData(DataSet[] sets, int k) {
        DataSet[] trainset = new DataSet[k];
        DataSet train;
        ArrayList <double[][]> all;
        
        for (int j = 0; j < k; j++) {
            all = new ArrayList<>();
            for (int i = 0; i < sets.length; i++) {
                if(i != j)
                    all.addAll(sets[i].data);  
            }
            train = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
            train.data = all;
            trainset[j] = train;
        }
        
        return trainset;
    }
    
    public DataSet[] getTestData(DataSet[] sets, int k) {
        DataSet[] testset = new DataSet[k];
        
        for (int i = 0; i < k; i++) {
            testset[i] = sets[i];
        }
        
        return testset;
    }*/
    
    public static void main(String[] args) {
        DataSet set = new DataSet(8, 1);
        /*
        for(int i = 0; i < 8; i++) {
            double[] a = new double[3];
            double[] b = new double[2];
            for(int k = 0; k < 3; k++) {
                a[k] = (double)((int)(Math.random() * 10)) / (double)10;
                if(k < 2) {
                    b[k] = (double)((int)(Math.random() * 10)) / (double)10;
                }
            }
            set.addData(a,b);
        }*/
         
        //double[][] ddd = set.readCSV();
        //set.splitData(ddd);
        set.readCSV();
        set.handleMissVal();
        set.minMaxNorm();
        //System.out.println(set);
        //set.shuffle();
        //System.out.println(set);

        //System.out.println(set.size());

        DataSet train = set.createTrainData((float) 0.8);
        DataSet test = set.createTestData((float) 0.2);

        System.out.println(train);
        System.out.println(test);
        
        //int k = 5;
        
        //DataSet[] sets = set.kFoldCrossValidation(k);
        /*System.out.println(sets[0]);
        System.out.println(sets[1]);
        System.out.println(sets[2]);
        System.out.println(sets[3]);
        System.out.println(sets[4]);
        
        System.out.println(sets[0].size());
        System.out.println(sets[1].size());
        System.out.println(sets[2].size());
        System.out.println(sets[3].size());
        System.out.println(sets[4].size());*/
        /*DataSet[] trainset = set.getTrainData(sets, k);
        for (int i = 0; i < k; i++) {
            System.out.println("SET "+i);
            System.out.println(trainset[i]);
            System.out.println(sets[i]);
            System.out.println("\n");
        }
        */
    }

}
