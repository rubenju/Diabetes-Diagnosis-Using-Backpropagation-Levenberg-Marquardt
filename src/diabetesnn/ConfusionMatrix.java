/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetesnn;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Wind10
 */
public class ConfusionMatrix {
    private int TP, TN, FP, FN;
    private float accuracy, sensitivity, specificity, precision, fmeasure;
    
    public float getAccuracy() {
        return accuracy;
    }

    public float getSensitivity() {
        return sensitivity;
    }

    public float getSpecificity() {
        return specificity;
    }

    public float getPrecision() {
        return precision;
    }

    public float getFmeasure() {
        return fmeasure;
    }
    
    public void setConfusionMatrix(int output, double target) {
        if(output==1 && target==1.0) {
            TP+=1;
        } else if(output==1 && target==0.0) {
            FP+=1;
        } else if(output==0 && target==1.0) {
            FN+=1;
        } else if(output==0 && target==0.0) {
            TN+=1;
        }      
    }
    
    public void computePerformance() {
        accuracy = (((float) (TP+TN)/(TP+TN+FN+FP)) * 100);
        sensitivity = (((float) TP/(TP+FN)) * 100);
        specificity = ((float) TN/(TN+FP)) * 100;
        precision = ((float) TP/(TP+FP)) * 100;
        fmeasure = (float) (2*((sensitivity*precision)/(sensitivity+precision)));
    }
    
    
    public void savePerformance(boolean isBPLM, int i) {
        try {
            File f = new File("ppperformance"+i+".txt");
            
            if(!f.exists()){
                System.out.println("Make new file");
                f.createNewFile();
            }
            
            FileWriter fw = new FileWriter(f, true);
            if(isBPLM){               
                fw.write("BPLM\n");
            }    
            else
                fw.write("BP\n");
            fw.write(Float.toString(accuracy)+"\n"+Float.toString(sensitivity)+"\n"+Float.toString(specificity)+"\n"+Float.toString(precision)+"\n"+Float.toString(fmeasure)+"\n\n");
            fw.close();
        } catch (IOException ex) {
            Logger.getLogger(ConfusionMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }       
    }
    
    public void savePerformance2(boolean isBPLM, int i) {
        try {
            File f = new File("tttrainperformance"+i+".txt");
            
            if(!f.exists()){
                System.out.println("Make new file");
                f.createNewFile();
            }
            
            FileWriter fw = new FileWriter(f, true);
            if(isBPLM){               
                fw.write("BPLM\n");
            }    
            else
                fw.write("BP\n");
            fw.write(Float.toString(accuracy)+"\n"+Float.toString(sensitivity)+"\n"+Float.toString(specificity)+"\n"+Float.toString(precision)+"\n"+Float.toString(fmeasure)+"\n\n");
            fw.close();
        } catch (IOException ex) {
            Logger.getLogger(ConfusionMatrix.class.getName()).log(Level.SEVERE, null, ex);
        }       
    }
    
    /*
    public static double[] createArray(int size, double init_value){
        if(size < 1){
            return null;
        }
        double[] ar = new double[size];
        for(int i = 0; i < size; i++){
            ar[i] = init_value;
        }
        return ar;
    }
    */
    
    /*
    public static void main(String[] args) {
        double[] arr = new double[10];
        for (int i = 0; i < 10; i++) {
            arr[i] = Network.randomValue(-0.5, 0.5);
        }
        System.out.println(Arrays.toString(arr));
        for (int i = 0; i < 10; i++) {
            arr[i] = Network.randomValue(-0.5, 0.5);
        }
        System.out.println(Arrays.toString(arr));
    }
    */
    
    /*
    public void saveWeight(String file, BP4 bp, BPLM4 bplm, boolean isBPLM) throws Exception{       
        File f = new File(file);
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(f));
        if(isBPLM)
            out.writeObject(bplm);
        else
            out.writeObject(bp);
        out.flush();
        out.close();
    }
    
    public static BP4 loadWeightBP(String file, BP4 bp) throws Exception{
        File f = new File(file);
        ObjectInputStream out = new ObjectInputStream(new FileInputStream(f));
        BP4 net = (BP4) out.readObject();
        out.close();
        return bp;
    }
    
    public static BPLM4 loadWeightBPLM(String file, BPLM4 bplm) throws Exception{
        File f = new File(file);
        ObjectInputStream out = new ObjectInputStream(new FileInputStream(f));
        BPLM4 net = (BPLM4) out.readObject();
        out.close();
        return bplm;
    }
    */
}
