/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetesnn;

import Jama.*;

/**
 *
 * @author Wind10
 */
public class BPLM extends BP{
    
    private final double lambda_max = 1e25;
    //private double lambda;
    
    private Matrix weights_m; //bobot yang akan diperbarui
    private Matrix errors; //error output terhadap target data
    private Matrix jacobian; //matriks Jacobian
    private Matrix Hessian; //matriks Hessian
    private Matrix Hdiag; //matriks diagonal Hessian
    private Matrix HessianI; //matriks Hessian invers
    private Matrix negativeStep; //delta bobot
    private Matrix g; //gradien (Jtrans*error)
    private Matrix delta_weights; //perubahan bobot   
    private Matrix candidateWeights_m; //kandidat untuk bobot baru   
    
    public BPLM(int... NETWORK_LAYER_SIZES) {
        super(NETWORK_LAYER_SIZES);
        weights_m = new Matrix(((2 + INPUT_SIZE) * this.NETWORK_LAYER_SIZES[NETWORK_SIZE - 2] + 1), 1);
    }

    /**
     *
     * @param target
     */
    @Override
    protected void backpropError(double[] target) {
        for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[this.NETWORK_SIZE - 1]; neuron++) { //setiap neuron pada layer output
            this.error_signal[this.NETWORK_SIZE - 1][neuron] = -1 * this.output_derivative[this.NETWORK_SIZE - 1][neuron];
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
    
    private void computeJacobian(int n) {
        int b = 0; //index bobot

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                jacobian.set(n, b, error_signal[layer][neuron]);
                b = b + 1;
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    jacobian.set(n, b, error_signal[layer][neuron] * output[layer - 1][prevNeuron]);
                    b = b + 1;
                }
            }
        }
    }

    private void getWeights_m(Matrix weights) { //ambil bobot, dari array menuju matriks
        int b = 0;

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                weights.set(b, 0, bias[layer][neuron]);
                b = b + 1;
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    weights.set(b, 0, this.weights[layer][neuron][prevNeuron]);
                    b = b + 1;
                }
            }
        }
    }

    private void setWeights_m(Matrix weights) { //tetapkan bobot, dari matriks menuju array
        int b = 0;

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                bias[layer][neuron] = weights.get(b, 0);
                b = b + 1;
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    this.weights[layer][neuron][prevNeuron] = weights.get(b, 0);
                    b = b + 1;
                }
            }
        }
    }

    private void setErrors(DataSet set, int i) { //set error dari tiap baris data terhadap output jaringan
        double error = set.getOutput(i)[0] - output[NETWORK_SIZE - 1][0];
        errors.set(i, 0, error); //tetapkan error dmatrix ejml
    }

    private void computeHessianandGradient() {
        Hessian = jacobian.transpose().times(jacobian); //H = Jt * J
        for (int i = 0; i < Hessian.getRowDimension(); i++) {
            Hdiag.set(i, 0, Hessian.get(i, i)); //ambil diagonal Hessian
        }
        g = jacobian.transpose().times(errors); //g = Jt* e
    }

    private void computeDiagonalHessian(double learning_rate) {
        for (int j = 0; j < Hessian.getRowDimension(); j++) {
            Hessian.set(j, j, Hdiag.get(j, 0) + learning_rate); //H = Hdiag + learning rate (perubahan pada diagonal saja)
        }

        //HessianI = Hessian.inverse(); //invers Hessian
        //delta_bobot = HessianI.times(g); // deltabobot = InvertedHessian * g
    }

    private double computeMSE() { //hitung MSE dengan vektor error
        double error = errors.normF();
        return error * error / (double) errors.getRowDimension();
    }

    private double computeSSE() { //hitung SSE dengan vektor error
        double error = errors.normF();
        return error * error / 2.0;
    }

    private void configure(int jdata, int jbobot) { //konfigurasi ukuran matriks
        errors = new Matrix(jdata, 1);
        candidateWeights_m = new Matrix(jbobot, 1);
        jacobian = new Matrix(jdata, jbobot);
        Hessian = new Matrix(jbobot, jbobot);
        HessianI = new Matrix(jbobot, jbobot);
        Hdiag = new Matrix(jbobot, 1);
        g = new Matrix(jbobot, 1);
        delta_weights = new Matrix(jbobot, 1);
        negativeStep = new Matrix(jbobot, 1);
        class_train = new int[jdata];
    }
    
    public void trainBPLM(DataSet set, double learning_rate, int maxIterations, double MSEt) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        //int count = 0;
        //double ftol = 1e-12, gtol = 1e-12;
        int jbobot = weights_m.getRowDimension();
        configure(set.size(), jbobot); //konfigurasi ukuran matriks
        //MSEarr = new double[maxIterations];
        double previousMSE, MSE;
        int i = 1;
        previousMSE = MSE = 1000; //MSE inisial        
        //boolean conv = false;
        
        //printWeights();
        while (MSEt <= MSE && i <= maxIterations) {           
            LUDecomposition decomposition;
            //boolean converged;
            //Matriks.decomposition = null;
            getWeights_m(weights_m);
            
            for (int data = 0; data < set.size(); data++) {
                forward(set.getInput(data));
                setErrors(set, data);
                backpropError(set.getOutput(data));
                computeJacobian(data);
            }
            
            previousMSE = computeMSE();
            
            //System.out.print("\nJACOBIAN : ");
            //jacobian.print(jacobian.getColumnDimension(), 10);
            
            //previousMSE = computeSSE();
            
            if(i==1)
                initialMSE = previousMSE;
            
            computeHessianandGradient();
            /*
            boolean a = true;
            if(a) {
                boolean converged = true;
                for (int j = 0; j < g.getRowDimension(); j++) {
                    if( Math.abs(g.get(j, 0) ) > gtol ) {
                        converged = false;
                        break;
                    }
                }
                
                if( converged )
                    break;
            }
            */
            
            boolean done = false;
            boolean nonSingular;
            
            while(!done) {
                
                computeDiagonalHessian(learning_rate);
                
                decomposition = new LUDecomposition(Hessian);
                
                nonSingular = decomposition.isNonsingular();
                
                if(nonSingular) {                  
                    delta_weights = decomposition.solve(g);
                    candidateWeights_m = weights_m.minus(delta_weights);
                    
                    setWeights_m(candidateWeights_m);
                    
                    for (int data = 0; data < set.size(); data++) {
                        forward(set.getInput(data));
                        setErrors(set, data);
                    }
                    
                    MSE = computeMSE();
                    //MSE = computeSSE();
                }
                
                if(!nonSingular || MSE >= previousMSE) {
                    learning_rate *= 10.0;
                    if(learning_rate > lambda_max) {
                        learning_rate = lambda_max;
                        done = true;
                    }
                    //if(MSE==previousMSE) {
                        //count++;
                    //}
                } else {
                    //boolean converged = ftol*previousMSE >= previousMSE-MSE;
                    //conv = converged;
                    
                    learning_rate /= 10.0;
                    done = true;
                    
                    //if( converged )
                    //    break;
                }
            }
            
            //MSEarr[i-1] = MSE;
            System.out.println("Iterasi : " + i + " || PrevMSE : " + previousMSE + " MSE : " + MSE);          
            previousMSE = MSE;
            //printBobot();
            //if(count==3)
              //  break;
            i++;
           //if( conv )
                //break;
            
        }
        //setBobot(bobot_kandidat);
        finalMSE = previousMSE;
        //writeMSE(MSEarr);
        /*
        printOutputNeuron();
        System.out.print("\nBOBOT : ");
        weights_m.print(weights_m.getColumnDimension(), 5);
        System.out.print("\nERRORS : ");
        errors.print(errors.getColumnDimension(), 5);
        System.out.print("\nJACOBIAN : ");
        jacobian.print(jacobian.getColumnDimension(), 10);
        System.out.print("\nHESSIAN : ");
        Hessian.print(Hessian.getColumnDimension(), 10);
        System.out.print("\nGRADIENTS :");
        g.print(g.getColumnDimension(), 5);
        System.out.print("\nDELTA WEIGHTS :");
        delta_weights.print(delta_weights.getColumnDimension(), 5);
        */
        System.out.println("\nTRAIN");
        System.out.println("Lambda : " + learning_rate);
        System.out.println("Inital MSE : " + initialMSE);
        System.out.println("Final MSE : " + finalMSE);
        System.out.println("Iterasi : " + (i-1));
        //test(set, net, false);
        ConfusionMatrix cf = new ConfusionMatrix();
        for (int data = 0; data < set.size(); data++) {
            forward(set.getInput(data));
            class_train[data] = thresholdOutput();
            cf.setConfusionMatrix(class_train[data], set.getOutput(data)[0]);
        }
        cf.computePerformance();
        System.out.println("Akurasi = "+cf.getAccuracy());
        //printBobot();
    }
    /*
    @Override
    public void writeMSE(double[] MSE) {
        try {
            File f = new File("MSEbplm.txt");
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
    }
    */
    /*
    public void hitungHessiandanGradient2(double beta) {
        Hessian = (jacobian.transpose().times(jacobian)).times(beta); //H = Jt * J
        for (int i = 0; i < Hessian.getRowDimension(); i++) {
            Hdiag.set(i, 0, Hessian.get(i, i)); //ambil diagonal Hessian
        }
        g = jacobian.transpose().times(errors); //g = Jt* e
    }
    
    public void hitungPerubahanBobot2(double learning_rate, double alpha) {
        for (int j = 0; j < Hessian.getRowDimension(); j++) {
            Hessian.set(j, j, Hdiag.get(j, 0) + learning_rate + alpha); //H = Hdiag + learning rate (perubahan pada diagonal saja)
        }

        //HessianI = Hessian.inverse(); //invers Hessian
        //delta_bobot = HessianI.times(g); // deltabobot = InvertedHessian * g
    }
    
    
    
    
    public double hitungSSW(Matrix bobot) {
        //double sum = 0;
        
        //for(int i=0; i<bobot.getRowDimension(); i++) {
        //    sum += bobot.get(i, 0) * bobot.get(i, 0);
        //}
        
        //return sum;
        
        double w = bobot.normF();
        return w * w / 2.0;
    }
    
    public void trainLM(DataSet set, double learning_rate, int maxIterations, double SSEt, boolean UseBR) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        
        double beta = 1, alpha = 0, gamma = 0, SSE=1000, SSW = 0;
        
        int jbobot = weights_m.getRowDimension();
        
        configure(set.size(), jbobot);
        ConfusionMatrix cf = new ConfusionMatrix();
        int i = 1;
        double previousSSE;
        while (SSEt <= SSE && i <= maxIterations) {
            double trace;
            LUDecomposition decomposition = null;
            
            getWeights_m(weights_m);
            hitungSSW(weights_m);
            for (int data = 0; data < set.size(); data++) {
                forward(set.getInput(data));
                setErrors(set, data);
                backpropError(set.getOutput(data));
                computeJacobian(data);
            }
            
            SSE = computeSSE();
            previousSSE = SSE;
            if (i == 1) {
                initialMSE = SSE;
            }
            
            hitungHessiandanGradient2(beta);
            //double objective = 0, current = 0;
            double objective = beta * SSE + alpha * SSW;
            double current = objective + 1.0;
            
            learning_rate /= 10;
            
            while (current >= objective && learning_rate < lambda_max) {
                
                learning_rate *= 10;
                
                hitungPerubahanBobot2(learning_rate, alpha);

                decomposition = new LUDecomposition(Hessian);
                
                if(!decomposition.isNonsingular())
                    continue;
                
                delta_bobot = decomposition.solve(g);
                candidateWeights_m = weights_m.minus(delta_bobot);
                setWeights_m(candidateWeights_m);
                hitungSSW(candidateWeights_m);
                
                for (int data = 0; data < set.size(); data++) {
                    forward(set.getInput(data));
                    setErrors(set, data);
                }
                
                SSE = computeSSE();
                current =  beta * SSE + alpha * SSW;
            }
            
            learning_rate /= 10;
            
            if(UseBR) {
                // Compute the trace for the inverse hessian
                
                trace = Hessian.inverse().trace();
                
                // Poland update's formula:
                gamma = jbobot - (alpha * trace);
                alpha = jbobot / (2.0 * SSW + trace);
                beta = Math.abs((set.size() - gamma) / (2.0 * SSE));
                //beta = (set.size() - gamma) / (2.0 * SSE);

                // Original MacKay's update formula:
                //  gamma = jbobot - (alpha * trace);
                //  alpha = gamma / (2.0 * SSW);
                //  beta = (gamma - set.size()) / (2.0 * SSE);
            }
            previousSSE = SSE;
            System.out.println("Iterasi : " + i + " || PrevMSE : " + previousSSE + " MSE : " + SSE);          
            objective = current;
            
            i++;
        }
        finalMSE = SSE;
        System.out.println("Lambda : " + learning_rate);
        System.out.println("Inital MSE : " + initialMSE);
        System.out.println("Final MSE : " + finalMSE);
        System.out.println("Iterasi : " + (i-1));
        
        for (int data = 0; data < set.size(); data++) {
            forward(set.getInput(data));
            class_train[data] = thresholdOutput();
            cf.setConfusionMatrix(class_train[data], set.getOutput(data)[0]);
        }
        cf.computePerformance();
        System.out.println("Akurasi = "+cf.getAccuracy());
        
        //net.savePerformance2();
    }
    
    public void trainBPLMBR(DataSet set, double learning_rate, int maxIterations, double MSEt) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
         
        int jbobot = weights_m.getRowDimension();
        ConfusionMatrix cf = new ConfusionMatrix();
        configure(set.size(), jbobot); //konfigurasi ukuran matriks
                
        double previousMSE, MSE;
        int i = 1;
        double beta = 1, alpha = 0, gamma = 0, SSE=1000, SSW = 0;
        //double objective = 0, current = 0;
        previousMSE = MSE = 1000; //MSE inisial        
         
        while (MSEt <= MSE && i <= maxIterations) {
            double trace;
            LUDecomposition decomposition;
            //Matriks.decomposition = null;
            getWeights_m(weights_m);
            hitungSSW(weights_m);
            for (int data = 0; data < set.size(); data++) {
                forward(set.getInput(data));
                backpropError(set.getOutput(data));
                computeJacobian(data);
                setErrors(set, data);
            }
            
            //previousMSE = hitungMSE();
            previousMSE = computeSSE();
            
            if(i==1)
                initialMSE = previousMSE;
            
            hitungHessiandanGradient2(beta);
            
            double objective = beta * SSE + alpha * SSW;
            double current = objective + 1.0;
            
            boolean done = false;
            boolean nonSingular;
            
            while(!done) {
                
                hitungPerubahanBobot2(learning_rate, alpha);
                
                decomposition = new LUDecomposition(Hessian);
                
                nonSingular = decomposition.isNonsingular();
                
                if(nonSingular) {                  
                    delta_bobot = decomposition.solve(g);
                    candidateWeights_m = weights_m.minus(delta_bobot);
                    
                    setWeights_m(candidateWeights_m);
                    hitungSSW(candidateWeights_m);
                    for (int data = 0; data < set.size(); data++) {
                        forward(set.getInput(data));
                        setErrors(set, data);
                    }
                    
                    //MSE = hitungMSE();
                    MSE = computeSSE();
                }
                
                if(!nonSingular || current >= objective) {
                    learning_rate *= 10.0;
                    
                    current =  beta * SSE + alpha * SSW;
                    
                    if(learning_rate > lambda_max) {
                        learning_rate = lambda_max;
                        done = true;
                    }
                } else {
                    learning_rate /= 10.0;
                    done = true;
                }
                
            }
            
            trace = Hessian.inverse().trace();

            // Poland update's formula:
            gamma = jbobot - (alpha * trace);
            alpha = jbobot / (2.0 * SSW + trace);
            beta = Math.abs((set.size() - gamma) / (2.0 * SSE));
            
            System.out.println("Iterasi : " + i + " || PrevMSE : " + previousMSE + " MSE : " + MSE);          
            previousMSE = MSE;
            
            i++;
        }
        //setBobot(bobot_kandidat);
        finalMSE = previousMSE;
               
        //System.out.print("BOBOT : ");
        //weights_m.print(weights_m.getColumnDimension(), 5);
        //System.out.println("");
        //System.out.print("ERRORS : ");
        //errors.print(errors.getColumnDimension(), 5);
        //System.out.println("");
        //System.out.print("JACOBIAN : ");
        //jacobian.print(jacobian.getColumnDimension(), 5);
        //System.out.println("");
        //System.out.print("HESSIAN : ");
        //Hessian.print(Hessian.getColumnDimension(), 5);
        //System.out.println("");
        
        //printOutput(set);
        
        System.out.println("Lambda : " + learning_rate);
        System.out.println("Inital MSE : " + initialMSE);
        System.out.println("Final MSE : " + finalMSE);
        System.out.println("Iterasi : " + (i-1));
        
        for (int data = 0; data < set.size(); data++) {
            forward(set.getInput(data));
            class_train[data] = thresholdOutput();
            cf.setConfusionMatrix(class_train[data], set.getOutput(data)[0]);
        }
        cf.computePerformance();
        System.out.println("Akurasi = "+cf.getAccuracy());
        //printBobot();
    }*/
}
