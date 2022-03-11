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
public class Matriks {
    
    static public LUDecomposition decomposition;
    
    public static boolean declareDecomposition(double[][] Hessian) {
        Matrix H = new Matrix(Hessian);
        Matriks.decomposition = new LUDecomposition(H);
        
        return decomposition.isNonsingular();
    }  
    
    public static double[][] computeDecomposition(double[][] gradient) {
        Matrix g = new Matrix(gradient);
        
        return decomposition.solve(g).getArray();
    }   
    
    public static double[][] plus(double[][] M, double[][] N) {
        int m1 = M.length;
        int n1 = N.length;
        int m2 = M[0].length;
        int n2 = N[0].length;
        
        if(m1 != n1 && m2 != n2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        
        double[][] result = new double[m1][n2];
        
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                result[i][j] = M[i][j] + N[i][j];
            }
        }
        
        return result;
    }
    
    public static double[][] minus(double[][] M, double[][] N) {
        int m1 = M.length;
        int n1 = N.length;
        int m2 = M[0].length;
        int n2 = N[0].length;
        
        if(m1 != n1 && m2 != n2) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        
        double[][] result = new double[m1][n2];
        
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                result[i][j] = M[i][j] - N[i][j];
            }
        }
        
        return result;
    }
    
    public static double[][] multiply(double[][] M, double[][] N) {
        int m1 = M.length;
        int n1 = N.length;
        int m2 = M[0].length;
        int n2 = N[0].length;
        
        double[][] result = new double[m1][n2];
        double temp;
        
        if (m2 != n1) {
            throw new RuntimeException("Illegal matrix dimensions.");
        }
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                temp = 0;
                for (int k = 0; k < n1; k++) {
                    temp = temp + M[i][k] * N[k][j];
                }
                result[i][j] = temp;
            }
        }
        
        return result;
    }
    
    public static double[][] inverse(double[][] M) {
        Matrix X = new Matrix(M);  
        return X.inverse().getArray();
    }
    
    public static double[][] transpose(double[][] M) {
        double[][] result = new double[M[0].length][M.length];
        
        for(int i = 0 ; i< M.length ; i++) {
            for (int j = 0; j < M[0].length; j++) {
                result[j][i] = M[i][j];
            }
        }
        
        return result;
    }
    
}
