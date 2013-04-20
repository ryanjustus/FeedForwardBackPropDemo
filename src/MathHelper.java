/**
 * Created with IntelliJ IDEA.
 * User: ryan
 * Date: 12/3/12
 * Time: 10:17 AM
 * To change this template use File | Settings | File Templates.
 */
public class MathHelper {

	public static double[] multiply(double[][] M, double[] x){
		double[] b = new double[M.length];
		for(int i=0;i<M.length;i++){
			double v =0;
			for(int j=0;j<M[0].length;j++){
				v+=M[i][j]*x[j];
			}
			b[i]=v;
		}
		return b;
	}

	public static double[] sigmoid(double[] x){
		for(int i=0;i<x.length;i++){
			x[i] = 1.0/(1.0+Math.exp(-1.0*x[i]));
		}
		return x;
	}

	public static double sigmoid(double x){
		return 1.0/(1.0+Math.exp(-1.0*x));
	}

	public static double error(double[] output, double[] expectation){
		double sumsq = 0;
		for(int i=0;i<output.length;i++){
			sumsq+=Math.pow(expectation[i]-output[i],2.0);
		}
		return 0.5*sumsq;
	}

	public static void print(double[][] matrix){
		for(int i=0;i<matrix.length;i++){
			for(int j=0;j<matrix[0].length;j++){
				System.out.print(matrix[i][j]+"  ");
			}
			System.out.println();
		}
	}
}
