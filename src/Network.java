import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created with IntelliJ IDEA.
 * User: ryan
 * Date: 11/26/12
 * Time: 6:14 PM
 * To change this template use File | Settings | File Templates.
 */
public class Network{

	private double[][] output;
	private double[][] hidden;

	//matrices to save the last deltaW
	private double[][] dHidden;
	private double[][] dOutput;

	private double eta = 0.1;  //Momentum constant
	private double alpha = 0.5; //Learning rate

	public double dErrorStop = .01;

	/**
	 *
	 * @param inputSize
	 * @param hiddenSize
	 * @param outputSize
	 */
	public Network(int inputSize, int hiddenSize, int outputSize){
		hidden = new double[hiddenSize][inputSize];
		init(hidden);

		dHidden = new double[hiddenSize][inputSize];

		output = new double[outputSize][hiddenSize];
		init(output);
		dOutput = new double[outputSize][hiddenSize];
	}

	/**
	 * Trains the neural network, if it takes 500K then it was probably a bad initial weight matrix so I will re-train
	 * @param input
	 * @param expected
	 */
	public double train(List<double[]> input, List<double[]> expected){
		int itr = 0;

		while(true){
			System.out.println(itr++);
			double err = trainEpoch(input,expected);
			System.out.println(err);
			if(err<dErrorStop || itr>500000){
				return err;
			}
		}
	}

	/**
	 * Initialize the matrix to small non-zero values (zero is a saddle point)
	 * @param matrix
	 */
	void init(double[][] matrix){
		for(int i=0;i<matrix.length;i++){
			for(int j=0;j<matrix[i].length;j++){
				matrix[i][j]= (Math.random()-1.0)/20.0;
			}
		}
	}

	/**
	 * Train an epoch
	 * @param input
	 * @param expected
	 * @return error for the entire epoch
	 */
	double trainEpoch(List<double[]> input, List<double[]> expected){
		double errorSq = 0;
		for(int i=0;i<input.size();i++){
			double[][] outputs = feedforward(input.get(i));
			errorSq+=backprop(outputs,expected.get(i));
		}
		return errorSq;
	}

	public double[][] feedforward(double[] input){
		 double[][] outputs = new double[3][];
		 outputs[2] = input; //save input into outputs[0];
		 double[] o1 = MathHelper.sigmoid(MathHelper.multiply(hidden, input));
		 outputs[1] = o1; //save hidden layer result in outputs[1]
		 double[] o2 = MathHelper.sigmoid(MathHelper.multiply(output, o1));
		 outputs[0] = o2; //output layer saved in outputs[0]
		 return outputs;
	}

	/**
	 *
	 * @param outputs 1/2*error squared for the input
	 * @param expectation
	 * @return
	 */
	public double backprop(double[][] outputs, double[] expectation){
		double[] output = outputs[0];
		double[] outputHidden = outputs[1];
		double[] input = outputs[2];

		//Calculate delta on output layer
		double[] deltaOutput = new double[output.length];
		double errorSq=0;
		for(int i=0;i<output.length;i++){
			double err = expectation[i]-output[i];
			errorSq+=err*err;
			deltaOutput[i] = output[i]*(1.0-output[i])*(err);
		}

		//Calculate delta on hidden layer
		double[] deltaHidden = new double[outputHidden.length];
		for(int i=0;i<hidden.length;i++){
			double sum = 0.0;
			for(int j=0;j<output.length;j++){
				sum+=this.output[j][i]*deltaOutput[j];
			}
			deltaHidden[i] = outputHidden[i]*(1.0-outputHidden[i])*sum;
		}


		//Update the output layer
		for(int i=0;i<deltaOutput.length;i++){
			for(int j=0;j<outputHidden.length;j++){
				double dw = eta*deltaOutput[i]*outputHidden[j]; //local derivitive
				double p = dOutput[i][j]*alpha; //Ignore Momentum for now

				this.output[i][j]+=dw+p; //add error +momentum to matrix
				dOutput[i][j]=dw+p; //Save the previous momentum

			}
		}

		//Update the hidden layer
		for(int i=0;i<hidden.length;i++){
			for(int j=0;j<hidden[i].length;j++){
				double dw = eta*deltaHidden[i]*input[j];
				double p = dHidden[i][j]*alpha; //calculate the momentum

				this.hidden[i][j]+=dw+p; //add error +momentum to matrix
				dHidden[i][j] = dw+p;//Save the previous momentum
			}
		}

		return errorSq;
	}




	public static void main(String[] args) throws IOException {

		//5 inputs, 32 hidden (2^5), 1 output ( EVEN>0.5  ODD<=0.5 )
		Network n = new Network(5,32,1);

		//Called when you train the data
		//List<double[]> input = buildInput();
		//List<double[]> output = buildOutput(input);
		//n.train(input,output);
		//n.saveWeights(new File("weights.txt"));

		n.loadWeights(new File("weights.txt")); //Called when you load from a file
		readUserIn(n);
	}

	static void readUserIn(Network n){
		Scanner in = new Scanner(System.in);
		System.out.println("input 5 digit binary number or 'q' to quit");

		outer:
		while(in.hasNext()){

			String line = in.nextLine().replaceAll("[\\s]+","");
			if(line.trim().endsWith("q")){
				System.out.println("exiting");
				return;
			}
			else{
				if(line.length()!=5){
					System.out.println("invalid length "+line.length() + " (must be 5)");
					continue outer;
				}
				double[] userIn = new double[5];
				for(int i=0;i<5;i++){
					char c = line.charAt(i);
					if( c=='0'){
						userIn[i]=0.0;
					}else if(c=='1'){
						userIn[i] = 1.0;
					}else{
						System.out.println("invalid character: "+c);
						continue outer;
					}
				}
				double userOut = n.feedforward(userIn)[0][0];
				System.out.print("Output: "+userOut+" -> ");
				if(userOut>=0.5){
					System.out.println("EVEN");
				}else{
					System.out.println("ODD");
				}
			}
		}

	}


	//Used to verify that everything was classified correctly
	public static void checkAll(Network n, List<double[]> input){
		for(int i=0;i<input.size();i++){
			System.out.println("--------------");
			double[] in = input.get(i);
			System.out.println(Arrays.toString(in));
			double[] out = n.feedforward(in)[0];
			boolean even = even(in);
			if(even && out[0]<=0.5){
				System.out.println("MISCLASSIFIED ERROR EVEN");
			}else if(!even && out[0]>0.5){
				System.out.println("MISCLASSIFIED ERROR ODD");
			}
			System.out.println(Arrays.toString(out));
		}
	}

	public static boolean even(double[] input){
		int countOnes = 0;
		for(int i=0;i<input.length;i++){
			double d = input[i];
			if(d>0.5){
				countOnes++;
			}
		}
		//Even number
		return (countOnes%2==0);
	}

	static List<double[]> buildOutput(List<double[]> input){
		List<double[]> precalced = new ArrayList<double[]>(input.size());
		for(double[] in: input){
			int countOnes = 0;
			for(int i=0;i<in.length;i++){
				double d = in[i];
				if(d>0.5){
					countOnes++;
				}
			}
			//Even number
			if(countOnes%2==0){
				System.out.println(Arrays.toString(in)+"-> even");
				precalced.add(new double[]{0.9});
			}
			else{
				System.out.println(Arrays.toString(in)+"-> odd");
				precalced.add(new double[]{0.1});
			}
		}
		return precalced;
	}

	public void loadWeights(File f) throws FileNotFoundException {
		Scanner s = new Scanner(f);
		for(int i=0;i<hidden.length;i++){
			for(int j=0;j<hidden[0].length;j++){
				hidden[i][j] = s.nextDouble();
			}
		}

		for(int i=0;i<output.length;i++){
			for(int j=0;j<output[0].length;j++){
				output[i][j] = s.nextDouble();
			}
		}
	}

	public void saveWeights(File f) throws IOException {
		FileWriter out = new FileWriter(f);

		for(int i=0;i<hidden.length;i++){
			for(int j=0;j<hidden[0].length;j++){
				out.write(hidden[i][j]+" ");
			}
			out.write("\n");
		}
		out.write("\n");
		for(int i=0;i<output.length;i++){
			for(int j=0;j<output[0].length;j++){
				out.write(output[i][j]+" ");
			}
			out.write("\n");
		}
		out.flush();
		out.close();
	}


	/**
	 * Handwritten input possibilities (there are only 32 of them)
	 * @return
	 */
	public static List<double[]> buildInput(){
		List<double[]> data = new ArrayList<double[]>(32);
		data.add(new double[]{0,0,0,0,0});
		data.add(new double[]{0,0,0,0,1});
		data.add(new double[]{0,0,0,1,0});
		data.add(new double[]{0,0,0,1,1});
		data.add(new double[]{0,0,1,0,0});
		data.add(new double[]{0,0,1,0,1});
		data.add(new double[]{0,0,1,1,0});
		data.add(new double[]{0,0,1,1,1});
		data.add(new double[]{0,1,0,0,0});
		data.add(new double[]{0,1,0,0,1});
		data.add(new double[]{0,1,0,1,0});
		data.add(new double[]{0,1,0,1,1});
		data.add(new double[]{0,1,1,0,0});
		data.add(new double[]{0,1,1,0,1});
		data.add(new double[]{0,1,1,1,0});
		data.add(new double[]{0,1,1,1,1});
		data.add(new double[]{1,0,0,0,0});
		data.add(new double[]{1,0,0,0,1});
		data.add(new double[]{1,0,0,1,0});
		data.add(new double[]{1,0,0,1,1});
		data.add(new double[]{1,0,1,0,0});
		data.add(new double[]{1,0,1,0,1});
		data.add(new double[]{1,0,1,1,0});
		data.add(new double[]{1,0,1,1,1});
		data.add(new double[]{1,1,0,0,0});
		data.add(new double[]{1,1,0,0,1});
		data.add(new double[]{1,1,0,1,0});
		data.add(new double[]{1,1,0,1,1});
		data.add(new double[]{1,1,1,0,0});
		data.add(new double[]{1,1,1,0,1});
		data.add(new double[]{1,1,1,1,0});
		data.add(new double[]{1,1,1,1,1});
		return data;
	}
}
