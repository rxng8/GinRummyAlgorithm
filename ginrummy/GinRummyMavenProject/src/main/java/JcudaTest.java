
import jcuda.*;
import jcuda.runtime.*;
public class JcudaTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
	}

}
