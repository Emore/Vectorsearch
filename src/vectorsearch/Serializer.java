package vectorsearch;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

public class Serializer {
	public void serialize(Object object, String fileName) {
		try {
			OutputStream file = new FileOutputStream(fileName);
			OutputStream buffer = new BufferedOutputStream(file);
			ObjectOutput output = new ObjectOutputStream( buffer );
			try {
				output.writeObject(object);
				System.out.println("Serialized " + fileName + ".");
			}
			finally {
				output.close();
			}
		} catch (IOException e) {
			System.err.println("Could not serialize: " + e.getMessage());
		}
	}
	
	public Object deserialize(String name) throws IOException {
		InputStream file = new FileInputStream(name);
		InputStream buffer = new BufferedInputStream(file);
		ObjectInput input = new ObjectInputStream (buffer);
		try {
			try {
				Object recovered = input.readObject();
				System.out.println("Deserialized " + name + ".");
				return recovered;
			} catch (ClassNotFoundException e) {
				throw new IOException(e.getMessage());
			}
		} finally {
			input.close();
		}
	}
}
