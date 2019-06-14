package clustering;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;


public class SeqFileGenerateMapper extends Mapper<Text, Text, Text, Text> {

    Text fileContent;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        fileContent = new Text();
    }

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {

        String content = new String(Files.readAllBytes(Paths.get(value.toString())));

        fileContent.set(content);
        context.write(key,fileContent);
    }
}
