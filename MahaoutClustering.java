package clustering;


import java.io.IOException;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileAsBinaryOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

public class MahoutClustering {


    /*public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

    }*/

    public static void main(String[] args) throws Exception {

        Path input = new Path("E:\\knovos\\MahoutClustering\\inputFile\\fileInfo.txt");
        Path output = new Path("E:\\knovos\\MahoutClustering\\output");
        Path seqFiles = new Path(output, "seqFiles");
        Path tokenizedDocumentsPath = new Path(output, "tokenizedDocumentsPath");
        Path termFrequencyVectorsPath = new Path(output, "tf-vectors");
        Path tfidfPath = new Path(output, "tfidfPath");
        Path centroidsOutputPath = new Path(output, "centroids");
        Path kMeansClustered = new Path(output, "kMeansClustered");


        Configuration conf = new Configuration();
        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", "=");
        FileSystem hdfs = FileSystem.get(conf);
        hdfs.delete(output, true);
        Job job = Job.getInstance(conf, "seqFile Generate");
        job.setJarByClass(MahoutClustering.class);
        job.setMapperClass(SeqFileGenerateMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, seqFiles);
        if (job.waitForCompletion(true)) {

            DocumentProcessor.tokenizeDocuments(seqFiles, StandardAnalyzer.class, tokenizedDocumentsPath, conf);

            DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocumentsPath, output,
                    DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
                    conf, 1, 1, 0.0f, PartialVectorMerger.NO_NORMALIZING,
                    true, 1, 100, false, false);

            Pair<Long[], List<Path>> documentFrequencies = TFIDFConverter.calculateDF(termFrequencyVectorsPath, tfidfPath,
                    conf, 100);

            TFIDFConverter.processTfIdf(termFrequencyVectorsPath, tfidfPath,
                    conf, documentFrequencies, 1, 100,
                    PartialVectorMerger.NO_NORMALIZING, false, false, false, 1);

            Path centroids = RandomSeedGenerator.buildRandom(conf,new Path(tfidfPath,"tfidf-vectors"),centroidsOutputPath,2,new CosineDistanceMeasure());

            KMeansDriver.run(conf,new Path(tfidfPath,"tfidf-vectors"),centroids,kMeansClustered,0.5,30,true,0.01,false);

        }
    }
}
