import org.json.simple.JSONObject;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.io.FileWriter;
import java.util.HashMap;

public class LDA {

    public static void main(String args[]) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "");  // input dir
        params.put("-k", "50");                     // n_topics
        params.put("-a", "1");                      // alpha
        params.put("-b", "1");                      // beta
        params.put("-i", "5000");                    // n_iter
        params.put("-u", "1000");                     // fiexburn_in
        params.put("-s", "10");                      // subsampling


        String arg = null;
        for (String s: args) {
            if (arg == null)
                arg = s;
            else {
                params.put(arg, s);
                arg = null;
            }
        }

        if (params.get("-d").equals("")) {
            System.out.println(params);
            System.exit(0);
        }

        System.out.println(params);

        Path word_num_file = Paths.get(params.get("-d"), "word_num.json");
        Path word_doc_file = Paths.get(params.get("-d"), "word_doc.json");
        Path vocab_file = Paths.get(params.get("-d"), "vocab.json");

        float alpha = Float.parseFloat(params.get("-a"));
        float beta = Float.parseFloat(params.get("-b"));
        int n_topics = Integer.parseInt(params.get("-k"));

        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-u"));
        int subsampling = Integer.parseInt(params.get("-s"));

        LDASampler LDASampler = new LDASampler(word_num_file, word_doc_file, vocab_file);
        int word_topic_matrix[][] = LDASampler.run(n_topics, alpha, beta, n_iter, burn_in, subsampling);
        int vocab_size = (int) word_topic_matrix.length;
        System.out.println(vocab_size);

        String vocab[] = LDASampler.get_vocab();

        System.out.println("Writing results to file");
        //String output_dir = "/Users/dcard/Projects/CMU/ARK/guac/datasets/mfc_v2/lda/";
        for (int k=0; k < n_topics; k++) {
            Path output_file = Paths.get(params.get("-d"), k + ".json");
            JSONObject obj = new JSONObject();

            for (int v=0; v < vocab_size; v++)
                obj.put(new String(vocab[v].getBytes("UTF-8"), "UTF-8"), word_topic_matrix[v][k]);

            try (FileWriter file = new FileWriter(output_file.toString())) {
                file.write(obj.toJSONString());
            }

        }

    }

}
