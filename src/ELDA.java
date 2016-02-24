import org.json.simple.JSONObject;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.io.FileWriter;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;

public class ELDA {

    public static void main(String args[]) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "");  // input dir
        params.put("-p", "20");                     // n_personas
        params.put("-k", "40");                     // n_topics
        params.put("-a", "1");                      // alpha
        params.put("-b", "1");                      // beta
        params.put("-g", "1");                      // gamma
        params.put("-i", "1000");                    // n_iter
        params.put("-u", "200");                     // burn_in
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

        Path tuple_vocab_file = Paths.get(params.get("-d"), "tuple_vocab.json");
        Path tuple_entity_file = Paths.get(params.get("-d"), "tuple_entity.json");
        Path entity_doc_file = Paths.get(params.get("-d"), "entity_doc.json");
        Path vocab_file = Paths.get(params.get("-d"), "vocab.json");
        Path docs_file = Paths.get(params.get("-d"), "docs.json");

        double alpha = Double.parseDouble(params.get("-a"));
        double beta = Double.parseDouble(params.get("-b"));
        double gamma = Double.parseDouble(params.get("-g"));
        int n_topics = Integer.parseInt(params.get("-k"));
        int n_personas = Integer.parseInt(params.get("-p"));

        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-u"));
        int subsampling = Integer.parseInt(params.get("-s"));

        ELDASampler sampler = new ELDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, vocab_file, docs_file);
        int topic_word_matrix[][] = sampler.run(n_personas, n_topics, alpha, beta, gamma, n_iter, burn_in, subsampling);
        String vocab[] = sampler.get_vocab();
        int vocab_size = (int) vocab.length;
        System.out.println(vocab_size);

        /*
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
        */

    }

}
