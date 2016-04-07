import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.io.FileWriter;
import java.util.List;

import org.json.simple.JSONObject;

public class ELDA {

    public static void main(String args[]) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "");  // input dir
        params.put("-o", "");  // output dir
        params.put("-p", "25");                     // n_personas
        params.put("-k", "25");                     // n_topics
        params.put("-a", "1");                      // alpha
        params.put("-b", "1");                      // beta
        params.put("-g", "1");                      // gamma
        params.put("-i", "3000");                    // n_iter
        params.put("-u", "1000");                     // burn_in
        params.put("-s", "25");                      // subsampling


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

        if (params.get("-o").equals("")) {
            System.out.println(params);
            System.exit(0);
        }

        System.out.println(params);

        String input_dir = params.get("-d");
        String output_dir = params.get("-o");

        double alpha = Double.parseDouble(params.get("-a"));
        double beta = Double.parseDouble(params.get("-b"));
        double gamma = Double.parseDouble(params.get("-g"));
        int n_topics = Integer.parseInt(params.get("-k"));
        int n_personas = Integer.parseInt(params.get("-p"));

        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-u"));
        int subsampling = Integer.parseInt(params.get("-s"));

        //ELDASampler sampler = new ELDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, vocab_file, docs_file);
        ERLDASampler sampler = new ERLDASampler(input_dir);
        sampler.run(n_personas, n_topics, alpha, beta, gamma, n_iter, burn_in, subsampling, output_dir);

    }
}
