import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.io.FileWriter;
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

        Path tuple_vocab_file = Paths.get(params.get("-d"), "tuple_vocab.json");
        Path tuple_entity_file = Paths.get(params.get("-d"), "tuple_entity.json");
        Path tuple_role_file = Paths.get(params.get("-d"), "tuple_role.json");
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

        //ELDASampler sampler = new ELDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, vocab_file, docs_file);
        ERLDASampler sampler = new ERLDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, tuple_role_file, vocab_file, docs_file);
        int persona_word_matrix[][] = sampler.run(n_personas, n_topics, alpha, beta, gamma, n_iter, burn_in, subsampling);
        String vocab[] = sampler.get_vocab();
        int vocab_size = (int) vocab.length;
        System.out.println(vocab_size);


        System.out.println("Writing results to file");
        for (int p=0; p < n_personas; p++) {
            Path output_file = Paths.get(params.get("-o"), p + ".json");
            JSONObject obj = new JSONObject();

            for (int v=0; v < vocab_size; v++)
                obj.put(new String(vocab[v].getBytes("UTF-8"), "UTF-8"), persona_word_matrix[p][v]);

            try (FileWriter file = new FileWriter(output_file.toString())) {
                file.write(obj.toJSONString());
            }

        }
    }

}
