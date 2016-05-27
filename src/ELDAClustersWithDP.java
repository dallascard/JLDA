import java.util.HashMap;

public class ELDAClustersWithDP {

    public static void main(String args[]) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "");  // input dir
        params.put("-o", "");  // output dir
        params.put("-p", "50");                     // n_personas
        params.put("-k", "100");                     // n_topics
        params.put("-a", "0.1");                      // alpha
        params.put("-b", "0.1");                      // beta
        params.put("-g", "0.1");                      // gamma
        params.put("-l", "0.5");                    // lambda
        params.put("-i", "11000");                    // n_iter
        params.put("-u", "10000");                     // burn_in
        params.put("-s", "10");                      // subsampling
        params.put("-w", "10.0");                      // subsampling
        params.put("-t", "5000");                // max story types

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
        double lambda = Double.parseDouble(params.get("-l"));

        int n_topics = Integer.parseInt(params.get("-k"));
        int n_personas = Integer.parseInt(params.get("-p"));

        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-u"));
        int subsampling = Integer.parseInt(params.get("-s"));
        double slice_width = Double.parseDouble(params.get("-w"));

        int max_story_types = Integer.parseInt(params.get("-t"));

        //ELDASampler sampler = new ELDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, vocab_file, docs_file);
        ERLDASamplerClustersWithDP sampler = new ERLDASamplerClustersWithDP(input_dir);
        sampler.run(n_personas, n_topics, alpha, beta, gamma, lambda, n_iter, burn_in, subsampling, output_dir, slice_width, max_story_types);

    }
}
