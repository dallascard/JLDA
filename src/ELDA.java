import java.util.HashMap;

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
        params.put("-w", "10.0");                      // subsampling
        params.put("-r", "");                      // subsampling


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

        String roles_flag = params.get("-r");
        boolean use_roles = true;
        if (roles_flag.length() > 0)
            use_roles = false;

        double alpha = Double.parseDouble(params.get("-a"));
        double beta = Double.parseDouble(params.get("-b"));
        double gamma = Double.parseDouble(params.get("-g"));
        int n_topics = Integer.parseInt(params.get("-k"));
        int n_personas = Integer.parseInt(params.get("-p"));

        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-u"));
        int subsampling = Integer.parseInt(params.get("-s"));
        double slice_width = Double.parseDouble(params.get("-w"));

        //ELDASampler sampler = new ELDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, vocab_file, docs_file);
        if (use_roles) {
            ERLDASampler sampler = new ERLDASampler(input_dir);
            sampler.run(n_personas, n_topics, alpha, beta, gamma, n_iter, burn_in, subsampling, output_dir, slice_width);
        }
        else {
            ERLDASamplerClusters sampler = new ERLDASamplerClusters(input_dir);
            sampler.run(n_personas, n_topics, alpha, beta, gamma, n_iter, burn_in, subsampling, output_dir, slice_width);
        }

    }
}
