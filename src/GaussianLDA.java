
/*
The Gaussian LDA part of this was draws heavily on the Gaussian LDA implementation at: https://github.com/rajarshd/Gaussian_LDA.git
which accompanies the paper "Gaussian LDA for Topic Models with Word Embeddings"
Citation:
@InProceedings{das-zaheer-dyer:2015,
  author    = {Das, Rajarshi  and  Zaheer, Manzil  and  Dyer, Chris},
  title     = {Gaussian LDA for Topic Models with Word Embeddings},
  booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  publisher = {Association for Computational Linguistics},
  url       = {http://www.aclweb.org/anthology/P15-1077}
}
*/

import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.io.FileWriter;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.io.BufferedWriter;

import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.factory.LinearSolverFactory;

import org.apache.commons.math3.special.Gamma;


public class GaussianLDA {

    public static void main(String args[]) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "");  // input dir
        params.put("-o", "");  // output dir
        params.put("-p", "25");                     // n_personas
        params.put("-k", "25");                     // n_topics
        params.put("-a", "1");                      // alpha
        params.put("-b", "1");                      // beta
        params.put("-i", "3000");                    // n_iter
        params.put("-u", "1000");                     // burn_in
        params.put("-s", "25");                      // subsampling
        params.put("-w", "10.0");                      // subsampling
        params.put("-v", "10");                      // word vector dimension


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
        int n_topics = Integer.parseInt(params.get("-k"));
        int n_personas = Integer.parseInt(params.get("-p"));

        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-u"));
        int subsampling = Integer.parseInt(params.get("-s"));
        double slice_width = Double.parseDouble(params.get("-w"));

        int vec_size = Integer.parseInt(params.get("-v"));

        //ELDASampler sampler = new ELDASampler(entity_doc_file, tuple_vocab_file, tuple_entity_file, vocab_file, docs_file);
        GaussianLDASampler sampler = new GaussianLDASampler(input_dir, vec_size);
        sampler.initialize(n_personas, n_topics);
        sampler.run(alpha, beta, n_iter, burn_in, subsampling, output_dir, slice_width);

    }
}


class GaussianLDASampler {
    private int vocab_size;
    private int n_tuples;
    private int n_docs;     // N
    private int n_entities;
    private int n_roles;
    private int n_personas;
    private int n_topics;   // K

    private int head_word_vocab_size;
    private int n_head_words;
    //private int head_phrase_vocab_size;
    //private int n_head_phrases;

    private int entity_doc[];
    private int tuple_vocab[];
    private int tuple_entity[];
    private int tuple_role[];
    private HashMap<Integer, List<Integer>> entity_tuples;
    private String vocab[];
    private String docs[];

    // ** Gaussian LDA parts **
    private static NormalInverseWishart prior;
    // embedding associated with each word of the vocab
    private static DenseMatrix64F[] data_vectors;
    // mean vector for each topic
    private static ArrayList<DenseMatrix64F> topic_means = new ArrayList<DenseMatrix64F>(); // AKA tableMeans
    // inverse covariance matrices for each topic
    private static ArrayList<DenseMatrix64F> topic_inverse_covariances = new ArrayList<DenseMatrix64F>();
    // determinant of covariance matrix for each topic
    private static ArrayList<Double> determinants = new ArrayList<Double>();
    // stores the sum of the vectors of customers at a given table
    private static ArrayList<DenseMatrix64F> sum_topic_vectors = new ArrayList<DenseMatrix64F>();
    // stores the squared sum of the vectors of customers at a given table
    private static ArrayList<DenseMatrix64F> sum_squared_topic_vectors = new ArrayList<DenseMatrix64F>();

    // caching variable for calculating topic parameters
    private static DenseMatrix64F k0mu0mu0T = null;

    // linear solver
    private LinearSolver<DenseMatrix64F> newSolver;


    private int head_vocab_list[];
    private int head_entity_list[];
    private String head_word_vocab[];
    private HashMap<Integer, List<Integer>> entity_head_words;

    //private int head_phrase_vocab_list[];
    //private int head_phrase_entity_list[];
    //private String head_phrase_vocab[];
    //private HashMap<Integer, List<Integer>> entity_head_phrases;

    private int entity_personas[];
    private int tuple_topics[];

    private int document_persona_counts[][];
    private int document_persona_totals[];
    private int topic_vocab_counts[][];
    private int persona_role_topic_counts[][][];
    private int persona_role_counts[][];
    private int topic_tuple_counts[];
    private int persona_role_vocab_counts[][][];
    private int persona_head_word_counts[][];
    private int t_document_persona_counts[][];
    private int t_persona_role_topic_counts[][][];
    private int t_topic_vocab_counts[][];
    private int t_persona_role_counts[][];
    private int t_topic_tuple_counts[];
    private int t_persona_role_vocab_counts[][][];
    private int t_entity_persona_counts[][];
    private int t_persona_head_word_counts[][];
    //private int t_persona_head_phrase_counts[][];

    private static BufferedWriter runLogger = null;
    private static BufferedWriter perplexities = null;

    // have the constructor read in the data
    public GaussianLDASampler(String input_dir, int dx) throws Exception {

        Data.D = dx;

        Path tuple_vocab_file = Paths.get(input_dir, "tuple_vocab.json");
        Path tuple_entity_file = Paths.get(input_dir, "tuple_entity.json");
        Path tuple_role_file = Paths.get(input_dir, "tuple_role.json");
        Path entity_doc_file = Paths.get(input_dir, "entity_doc.json");
        Path vocab_file = Paths.get(input_dir, "vocab.json");
        Path docs_file = Paths.get(input_dir, "docs.json");
        Path vectors_file = Paths.get(input_dir, "tuple_vectors.json");

        Path head_vocab_file = Paths.get(input_dir, "head_word_vocab_list.json");
        Path head_entity_file = Paths.get(input_dir, "head_word_entity_list.json");
        Path head_word_vocab_file = Paths.get(input_dir, "head_word_vocab.json");

        //Path head_phrase_vocab_file = Paths.get(input_dir, "head_phrase_vocab_list.json");
        //Path head_phrase_entity_file = Paths.get(input_dir, "head_phrase_entity_list.json");
        //Path head_phrase_full_vocab_file = Paths.get(input_dir, "head_phrase_vocab.json");

        JSONParser parser = new JSONParser();
        JSONArray entity_doc_json = (JSONArray) parser.parse(new FileReader(entity_doc_file.toString()));
        JSONArray tuple_vocab_json = (JSONArray) parser.parse(new FileReader(tuple_vocab_file.toString()));
        JSONArray tuple_entity_json = (JSONArray) parser.parse(new FileReader(tuple_entity_file.toString()));
        JSONArray tuple_role_json = (JSONArray) parser.parse(new FileReader(tuple_role_file.toString()));
        JSONArray vocab_json = (JSONArray) parser.parse(new FileReader(vocab_file.toString()));
        JSONArray docs_json = (JSONArray) parser.parse(new FileReader(docs_file.toString()));
        JSONArray head_vocab_json = (JSONArray) parser.parse(new FileReader(head_vocab_file.toString()));
        JSONArray head_entity_json = (JSONArray) parser.parse(new FileReader(head_entity_file.toString()));
        JSONArray head_word_vocab_json = (JSONArray) parser.parse(new FileReader(head_word_vocab_file.toString()));
        JSONArray tuple_vectors_json = (JSONArray) parser.parse(new FileReader(vectors_file.toString()));

        //JSONArray head_phrase_vocab_json = (JSONArray) parser.parse(new FileReader(head_phrase_vocab_file.toString()));
        //JSONArray head_phrase_entity_json = (JSONArray) parser.parse(new FileReader(head_phrase_entity_file.toString()));
        //JSONArray head_phrase_full_vocab_json = (JSONArray) parser.parse(new FileReader(head_phrase_full_vocab_file.toString()));


        n_tuples = tuple_vocab_json.size();
        System.out.println("n_tuples=" + n_tuples);
        n_entities = entity_doc_json.size();
        System.out.println("n_entities=" + n_entities);
        n_head_words = head_vocab_json.size();
        System.out.println("n_head_words=" + n_head_words);
        assert n_tuples == tuple_vectors_json.size();
        System.out.println("vector size=" + Data.D);

        // transfer entity to document mapping from json to array, and count the number of documents
        n_docs = 0;
        entity_doc = new int[n_entities];
        for (int i = 0; i < n_entities; i++) {
            entity_doc[i] = ((Long) entity_doc_json.get(i)).intValue();
            if (entity_doc[i] > n_docs) {
                n_docs = entity_doc[i];
            }
        }
        n_docs += 1;  // one larger than largest index

        // transfer the tuple-vocab and tuple-entity mappings to arrays and count the vocabulary size
        vocab_size = 0;
        n_roles = 0;
        tuple_vocab = new int[n_tuples];
        tuple_entity = new int[n_tuples];
        tuple_role = new int[n_tuples];
        DenseMatrix64F vector_matrix = new DenseMatrix64F(n_tuples, Data.D);  // initialize the vector matrix
        // also record all the tuples associated with each entity
        entity_tuples = new HashMap<>();

        for (int i = 0; i < n_tuples; i++) {
            tuple_vocab[i] = ((Long) tuple_vocab_json.get(i)).intValue();
            tuple_entity[i] = ((Long) tuple_entity_json.get(i)).intValue();
            tuple_role[i] = ((Long) tuple_role_json.get(i)).intValue();

            JSONArray vector = (JSONArray) tuple_vectors_json.get(i);
            for (int m = 0; m < Data.D; m++) {
                vector_matrix.set(i, m, (double) vector.get(m));
            }
            if (tuple_vocab[i] > vocab_size) {
                vocab_size = tuple_vocab[i];
            }
            if (tuple_role[i] > n_roles) {
                n_roles = tuple_role[i];
            }
            // if we haven't seen this entity before, make a new list for it
            if (entity_tuples.get(tuple_entity[i]) == null) {
                List<Integer> tuples = new ArrayList<>();
                tuples.add(i);
                entity_tuples.put(tuple_entity[i], tuples);
            }
            // otherwise, add this tuple to the appropriate list
            else {
                // probably a better way to do this...
                List<Integer> tuples = entity_tuples.get(tuple_entity[i]);
                tuples.add(i);
                entity_tuples.put(tuple_entity[i], tuples);
            }
        }

        data_vectors = new DenseMatrix64F[n_tuples]; //splitting into vectors
        CommonOps.rowsToVector(vector_matrix, data_vectors);
        System.out.println(data_vectors[0]);

        vocab_size += 1;  // one larger than largest index
        n_roles += 1;

        vocab = new String[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            vocab[i] = (String) vocab_json.get(i);
        }

        docs = new String[n_docs];
        for (int i = 0; i < n_docs; i++) {
            docs[i] = (String) docs_json.get(i);
        }

        head_vocab_list = new int[n_head_words];
        head_entity_list = new int[n_head_words];
        entity_head_words = new HashMap<>();

        head_word_vocab_size = 0;
        for (int i = 0; i < n_head_words; i++) {
            head_vocab_list[i] = ((Long) head_vocab_json.get(i)).intValue();
            head_entity_list[i] = ((Long) head_entity_json.get(i)).intValue();
            if (entity_head_words.get(head_entity_list[i]) == null) {
                List<Integer> head_words = new ArrayList<>();
                head_words.add(i);
                entity_head_words.put(head_entity_list[i], head_words);
            }
            else {
                List<Integer> head_words = entity_head_words.get(head_entity_list[i]);
                head_words.add(i);
                entity_head_words.put(head_entity_list[i], head_words);
            }
        }
        head_word_vocab_size = head_word_vocab_json.size();

        head_word_vocab = new String[head_word_vocab_size];
        for (int i = 0; i < head_word_vocab_size; i++) {
            head_word_vocab[i] = (String) head_word_vocab_json.get(i);
        }

    }


    void initialize(int n_personas, int n_topics) {

        this.n_personas = n_personas;
        this.n_topics = n_topics;

        // initialize arrays
        System.out.println("Initializing arrays");
        entity_personas = new int[n_entities];
        tuple_topics = new int[n_tuples];
        document_persona_counts = new int[n_docs][n_personas];
        document_persona_totals = new int[n_docs];
        persona_role_topic_counts = new int[n_personas][n_roles][n_topics];
        topic_vocab_counts = new int[n_topics][vocab_size];
        persona_role_counts = new int[n_personas][n_roles];
        topic_tuple_counts = new int[n_topics];
        persona_role_vocab_counts = new int[n_personas][n_roles][vocab_size];

        persona_head_word_counts = new int[n_personas][head_word_vocab_size];
        //int [][] persona_head_phrase_counts = new int[n_personas][head_phrase_vocab_size];

        t_document_persona_counts = new int[n_docs][n_personas];
        t_persona_role_topic_counts = new int[n_personas][n_roles][n_topics];
        t_topic_vocab_counts = new int[n_topics][vocab_size];
        t_persona_role_counts = new int[n_personas][n_roles];
        t_topic_tuple_counts = new int[n_topics];
        t_persona_role_vocab_counts = new int[n_personas][n_roles][vocab_size];
        t_entity_persona_counts = new int[n_entities][n_personas];
        t_persona_head_word_counts = new int[n_personas][head_word_vocab_size];
        //t_persona_head_phrase_counts= new int[n_personas][head_phrase_vocab_size];

        // ** Gaussian LDA part of setup **
        prior = new NormalInverseWishart();

        prior.mu_0 = Util.getSampleMean(data_vectors);
        prior.nu_0 = Data.D; //initializing to the dimension
        prior.sigma_0 = CommonOps.identity(Data.D); //setting as the identity matrix
        CommonOps.scale(3 * Data.D, prior.sigma_0);
        prior.k_0 = 0.1;

        if (prior.nu_0 < (double) Data.D) {
            System.out.println("The initial degrees of freedom of the prior is less than the dimension!. Setting it to the number of dimension: " + Data.D);
            prior.nu_0 = Data.D;
        }

        double degOfFreedom = prior.nu_0 - Data.D + 1;
        //Now calculate the covariance matrix of the multivariate T-distribution
        double coeff = (double) (prior.k_0 + 1) / (prior.k_0 * (degOfFreedom));
        DenseMatrix64F sigma_T = new DenseMatrix64F(Data.D, Data.D);
        CommonOps.scale(coeff, prior.sigma_0, sigma_T);

        newSolver = LinearSolverFactory.general(Data.D, Data.D);
        LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.symmPosDef(Data.D);
        if (!solver.setA(sigma_T))
            throw new RuntimeException("Invert failed");
        DenseMatrix64F sigma_TInv = new DenseMatrix64F(Data.D, Data.D);
        solver.invert(sigma_TInv);

        double sigmaTDet = CommonOps.det(sigma_T);

        // initialize all of our compuational shortcuts to zeros
        for (int k = 0; k < n_topics; k++) {
            DenseMatrix64F zero = new DenseMatrix64F(Data.D, 1);
            sum_topic_vectors.add(zero);
            zero = new DenseMatrix64F(Data.D, Data.D);
            sum_squared_topic_vectors.add(zero);
            zero = new DenseMatrix64F(Data.D, Data.D);
            topic_inverse_covariances.add(zero);
            determinants.add(0.0);
            zero = new DenseMatrix64F(Data.D, 1);
            topic_means.add(zero);
        }


        // randomly assign entities to personas
        for (int e = 0; e < n_entities; e++) {
            int d_e = entity_doc[e];
            int p = ThreadLocalRandom.current().nextInt(0, n_personas);
            entity_personas[e] = p;
            document_persona_counts[d_e][p] += 1;
            document_persona_totals[d_e] += 1;
        }

        System.out.println(n_entities + " entities");

        // randomly assign tuples to topics
        for (int j = 0; j < n_tuples; j++) {
            int v_j = tuple_vocab[j];
            int e_j = tuple_entity[j];
            int r_j = tuple_role[j];
            int p_j = entity_personas[e_j];
            int k = ThreadLocalRandom.current().nextInt(0, n_topics);

            tuple_topics[j] = k;
            persona_role_topic_counts[p_j][r_j][k] += 1;
            topic_vocab_counts[k][v_j] += 1;
            persona_role_counts[p_j][r_j] += 1;
            topic_tuple_counts[k] += 1;
            persona_role_vocab_counts[p_j][r_j][v_j] += 1;

            // update sum and squard sum trackers
            DenseMatrix64F sum = sum_topic_vectors.get(k);
            CommonOps.add(data_vectors[j], sum, sum);
            DenseMatrix64F sum_squared = sum_squared_topic_vectors.get(k);
            DenseMatrix64F tuple_vector_transpose = new DenseMatrix64F(1, Data.D);
            DenseMatrix64F square_vector = new DenseMatrix64F(Data.D, Data.D);
            tuple_vector_transpose = CommonOps.transpose(data_vectors[j], tuple_vector_transpose);
            //Multiply x_ix_i^T and add it to the sum_squared for this topic
            CommonOps.mult(data_vectors[j], tuple_vector_transpose, square_vector);
            CommonOps.add(sum_squared, square_vector, sum_squared);
        }

        System.out.println(n_tuples + " tuples");

        // store the head words for later interpretability, even though we're not using them here
        for (int j = 0; j < n_head_words; j++) {
            int e_j = head_entity_list[j];
            int v_j = head_vocab_list[j];
            int p_j = entity_personas[e_j];
            persona_head_word_counts[p_j][v_j] += 1;
        }

        /*
        for (int j=0; j < n_head_phrases; j++) {
            int e_j = head_phrase_entity_list[j];
            int v_j = head_phrase_vocab_list[j];
            int p_j = entity_personas[e_j];
            persona_head_phrase_counts[p_j][v_j] += 1;
        }
        */

        // compute initial topic parameters for each topic
        for (int k = 0; k < n_topics; k++) {
            calculate_topic_params(k);
        }

        //double check again
        for (int k = 0; k < n_topics; k++) {
            if (topic_tuple_counts[k] == 0) {
                System.out.println("Still some tables are empty....exiting!");
                System.exit(1);
            }
        }

        System.out.println("number of documents=" + n_docs);
        System.out.println("number of tuples=" + n_tuples);
        System.out.println("number of roles=" + n_roles);
        System.out.println("vocab size=" + vocab_size);

        System.out.println("Initialization complete");

    }

    public int[][][] run(double alpha, double beta, int n_iter, int burn_in, int subsampling, String outputDir, double slice_width) throws Exception {

        // Determine random orders in which to visit the entities and tuples
        List<Integer> entity_order = new ArrayList<>();
        for (int i = 0; i < n_entities; i++) {
            entity_order.add(i);
        }

        List<Integer> tuple_order = new ArrayList<>();
        for (int i = 0; i < n_tuples; i++) {
            tuple_order.add(i);
        }

        // start sampling
        System.out.println("Doing burn-in");
        for (int i=0; i < n_iter; i++) {

            // slice sample hyperparameters
            if ((i > 0) & (i % 20 == 0)) {
                if ((i < 500) | (i % 100 == 0)) {
                    alpha = slice_sample_alpha(alpha, slice_width);
                    beta = slice_sample_beta(beta, slice_width);
                    System.out.println("alpha=" + alpha + "; beta=" + beta);

                }
            }

            // first sample personas
            for (int q=0; q < n_entities; q++) {
                double pr[] = new double[n_personas];
                int e = entity_order.get(q);
                int d_e = entity_doc[e];
                int p_e = entity_personas[e];

                document_persona_counts[d_e][p_e] -= 1;
                for (int p=0; p < n_personas; p++) {
                    pr[p] = Math.log(document_persona_counts[d_e][p] + alpha);
                }
                List<Integer> tuples = entity_tuples.get(e);
                assert tuples != null;
                for (int t : tuples) {
                    int topic_t = tuple_topics[t];
                    int role_t = tuple_role[t];

                    persona_role_topic_counts[p_e][role_t][topic_t] -= 1;
                    persona_role_counts[p_e][role_t] -= 1;
                }
                for (int t : tuples) {
                    int topic_t = tuple_topics[t];
                    int role_t = tuple_role[t];

                    for (int p = 0; p < n_personas; p++) {
                        pr[p] += Math.log(persona_role_topic_counts[p][role_t][topic_t] + beta) - Math.log(persona_role_counts[p][role_t] + beta * n_topics);
                    }
                }
                for (int t : tuples) {
                    int topic_t = tuple_topics[t];
                    int role_t = tuple_role[t];
                    // add the subtracted counts back in so that they don't affect the next tuple
                    persona_role_topic_counts[p_e][role_t][topic_t] += 1;
                    persona_role_counts[p_e][role_t] += 1;
                }

                double p_sum = 0;
                for (int p=0; p < n_personas; p++) {
                    pr[p] = Math.exp(pr[p]);
                    p_sum += pr[p];
                }

                for (int p=0; p < n_personas; p++) {
                    pr[p] = pr[p]/p_sum;
                }

                double f = ThreadLocalRandom.current().nextDouble();
                int p = 0;
                double temp = pr[p];
                while (f > temp) {
                    p += 1;
                    temp += pr[p];
                }

                entity_personas[e] = p;
                document_persona_counts[d_e][p] += 1;
                for (int t : tuples) {
                    int topic_t = tuple_topics[t];
                    int role_t = tuple_role[t];
                    // transfer the persona topic counts to the new persona
                    persona_role_topic_counts[p_e][role_t][topic_t] -= 1;
                    persona_role_counts[p_e][role_t] -= 1;
                    persona_role_topic_counts[p][role_t][topic_t] += 1;
                    persona_role_counts[p][role_t] += 1;

                    // update counts of words assoicated with each persona
                    int v_t = tuple_vocab[t];
                    persona_role_vocab_counts[p_e][role_t][v_t] -= 1;
                    persona_role_vocab_counts[p][role_t][v_t] += 1;
                }
                List<Integer> head_words = entity_head_words.get(e);
                for (int t : head_words) {
                    int v_t = head_vocab_list[t];
                    persona_head_word_counts[p_e][v_t] -= 1;
                    persona_head_word_counts[p][v_t] += 1;
                }
                /*
                List<Integer> head_phrases = entity_head_phrases.get(e);
                for (int t : head_phrases) {
                    int v_t = head_phrase_vocab_list[t];
                    persona_head_phrase_counts[p_e][v_t] -= 1;
                    persona_head_phrase_counts[p][v_t] += 1;
                }
                */
            }

            // then sample topics
            for (int q=0; q < n_tuples; q++) {
                double pr[] = new double[n_topics];
                int j = tuple_order.get(q);
                int e_j = tuple_entity[j];
                int z_j = tuple_topics[j];
                int r_j = tuple_role[j];
                int v_j = tuple_vocab[j];
                int p_j = entity_personas[e_j];

                // remove the counts for this word
                persona_role_topic_counts[p_j][r_j][z_j] -= 1;
                topic_vocab_counts[z_j][v_j] -= 1;
                topic_tuple_counts[z_j] -= 1;

                // subtract the vector for this tuple from the corresponding topic sum and sum squared counters
                DenseMatrix64F topic_vector_transpose = new DenseMatrix64F(1, Data.D);
                topic_vector_transpose = CommonOps.transpose(data_vectors[j], topic_vector_transpose);
                DenseMatrix64F squared_vector = new DenseMatrix64F(Data.D, Data.D);
                CommonOps.mult(data_vectors[j], topic_vector_transpose, squared_vector);

                DenseMatrix64F old_topic_sum = sum_topic_vectors.get(z_j);
                DenseMatrix64F old_topic_sum_squared = sum_squared_topic_vectors.get(z_j);
                DenseMatrix64F new_topic_sum = new DenseMatrix64F(Data.D, 1);
                CommonOps.sub(old_topic_sum, data_vectors[j], new_topic_sum); //subtracting the vector of this customer.
                sum_topic_vectors.set(z_j, new_topic_sum);

                DenseMatrix64F new_topic_sum_squared = new DenseMatrix64F(Data.D, Data.D);
                CommonOps.sub(old_topic_sum_squared, squared_vector, new_topic_sum_squared);
                sum_squared_topic_vectors.set(z_j, new_topic_sum_squared);

                // now recalculate topic parameters
                calculate_topic_params(z_j);

                ArrayList<Double> posterior = new ArrayList<>();
                Double max = Double.NEGATIVE_INFINITY;
                //go over each topic
                for(int k = 0; k < n_topics; k++)
                {
                    double count = persona_role_topic_counts[p_j][r_j][k] + beta;
                    double logLikelihood = log_multivariate_t_density(data_vectors[z_j], k);
                    //add log prior in the posterior vector
                    double logPosterior = Math.log(count) + logLikelihood;
                    posterior.add(logPosterior);
                    if(logPosterior > max)
                        max = logPosterior;
                }
                //to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                //the log p_max cancels put and prevents overflow in the exponentiating phase.
                for(int k = 0 ; k < n_topics; k++)
                {
                    double p = posterior.get(k);
                    p = p - max;
                    double expP = Math.exp(p);
                    posterior.set(k, expP);
                }

                //now sample a topic from this posterior vector. The sample method will normalize the vector
                //so no need to normalize now.
                int k = Util.sample(posterior);

                /*
                // OLD METHOD pre-Guassian
                // compute probabilities
                double p_sum = 0;
                for (int k = 0; k < n_topics; k++) {
                    pr[k] = (persona_role_topic_counts[p_j][r_j][k] + beta) * (topic_vocab_counts[k][v_j] + gamma) / (topic_tuple_counts[k] + gamma * vocab_size);
                    assert pr[k] > 0;
                    p_sum += pr[k];
                }

                // sample a topic
                double f = ThreadLocalRandom.current().nextDouble() * p_sum;
                int k = 0;
                double temp = pr[k];
                while (f > temp) {
                    k += 1;
                    temp += pr[k];
                }
                */

                // update assignments and counts
                tuple_topics[j] = k;
                persona_role_topic_counts[p_j][r_j][k] += 1;
                topic_vocab_counts[k][v_j] += 1;
                topic_tuple_counts[k] += 1;

                // update sum and sum squared trackers
                DenseMatrix64F sum = sum_topic_vectors.get(k);
                CommonOps.add(data_vectors[j], sum, sum);

                DenseMatrix64F sum_squared = sum_squared_topic_vectors.get(k);
                CommonOps.add(sum_squared, squared_vector, sum_squared);

                calculate_topic_params(k); //update the table params.

            }

            // keep running sums of the selected samples
            if (i > burn_in) {
                if (i % subsampling == 0) {
                    System.out.print("-");
                    for (int p = 0; p < n_personas; p++) {
                        for (int r = 0; r < n_roles; r++) {
                            t_persona_role_counts[p][r] += persona_role_counts[p][r];
                            for (int k = 0; k < n_topics; k++) {
                                t_persona_role_topic_counts[p][r][k] += persona_role_topic_counts[p][r][k];
                            }
                            for (int v = 0; v < vocab_size; v++)
                                t_persona_role_vocab_counts[p][r][v] += persona_role_vocab_counts[p][r][v];

                        }
                        for (int d = 0; d < n_docs; d++) {
                            t_document_persona_counts[d][p] += document_persona_counts[d][p];
                        }
                        for (int v = 0; v < head_word_vocab_size; v++) {
                            t_persona_head_word_counts[p][v] += persona_head_word_counts[p][v];
                        }
                        //for (int v = 0; v < head_phrase_vocab_size; v++) {
                        //    t_persona_head_phrase_counts[p][v] += persona_head_phrase_counts[p][v];
                        //}
                    }
                    for (int k = 0; k < n_topics; k++) {
                        t_topic_tuple_counts[k] += topic_tuple_counts[k];
                        for (int v = 0; v < vocab_size; v++)
                            t_topic_vocab_counts[k][v] += topic_vocab_counts[k][v];
                    }
                    for (int e = 0; e < n_entities; e++) {
                        t_entity_persona_counts[e][entity_personas[e]] += 1;
                    }
                }
            }
            else if (i % subsampling == 0) {
                System.out.print(".");
            }

        }

        // return final word-topic matrices

        System.out.println("PERSONA-VOCAB COUNTS");
        for (int p=0; p < n_personas; p++) {
            System.out.println("**" + p + "**");
            List<Integer> list = new ArrayList<>();
            for (int v = 0; v < vocab_size; v++) {
                for (int r = 0; r < n_roles; r++)
                    list.add(t_persona_role_vocab_counts[p][r][v]);
            }

            Collections.sort(list);
            Collections.reverse(list);
            int n_to_print = 10;
            int threshold = list.get(n_to_print);
            int n_printed = 0;
            for (int v = 0; v < vocab_size; v++) {
                for (int r = 0; r < n_roles; r++) {
                    if (t_persona_role_vocab_counts[p][r][v] >= threshold) {
                        System.out.println(r + ":" + vocab[v] + ": " + t_persona_role_vocab_counts[p][r][v]);
                        n_printed += 1;
                        if (n_printed >= n_to_print) {
                            r = n_roles;
                            v = vocab_size;
                        }
                    }
                }
            }
            System.out.println("");
        }

        /*
        for (int p=0; p < n_personas; p++) {
            System.out.println("** persona " + p + "**");
            List<Integer> list = new ArrayList<>();
            for (int k = 0; k < n_personas; k++)
                list.add(t_persona_topic_counts[p][k]);
            Collections.sort(list);
            Collections.reverse(list);
            int n_to_print = 2;
            int threshold = list.get(n_to_print);
            for (int k = 0; k < n_personas; k++) {
                if (t_persona_topic_counts[p][k] >= threshold)
                    System.out.println(k + ": " + t_persona_topic_counts[p][k]);
            }
            System.out.println("");
        }
        */

        Path output_file = Paths.get(outputDir, "topic_vocab_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int v=0; v < vocab_size; v++) {
                file.write(vocab[v] + ",");
                for (int k=0; k < n_topics; k++) {
                    file.write(t_topic_vocab_counts[k][v] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "persona_role_topic_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int k=0; k < n_topics; k++) {
                file.write(k + ",");
                for (int p=0; p < n_personas; p++) {
                    for (int r=0; r < n_roles; r++) {
                        file.write(t_persona_role_topic_counts[p][r][k] + ",");
                    }
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "entity_persona_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int e=0; e < n_entities; e++) {
                file.write(e + ",");
                for (int p=0; p < n_personas; p++) {
                    file.write(t_entity_persona_counts[e][p] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "persona_role_vocab_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int r=0; r < n_roles; r++) {
                for (int v=0; v < vocab_size; v++) {
                    file.write(r + ":" + vocab[v] + ',');
                    for (int p=0; p < n_personas; p++) {
                        file.write(t_persona_role_vocab_counts[p][r][v] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        output_file = Paths.get(outputDir, "persona_head_word_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int v=0; v < head_word_vocab_size; v++) {
                file.write(head_word_vocab[v] + ',');
                for (int p=0; p < n_personas; p++) {
                    file.write(persona_head_word_counts[p][v] + ",");
                }
                file.write("\n");
            }
        }

        /*
        output_file = Paths.get(outputDir, "persona_head_phrase_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int v=0; v < head_phrase_vocab_size; v++) {
                file.write(head_phrase_vocab[v] + ',');
                for (int p=0; p < n_personas; p++) {
                    file.write(persona_head_phrase_counts[p][v] + ",");
                }
                file.write("\n");
            }
        }
        */

        return t_persona_role_vocab_counts;
    }


    private double calc_log_p_alpha(double alpha) {
        double log_p = Math.log(alpha)  ;
        for (int d=0; d < n_docs; d++) {
            for (int k=0; k < n_personas; k++) {
                log_p += Gamma.logGamma(alpha + document_persona_counts[d][k]) - Gamma.logGamma(alpha);
            }
            log_p -= Gamma.logGamma(n_personas * alpha + document_persona_totals[d]) + Gamma.logGamma(n_personas * alpha);
        }
        return log_p;
    }

    private double slice_sample_alpha(double alpha, double slice_width) {
        //System.out.println("Tuning alpha");
        int alpha_count = 0;
        double log_p_current_alpha = calc_log_p_alpha(alpha);
        double log_alpha = Math.log(alpha);
        double u = Math.log(ThreadLocalRandom.current().nextDouble()) + log_p_current_alpha;
        //System.out.println("current log p = " + log_p_current_alpha);
        //System.out.println("Target log p = " + u);
        double offset = ThreadLocalRandom.current().nextDouble();
        double left = log_alpha - offset * slice_width;
        double right = left + slice_width;
        double new_log_alpha = left + ThreadLocalRandom.current().nextDouble() * (right - left);
        double log_p_new_alpha = calc_log_p_alpha(Math.exp(new_log_alpha));
        //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new alpha = " + Math.exp(new_log_alpha) + "; log p = " + log_p_new_alpha);
        while (log_p_new_alpha < u) {
            if (new_log_alpha < log_alpha) {
                left = new_log_alpha;
            } else {
                right = new_log_alpha;
            }
            new_log_alpha = left + ThreadLocalRandom.current().nextDouble() * (right - left);
            log_p_new_alpha = calc_log_p_alpha(Math.exp(new_log_alpha));
            //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new alpha = " + Math.exp(new_log_alpha) + "; log p = " + log_p_new_alpha);
        }
        alpha = Math.exp(new_log_alpha);
        //System.out.println("new alpha = " + alpha);
        return alpha;
    }

    private double calc_log_p_beta(double beta) {
        double log_p = Math.log(beta)  ;
        for (int p=0; p < 1; p++) {
            for (int r=0; r < 1; r++) {
                for (int k = 0; k < n_topics; k++) {
                    log_p += Gamma.logGamma(beta + persona_role_topic_counts[p][r][k]) - Gamma.logGamma(beta);
                }
                log_p -= Gamma.logGamma(n_topics * beta + persona_role_counts[p][r]) + Gamma.logGamma(n_topics * beta);
            }
        }
        return log_p;
    }

    private double slice_sample_beta(double beta, double slice_width) {
        //System.out.println("Tuning beta");
        int beta_count = 0;
        double log_p_current_beta = calc_log_p_beta(beta);
        double log_beta = Math.log(beta);
        double u = Math.log(ThreadLocalRandom.current().nextDouble()) + log_p_current_beta;
        //System.out.println("current log p = " + log_p_current_beta);
        //System.out.println("Target log p = " + u);
        double offset = ThreadLocalRandom.current().nextDouble();
        double left = log_beta - offset * slice_width;
        double right = left + slice_width;
        double new_log_beta = left + ThreadLocalRandom.current().nextDouble() * (right - left);
        double log_p_new_beta = calc_log_p_beta(Math.exp(new_log_beta));
        //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new beta = " + Math.exp(new_log_beta) + "; log p = " + log_p_new_beta);
        while (log_p_new_beta < u) {
            if (new_log_beta < log_beta) {
                left = new_log_beta;
            } else {
                right = new_log_beta;
            }
            new_log_beta = left + ThreadLocalRandom.current().nextDouble() * (right - left);
            log_p_new_beta = calc_log_p_beta(Math.exp(new_log_beta));
            //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new beta = " + Math.exp(new_log_beta) + "; log p = " + log_p_new_beta);
        }
        beta = Math.exp(new_log_beta);
        //System.out.println("new beta = " + beta);
        return beta;
    }


    /**
     * This method calculates the topic params (bayesian mean, covariance^-1, determinant etc.)
     * All it needs is the table_id and the tableCounts, tableMembers and sumTableCustomers should be updated correctly before calling this.
     */
    private void calculate_topic_params(int k)
    {
        int count = topic_tuple_counts[k];
        double nu_n = prior.nu_0 + count;
        double k_n = prior.k_0 + count;

        //calculate mu_n
        DenseMatrix64F mu_n = new DenseMatrix64F(Data.D, 1);
        CommonOps.scale(prior.k_0, prior.mu_0, mu_n); //k_0 * mu_o
        //Now add N X_bar
        CommonOps.add(mu_n, sum_topic_vectors.get(k), mu_n);
        CommonOps.divide(k_n, mu_n, mu_n); // divide by k_n
        if(topic_means.size() > k)
            topic_means.set(k, mu_n);
        else //for new table
        {
            assert k <= topic_means.size();
            topic_means.add(mu_n);
        }
        //Re-calculate Sigma_N
        //Sigma_N = Sigma_0 + C + k_0 * N /(k_0 + N) (\bar(X) - \mu_0) (\bar(X) - \mu_0)^T
        // C = \Sigma_{i=1}^{N} (X_i - \bar(X)) (X_i - \bar(X))^T, Since there is one customer at the table, C is zero
//			DenseMatrix64F xBar = new DenseMatrix64F(Data.D,1);
//			CommonOps.divide(count, sumTableCustomers.get(tableId), xBar);
//			//Now calculate C
//			DenseMatrix64F C = new DenseMatrix64F(Data.D, Data.D);
//			Set<Integer> members = tableMembers.get(tableId);
//			for(int member:members)
//			{
//				DenseMatrix64F xi_xBar = new  DenseMatrix64F(Data.D,1);
//				CommonOps.sub(dataVectors[member], xBar, xi_xBar);
//				DenseMatrix64F xi_xBar_T = new  DenseMatrix64F(1,Data.D);
//				xi_xBar_T = CommonOps.transpose(xi_xBar,xi_xBar_T);
//				DenseMatrix64F mul = new DenseMatrix64F(Data.D, Data.D);
//				CommonOps.mult(xi_xBar, xi_xBar_T, mul);
//				//System.out.println(mul.get(1, 2));
//				//System.out.println("i");
//				CommonOps.add(mul, C, C);
//			}
//			//Now calculate k_0 * N /(k_0 + N) (\bar(X) - \mu_0) (\bar(X) - \mu_0)^T
//			DenseMatrix64F xBar_x0 = new  DenseMatrix64F(Data.D,1);
//			DenseMatrix64F xBar_x0_T = new  DenseMatrix64F(1,Data.D);
//			CommonOps.sub(xBar, prior.mu_0, xBar_x0); //calculating (\bar(X) - \mu_0)
//			//Now compute the transpose
//			xBar_x0_T = CommonOps.transpose(xBar_x0, xBar_x0_T);
//			DenseMatrix64F xBar_x0_xBar_x0_T = new DenseMatrix64F(Data.D,Data.D);
//			CommonOps.mult(xBar_x0, xBar_x0_T, xBar_x0_xBar_x0_T);
//			//calculate the scaling term k_0 * N /(k_0 + N)
//			double scale = (prior.k_0 * count)/(double)(k_n);
//			//calculate k_0 * N /(k_0 + N) (\bar(X) - \mu_0) (\bar(X) - \mu_0)^T
//			CommonOps.scale(scale, xBar_x0_xBar_x0_T, xBar_x0_xBar_x0_T);
//			//Now add Sigma_0, then C, then xBar_x0_xBar_x0_T
//			DenseMatrix64F sigmaN = new DenseMatrix64F(Data.D,Data.D);
//			CommonOps.add(prior.sigma_0,C, sigmaN);
//			CommonOps.add(sigmaN,xBar_x0_xBar_x0_T, sigmaN);
//			//System.out.println(sigmaN);
//			//Now will scale the covariance matrix by (k_N + 1)/(k_N (v_N - D + 1)) (This is because the t-distribution take the matrix scaled as input)
//			double scaleTdistrn = (k_n + 1)/(k_n * (nu_n - Data.D + 1));
//			CommonOps.scale(scaleTdistrn, sigmaN, sigmaN);

        //we will be using the new update
        //Sigma_N = Sigma_0 + \sum(y_iy_i^T) - (k_n)\mu_N\mu_N^T + k_0\mu_0\mu_0^T
        //calculate \mu_N\mu_N^T
        DenseMatrix64F mu_n_T = new DenseMatrix64F(1, Data.D);
        mu_n_T = CommonOps.transpose(mu_n, mu_n_T);
        DenseMatrix64F mu_n_mu_nT = new DenseMatrix64F(Data.D, Data.D);
        CommonOps.mult(mu_n, mu_n_T, mu_n_mu_nT);
        CommonOps.scale(k_n, mu_n_mu_nT);

        //cache k_0\mu_0\mu_0^T, only compute it once
        if(k0mu0mu0T == null)
        {
            //compute mu0^T
            DenseMatrix64F mu0T = new DenseMatrix64F(1, Data.D);
            mu0T = CommonOps.transpose(prior.mu_0, mu0T);
            k0mu0mu0T = new DenseMatrix64F(Data.D, Data.D);
            CommonOps.mult(prior.mu_0, mu0T, k0mu0mu0T);
            CommonOps.scale(prior.k_0, k0mu0mu0T);
        }
        DenseMatrix64F sigmaN = new DenseMatrix64F(Data.D, Data.D);
        CommonOps.add(prior.sigma_0, sum_squared_topic_vectors.get(k), sigmaN);
        CommonOps.subEquals(sigmaN, mu_n_mu_nT);
        CommonOps.add(sigmaN, k0mu0mu0T, sigmaN);
        double scaleTdistrn = (k_n + 1)/(k_n * (nu_n - Data.D + 1));
        CommonOps.scale(scaleTdistrn, sigmaN, sigmaN);

        //calculate det(Sigma)
        double det = CommonOps.det(sigmaN);
        //System.out.println(det+" "+count);
        //System.out.println(det);
        if(k < determinants.size())
            determinants.set(k, det);
        else
            determinants.add(det);
        //Now calculate Sigma^(-1) and det(Sigma) and store them
        //calculate Sigma^(-1)
        if( !newSolver.setA(sigmaN) )
            throw new RuntimeException("Invert failed");

        DenseMatrix64F sigmaNInv = new DenseMatrix64F(Data.D, Data.D);
        newSolver.invert(sigmaNInv);
        if(k < topic_inverse_covariances.size())
            topic_inverse_covariances.set(k, sigmaNInv);//storing the inverse covariances
        else
            topic_inverse_covariances.add(sigmaNInv);
    }


    /**
     * @param x data point
     * @param k topic
     * @return
     */
    //private static double logMultivariateTDensity(DenseMatrix64F x, DenseMatrix64F mu, DenseMatrix64F sigmaInv,double det, double nu)
    private double log_multivariate_t_density(DenseMatrix64F x, int k)
    {
        DenseMatrix64F mu = topic_means.get(k);
        DenseMatrix64F sigmaInv = topic_inverse_covariances.get(k);
        double det = determinants.get(k);
        int count = topic_tuple_counts[k]; //this is the prior
        //Now calculate the likelihood
        //calculate degrees of freedom of the T-distribution
        double nu = prior.nu_0 + count - Data.D + 1;
        //calculate (x = mu)
        DenseMatrix64F x_minus_mu = new DenseMatrix64F(Data.D, 1);
        CommonOps.sub(x, mu, x_minus_mu);
        //take the transpose
        DenseMatrix64F x_minus_mu_transpose = new DenseMatrix64F(1, Data.D);
        x_minus_mu_transpose  = CommonOps.transpose(x_minus_mu, x_minus_mu_transpose );
        //Calculate (x = mu)^TSigma^(-1)(x = mu)
        DenseMatrix64F prod = new DenseMatrix64F(1, Data.D);
        CommonOps.mult(x_minus_mu_transpose, sigmaInv, prod);
        DenseMatrix64F prod1 = new DenseMatrix64F(1, 1);
        CommonOps.mult(prod, x_minus_mu, prod1);
        //Finally get the value in a double.
        assert prod1.numCols == 1;
        assert prod1.numRows == 1;
        double val = prod1.get(0, 0); //prod1 is a 1x1 matrix
        //System.out.println("val = "+val);
        //System.out.println("det = "+det);
        double logprob = 0.0;
        logprob = Gamma.logGamma((nu + Data.D)/2) - (Gamma.logGamma(nu/2) + Data.D/2 * (Math.log(nu)+Math.log(Math.PI)) + 0.5 * Math.log(det) + (nu + Data.D)/2* Math.log(1+val/nu));
        return logprob;
    }


}
