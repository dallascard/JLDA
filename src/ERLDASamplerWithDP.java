import cern.jet.stat.tdouble.Gamma;
import javafx.beans.property.IntegerProperty;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.nio.file.Path;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;



class ERLDASamplerWithDP {
    private int vocab_size;
    private int n_tuples;
    private int n_docs;
    private int n_entities;
    private int n_roles;
    private int n_personas;
    private int n_topics;
    private int max_story_types;
    private int n_story_types_used;
    private int n_most_story_types_used;

    private int head_word_vocab_size;
    private int n_head_words;
    private int head_phrase_vocab_size;
    private int n_head_phrases;

    private int entity_doc[];
    private int tuple_vocab[];
    private int tuple_entity[];
    private int tuple_role[];
    private HashMap<Integer, List<Integer>> entity_tuples;
    private HashMap<Integer, List<Integer>> doc_entities;
    private String vocab[];
    private String docs[];

    private int head_vocab_list[];
    private int head_entity_list[];
    private String head_word_vocab[];
    private HashMap<Integer, List<Integer>> entity_head_words;

    private int head_phrase_vocab_list[];
    private int head_phrase_entity_list[];
    private String head_phrase_vocab[];
    private HashMap<Integer, List<Integer>> entity_head_phrases;

    private int entity_personas[];
    private int tuple_topics[];
    private int doc_story_type[];

    private int document_persona_counts[][];
    private int document_persona_totals[];
    private int topic_vocab_counts[][];
    private int persona_role_topic_counts[][][];
    private int persona_role_counts[][];
    private int topic_tuple_counts[];
    private int persona_role_vocab_counts[][][];
    private int persona_head_word_counts[][];
    private int persona_head_phrase_counts[][];
    private int story_type_persona_counts[][];
    private int story_type_entity_counts[];
    private int story_type_doc_counts[];
    private HashMap<Integer, Integer>  story_type_index;

    private int t_document_persona_counts[][];
    private int t_persona_role_topic_counts[][][];
    private int t_topic_vocab_counts[][];
    private int t_persona_role_counts[][];
    private int t_topic_tuple_counts[];
    private int t_persona_role_vocab_counts[][][];
    private int t_entity_persona_counts[][];
    private int t_persona_head_word_counts[][];
    private int t_persona_head_phrase_counts[][];
    private int t_story_type_persona_counts[][];
    private int t_doc_story_type_counts[][];
    private List<List<Integer>> t_doc_story_type_lists;
    private HashMap<Integer, int[]> t_story_type_to_personas_map;

    public ERLDASamplerWithDP(String input_dir) throws Exception {

        Path tuple_vocab_file = Paths.get(input_dir, "tuple_vocab.json");
        Path tuple_entity_file = Paths.get(input_dir, "tuple_entity.json");
        Path tuple_role_file = Paths.get(input_dir, "tuple_role.json");
        Path entity_doc_file = Paths.get(input_dir, "entity_doc.json");
        Path vocab_file = Paths.get(input_dir, "vocab.json");
        Path docs_file = Paths.get(input_dir, "docs.json");

        Path head_vocab_file = Paths.get(input_dir, "head_word_vocab_list.json");
        Path head_entity_file = Paths.get(input_dir, "head_word_entity_list.json");
        Path head_word_vocab_file = Paths.get(input_dir, "head_word_vocab.json");

        Path head_phrase_vocab_file = Paths.get(input_dir, "head_phrase_vocab_list.json");
        Path head_phrase_entity_file = Paths.get(input_dir, "head_phrase_entity_list.json");
        Path head_phrase_full_vocab_file = Paths.get(input_dir, "head_phrase_vocab.json");

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
        JSONArray head_phrase_vocab_json = (JSONArray) parser.parse(new FileReader(head_phrase_vocab_file.toString()));
        JSONArray head_phrase_entity_json = (JSONArray) parser.parse(new FileReader(head_phrase_entity_file.toString()));
        JSONArray head_phrase_full_vocab_json = (JSONArray) parser.parse(new FileReader(head_phrase_full_vocab_file.toString()));


        n_tuples = tuple_vocab_json.size();
        System.out.println("n_tuples=" + n_tuples);
        n_entities = entity_doc_json.size();
        System.out.println("n_entities=" + n_entities);
        n_head_words = head_vocab_json.size();
        System.out.println("n_head_words=" + n_head_words);
        n_head_phrases = head_phrase_vocab_json.size();
        System.out.println("n_head_phrases=" + n_head_phrases);

        // transfer entity to document mapping from json to array, and count the number of documents
        n_docs = 0;
        doc_entities = new HashMap<>();
        entity_doc = new int[n_entities];
        for (int i = 0; i < n_entities; i++) {
            entity_doc[i] = ((Long) entity_doc_json.get(i)).intValue();
            if (entity_doc[i] > n_docs) {
                n_docs = entity_doc[i];
            }
            // keep a list of the entities in each document
            if (doc_entities.get(entity_doc[i]) == null) {
                List<Integer> entities = new ArrayList<>();
                entities.add(i);
                doc_entities.put(entity_doc[i], entities);
            }
            // otherwise, add this tuple to the appropriate list
            else {
                List<Integer> entities = doc_entities.get(entity_doc[i]);
                entities.add(i);
                doc_entities.put(entity_doc[i], entities);
            }
        }
        n_docs += 1;  // one larger than largest index

        // transfer the tuple-vocab and tuple-entity mappings to arrays and count the vocabulary size
        vocab_size = 0;
        n_roles = 0;
        tuple_vocab = new int[n_tuples];
        tuple_entity = new int[n_tuples];
        tuple_role = new int[n_tuples];
        // also record all the tuples associated with each entity
        entity_tuples = new HashMap<>();

        for (int i = 0; i < n_tuples; i++) {
            tuple_vocab[i] = ((Long) tuple_vocab_json.get(i)).intValue();
            tuple_entity[i] = ((Long) tuple_entity_json.get(i)).intValue();
            tuple_role[i] = ((Long) tuple_role_json.get(i)).intValue();
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

        head_phrase_vocab_list = new int[n_head_phrases];
        head_phrase_entity_list = new int[n_head_phrases];
        entity_head_phrases = new HashMap<>();

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

        head_phrase_vocab_size = 0;
        for (int i = 0; i < n_head_phrases; i++) {
            head_phrase_vocab_list[i] = ((Long) head_phrase_vocab_json.get(i)).intValue();
            head_phrase_entity_list[i] = ((Long) head_phrase_entity_json.get(i)).intValue();
            if (entity_head_phrases.get(head_phrase_entity_list[i]) == null) {
                List<Integer> head_phrases = new ArrayList<>();
                head_phrases.add(i);
                entity_head_phrases.put(head_phrase_entity_list[i], head_phrases);
            }
            else {
                List<Integer> head_phrases = entity_head_phrases.get(head_phrase_entity_list[i]);
                head_phrases.add(i);
                entity_head_phrases.put(head_phrase_entity_list[i], head_phrases);
            }
        }
        head_phrase_vocab_size = head_phrase_full_vocab_json.size();

        head_phrase_vocab = new String[head_phrase_vocab_size];
        for (int i = 0; i < head_phrase_vocab_size; i++) {
            head_phrase_vocab[i] = (String) head_phrase_full_vocab_json.get(i);
        }

        System.out.println("number of documents=" + n_docs);
        System.out.println("number of tuples=" + n_tuples);
        System.out.println("number of roles=" + n_roles);
        System.out.println("vocab size=" + vocab_size);

    }

    public int[][][] run(int n_personas, int n_topics, double alpha, double beta, double gamma, double lambda, int n_iter, int burn_in, int subsampling, String outputDir, double slice_width, int max_story_types) throws Exception {

        test_dir(outputDir);

        this.n_personas = n_personas;
        this.n_topics = n_topics;
        this.max_story_types = max_story_types;

        // initialize arrays
        System.out.println("Initializing arrays");
        entity_personas = new int[n_entities];
        tuple_topics = new int[n_tuples];
        doc_story_type = new int[n_docs];
        document_persona_counts = new int[n_docs][n_personas];
        document_persona_totals = new int[n_docs];
        persona_role_topic_counts = new int[n_personas][n_roles][n_topics];
        topic_vocab_counts = new int[n_topics][vocab_size];
        persona_role_counts = new int[n_personas][n_roles];
        topic_tuple_counts = new int[n_topics];
        persona_role_vocab_counts = new int[n_personas][n_roles][vocab_size];
        story_type_persona_counts = new int[max_story_types][n_personas];
        story_type_entity_counts = new int[max_story_types];
        story_type_doc_counts = new int[max_story_types];
        story_type_index = new HashMap<>();

        persona_head_word_counts = new int[n_personas][head_word_vocab_size];
        persona_head_phrase_counts = new int[n_personas][head_phrase_vocab_size];

        t_document_persona_counts = new int[n_docs][n_personas];
        t_persona_role_topic_counts = new int[n_personas][n_roles][n_topics];
        t_topic_vocab_counts = new int[n_topics][vocab_size];
        t_persona_role_counts = new int[n_personas][n_roles];
        t_topic_tuple_counts = new int[n_topics];
        t_persona_role_vocab_counts = new int[n_personas][n_roles][vocab_size];
        t_entity_persona_counts = new int[n_entities][n_personas];
        t_persona_head_word_counts= new int[n_personas][head_word_vocab_size];
        t_story_type_persona_counts = new int[max_story_types][n_personas];
        t_doc_story_type_counts = new int[n_docs][max_story_types];
        t_persona_head_phrase_counts= new int[n_personas][head_phrase_vocab_size];
        t_story_type_to_personas_map = new HashMap<>();
        t_doc_story_type_lists = new ArrayList<>(n_docs);
        for (int d = 0; d < n_docs; d++) {
            t_doc_story_type_lists.add(new ArrayList<>());
        }

        //t_story_type_personas_list_counts = new ArrayList<>(n_personas);


        // do random initalization
        for (int e=0; e < n_entities; e++) {
            int d_e = entity_doc[e];
            int p = ThreadLocalRandom.current().nextInt(0, n_personas);
            entity_personas[e] = p;
            document_persona_counts[d_e][p] += 1;
            document_persona_totals[d_e] += 1;
        }

        System.out.println(n_entities + " entities");

        for (int j=0; j < n_tuples; j++) {
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
        }

        // initialize story types
        int s_0 = 0;
        doc_story_type[0] = s_0;         // set the first document to a new story type (0)
        story_type_doc_counts[s_0] += 1;    // increment counts
        List<Integer> entities = doc_entities.get(0);
        for (int e : entities) {
            int p_e = entity_personas[e];
            story_type_persona_counts[s_0][p_e] += 1;
            story_type_entity_counts[s_0] += 1;
        }
        story_type_index.put(0, s_0);
        t_story_type_to_personas_map.put(s_0, new int[n_personas]);
        n_story_types_used = 1;

        for (int i = 1; i < n_docs; i++) {
            int s_new = n_story_types_used;
            int n_story_types = n_story_types_used + 1;

            double [] pr = new double[n_story_types];

            for (int s = 0; s < n_story_types; s++) {
                if (s == s_new) {
                    pr[s] = Math.log(lambda);
                }
                else {
                    pr[s] = Math.log(story_type_doc_counts[s]);
                }
                // get the entities for this document
                entities = doc_entities.get(i);
                int [] local_persona_counts = new int[n_personas];
                int e_i = 0;
                for (int e : entities) {
                    int p_e = entity_personas[e];
                    //pr[s] += Math.log(story_type_persona_counts[s][p_e] + alpha) - Math.log(story_type_entity_counts[s] + alpha * n_personas);
                    pr[s] += Math.log(story_type_persona_counts[s][p_e] + alpha + local_persona_counts[p_e]) - Math.log(story_type_entity_counts[s] + alpha * n_personas + e_i);
                    local_persona_counts[p_e] += 1;
                    e_i += 1;
                }
            }

            double p_sum = 0;
            for (int s = 0; s < n_story_types; s++) {
                pr[s] = Math.exp(pr[s]);
                p_sum += pr[s];
            }

            for (int s = 0; s < n_story_types; s++) {
                pr[s] = pr[s]/p_sum;
            }

            double f = ThreadLocalRandom.current().nextDouble();
            int s = 0;
            double temp = pr[s];
            while (f > temp) {
                s += 1;
                temp += pr[s];
            }

            doc_story_type[i] = s;
            story_type_doc_counts[s] += 1;    // increment counts
            entities = doc_entities.get(i);
            for (int e : entities) {
                int p_e = entity_personas[e];
                story_type_persona_counts[s][p_e] += 1;
                story_type_entity_counts[s] += 1;
            }
            if (s == s_new) {
                n_story_types_used += 1;
                story_type_index.put(s, s);
                t_story_type_to_personas_map.put(s, new int[n_personas]);
            }
        }
        n_most_story_types_used = n_story_types_used;

        System.out.println("initial number of story types = " + n_story_types_used);
        for (int s = 0; s < n_story_types_used; s++) {
            System.out.println(s + ": " + story_type_doc_counts[s]);
        }

        System.out.println(n_tuples + " tuples");

        // Determine random orders in which to visit the entities and tuples
        List<Integer> entity_order = new ArrayList<>();
        for (int i = 0; i < n_entities; i++) {
            entity_order.add(i);
        }

        List<Integer> tuple_order = new ArrayList<>();
        for (int i = 0; i < n_tuples; i++) {
            tuple_order.add(i);
        }

        for (int j=0; j < n_head_words; j++) {
            int e_j = head_entity_list[j];
            int v_j = head_vocab_list[j];
            int p_j = entity_personas[e_j];
            persona_head_word_counts[p_j][v_j] += 1;
        }

        for (int j=0; j < n_head_phrases; j++) {
            int e_j = head_phrase_entity_list[j];
            int v_j = head_phrase_vocab_list[j];
            int p_j = entity_personas[e_j];
            persona_head_phrase_counts[p_j][v_j] += 1;
        }

        // start sampling
        System.out.println("Doing burn-in");
        for (int epoch=0; epoch < n_iter; epoch++) {

            // slice sample hyperparameters
            if ((epoch > 0) & (epoch % 20 == 0)) {
                if ((epoch < 500) | (epoch % 100 == 0)) {
                    alpha = slice_sample_alpha(alpha, slice_width);
                    beta = slice_sample_beta(beta, slice_width);
                    gamma = slice_sample_gamma(gamma, slice_width);
                    lambda = slice_sample_lambda(lambda, slice_width);
                    System.out.println("alpha=" + alpha + "; beta=" + beta + "; gamma=" + gamma + "; lambda=" + lambda + "; story types=" + n_story_types_used);
                }
            }

            // first sample story types
            for (int i = 1; i < n_docs; i++) {
                int s_i = doc_story_type[i];
                // deallocate this article from the story type
                doc_story_type[i] = -1;
                story_type_doc_counts[s_i] -= 1;
                entities = doc_entities.get(i);
                for (int e : entities) {
                    int p_e = entity_personas[e];
                    story_type_persona_counts[s_i][p_e] -= 1;
                    story_type_entity_counts[s_i] -= 1;
                }
                // if one of the story types is now empty, reassign
                if (story_type_doc_counts[s_i] == 0) {
                    //System.out.println("Eliminating story type");
                    // make sure all counts reach zero at the same time
                    assert story_type_entity_counts[s_i] == 0;
                    for (int p = 0; p < n_personas; p++) {
                        assert story_type_persona_counts[s_i][p] == 0;
                    }

                    // now more the last story type to the (now)-emtpy story type
                    // update counts
                    n_story_types_used -= 1;
                    int s_last = n_story_types_used;
                    story_type_doc_counts[s_i] = story_type_doc_counts[s_last];
                    story_type_doc_counts[s_last] = 0;
                    story_type_entity_counts[s_i] = story_type_entity_counts[s_last];
                    story_type_entity_counts[s_last] = 0;
                    for (int p = 0; p < n_personas; p++) {
                        story_type_persona_counts[s_i][p] = story_type_persona_counts[s_last][p];
                        story_type_persona_counts[s_last][p] = 0;
                    }
                    // update assigments
                    for (int i2 = 0; i2 < n_docs; i2++ )  {
                        if (doc_story_type[i2] == s_last)  {
                            doc_story_type[i2] = s_i;
                        }
                    }
                    t_story_type_to_personas_map.remove(story_type_index.get(s_i));
                    story_type_index.put(s_i, story_type_index.get(s_last));
                    story_type_index.remove(s_last);
                }

                // now sample a new story type for this document
                int s_new = n_story_types_used;
                int n_story_types = n_story_types_used + 1;

                double [] pr = new double[n_story_types];

                for (int s = 0; s < n_story_types; s++) {
                    if (s == s_new) {
                        pr[s] = Math.log(lambda);
                    }
                    else {
                        pr[s] = Math.log(story_type_doc_counts[s]);
                    }
                    // get the entities for this document
                    entities = doc_entities.get(i);
                    int [] local_persona_counts = new int[n_personas];
                    int e_i = 0;
                    for (int e : entities) {
                        int p_e = entity_personas[e];
                        //pr[s] += Math.log(story_type_persona_counts[s][p_e] + alpha) - Math.log(story_type_entity_counts[s] + alpha * n_personas);
                        pr[s] += Math.log(story_type_persona_counts[s][p_e] + alpha + local_persona_counts[p_e]) - Math.log(story_type_entity_counts[s] + alpha * n_personas + e_i);
                        local_persona_counts[p_e] += 1;
                        e_i += 1;
                    }
                }

                double p_sum = 0;
                for (int s = 0; s < n_story_types; s++) {
                    pr[s] = Math.exp(pr[s]);
                    p_sum += pr[s];
                }

                for (int s = 0; s < n_story_types; s++) {
                    pr[s] = pr[s]/p_sum;
                }

                double f = ThreadLocalRandom.current().nextDouble();
                int s = 0;
                double temp = pr[s];
                while (f > temp) {
                    s += 1;
                    temp += pr[s];
                }

                doc_story_type[i] = s;
                story_type_doc_counts[s] += 1;    // increment counts
                entities = doc_entities.get(i);
                for (int e : entities) {
                    int p_e = entity_personas[e];
                    story_type_persona_counts[s][p_e] += 1;
                    story_type_entity_counts[s] += 1;
                }
                if (s == s_new) {
                    //System.out.println("Creating new story type");
                    n_story_types_used += 1;
                    story_type_index.put(s, n_most_story_types_used);
                    t_story_type_to_personas_map.put(n_most_story_types_used, new int[n_personas]);
                    n_most_story_types_used += 1;

                }
            }

            // then sample personas
            for (int q=0; q < n_entities; q++) {
                double pr[] = new double[n_personas];
                int e = entity_order.get(q);
                int d_e = entity_doc[e];
                int p_e = entity_personas[e];
                int s_e = doc_story_type[d_e];

                story_type_persona_counts[s_e][p_e] -= 1;
                //story_type_entity_counts[s_e] -= 1;
                document_persona_counts[d_e][p_e] -= 1;

                for (int p=0; p < n_personas; p++) {
                    pr[p] = Math.log(story_type_persona_counts[s_e][p] + alpha);
                }
                List<Integer> tuples = entity_tuples.get(e);
                assert tuples != null;
                for (int t : tuples) {
                    int topic_t = tuple_topics[t];
                    int role_t = tuple_role[t];

                    persona_role_topic_counts[p_e][role_t][topic_t] -= 1;
                    persona_role_counts[p_e][role_t] -= 1;
                }
                for (int p = 0; p < n_personas; p++) {
                    int [] local_role_counts = new int[n_roles];
                    int [][] local_role_topic_counts = new int[n_roles][n_topics];
                    for (int t : tuples) {
                        int topic_t = tuple_topics[t];
                        int role_t = tuple_role[t];
                        //pr[p] += Math.log(persona_role_topic_counts[p][role_t][topic_t] + beta) - Math.log(persona_role_counts[p][role_t] + beta * n_topics);
                        pr[p] += Math.log(persona_role_topic_counts[p][role_t][topic_t] + beta + local_role_topic_counts[role_t][topic_t]) - Math.log(persona_role_counts[p][role_t] + beta * n_topics + local_role_counts[role_t]);
                        local_role_counts[role_t] += 1;
                        local_role_topic_counts[role_t][topic_t] += 1;
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
                story_type_persona_counts[s_e][p] += 1;
                //story_type_entity_counts[s_e] += 1;
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
                List<Integer> head_phrases = entity_head_phrases.get(e);
                for (int t : head_phrases) {
                    int v_t = head_phrase_vocab_list[t];
                    persona_head_phrase_counts[p_e][v_t] -= 1;
                    persona_head_phrase_counts[p][v_t] += 1;
                }
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

                // compute probabilities
                double p_sum = 0;
                for (int k = 0; k < n_topics; k++) {
                    pr[k] = (persona_role_topic_counts[p_j][r_j][k] + beta) * (topic_vocab_counts[k][v_j] + gamma) / (topic_tuple_counts[k] + gamma * vocab_size);
                    //pr[k] = (topic_vocab_counts[k][v_j] + gamma) / (topic_tuple_counts[k] + gamma * vocab_size);
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

                tuple_topics[j] = k;
                persona_role_topic_counts[p_j][r_j][k] += 1;
                topic_vocab_counts[k][v_j] += 1;
                topic_tuple_counts[k] += 1;

            }

            // keep running sums of the selected samples
            if (epoch > burn_in) {
                if (epoch % subsampling == 0) {
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
                        for (int v = 0; v < head_phrase_vocab_size; v++) {
                            t_persona_head_phrase_counts[p][v] += persona_head_phrase_counts[p][v];
                        }
                        for (int s = 0; s < max_story_types; s++ ) {
                            t_story_type_persona_counts[s][p] += story_type_persona_counts[s][p];
                            t_story_type_to_personas_map.get(story_type_index.get(s))[p] += story_type_persona_counts[s][p];
                        }
                    }
                    for (int k = 0; k < n_topics; k++) {
                        t_topic_tuple_counts[k] += topic_tuple_counts[k];
                        for (int v = 0; v < vocab_size; v++)
                            t_topic_vocab_counts[k][v] += topic_vocab_counts[k][v];
                    }
                    for (int e = 0; e < n_entities; e++) {
                        t_entity_persona_counts[e][entity_personas[e]] += 1;
                    }
                    for (int d = 0; d < n_docs; d++) {
                        t_doc_story_type_counts[d][doc_story_type[d]] += 1;
                        t_doc_story_type_lists.get(d).add(story_type_index.get(doc_story_type[d]));
                    }

                }
            }
            else if (epoch % subsampling == 0) {
                System.out.print(". ");
                //for (int s = 0; s < n_story_types_used; s++) {
                //    System.out.println(s + ": " + story_type_doc_counts[s]);
                //}
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
                    file.write(t_persona_head_word_counts[p][v] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "document_persona.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int d=0; d < n_docs; d++) {
                file.write(docs[d] + ',');
                for (int p=0; p < n_personas; p++) {
                    file.write(t_document_persona_counts[d][p] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "document_story_types.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int d = 0; d < n_docs; d++) {
                file.write(docs[d] + ',');
                for (int s = 0; s < n_most_story_types_used; s++) {
                    file.write(t_doc_story_type_counts[d][s] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "document_story_types_new.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int d = 0; d < n_docs; d++) {
                file.write(docs[d] + ',');
                int [] ds_counts = new int[n_most_story_types_used];
                List<Integer> ds_list = t_doc_story_type_lists.get(d);
                for (int i : ds_list) {
                    ds_counts[i] += 1;
                }
                for (int s = 0; s < n_most_story_types_used; s++) {
                    file.write(ds_counts[s] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "persona_story_types.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int p = 0; p < n_personas; p++) {
                Integer p_num = p;
                file.write(p_num.toString() + ',');
                for (int s = 0; s < n_most_story_types_used; s++) {
                    file.write(t_story_type_persona_counts[s][p] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "persona_story_types_new.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int p = 0; p < n_personas; p++) {
                Integer p_num = p;
                file.write(p_num.toString() + ',');
                for (int s : story_type_index.values()) {
                    file.write(t_story_type_to_personas_map.get(s)[p] + ",");
                }
                file.write("\n");
            }
        }

        output_file = Paths.get(outputDir, "story_types.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int s : story_type_index.keySet()) {
                file.write(s + "," + story_type_index.get(s) + "\n");
            }
        }

        output_file = Paths.get(outputDir, "persona_head_phrase_counts.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            for (int v=0; v < head_phrase_vocab_size; v++) {
                file.write(head_phrase_vocab[v] + ',');
                for (int p=0; p < n_personas; p++) {
                    file.write(t_persona_head_phrase_counts[p][v] + ",");
                }
                file.write("\n");
            }
        }

        return t_persona_role_vocab_counts;
    }


    // compute gamma(start)/gamma(start-steps) without actually computing the gamma functions;
    // note: gamma(t+1) = t * gamma(t); gamma(2)=gamma(1)=1
    private double partial_gamma(double start, int steps) {
        if (start == steps) {
            return Gamma.gamma(start);
        }
        else if (steps > start) {
            return -1;
        }
        else if (steps == 0) {
            return 1;
        }
        else {
            return partial_gamma(start-1, steps-1) * (start-1);
        }
    }

    private double calc_log_p_alpha(double alpha) {
        double log_p = Math.log(alpha)  ;
        for (int s=0; s < n_story_types_used; s++) {
            for (int p=0; p < n_personas; p++) {
                //log_p += Math.log(partial_gamma(alpha + document_persona_counts[d][k], document_persona_counts[d][k]));
                log_p += Gamma.logGamma(alpha + story_type_persona_counts[s][p]) - Gamma.logGamma(alpha);
            }
            //log_p -= Math.log(partial_gamma(n_personas * alpha + document_persona_totals[d], document_persona_totals[d]));
            log_p -= Gamma.logGamma(n_personas * alpha + story_type_entity_counts[s]) + Gamma.logGamma(n_personas * alpha);
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
                    //log_p += Math.log(partial_gamma(beta + persona_role_topic_counts[p][r][k], persona_role_topic_counts[p][r][k]));
                    log_p += Gamma.logGamma(beta + persona_role_topic_counts[p][r][k]) - Gamma.logGamma(beta);
                }
                //log_p -= Math.log(partial_gamma(n_topics * beta + persona_role_counts[p][r], persona_role_counts[p][r]));
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


    private double calc_log_p_gamma(double gamma) {
        double log_p = Math.log(gamma)  ;
        for (int k=0; k < n_topics; k++) {
            for (int v=0; v < vocab_size; v++) {
                //log_p += Math.log(partial_gamma(gamma + topic_vocab_counts[k][v], topic_vocab_counts[k][v]));
                log_p += Gamma.logGamma(gamma + topic_vocab_counts[k][v]) - Gamma.logGamma(gamma);
            }
            //log_p -= Math.log(partial_gamma(vocab_size * gamma+ topic_tuple_counts[k], topic_tuple_counts[k]));l
            log_p -= Gamma.logGamma(vocab_size * gamma + topic_tuple_counts[k]) + Gamma.logGamma(vocab_size * gamma);
        }
        return log_p;
    }

    private double slice_sample_gamma(double gamma, double slice_width) {
        //System.out.println("Tuning gamma");
        int gamma_count = 0;
        double log_p_current_gamma = calc_log_p_gamma(gamma);
        double log_gamma = Math.log(gamma);
        double u = Math.log(ThreadLocalRandom.current().nextDouble()) + log_p_current_gamma;
        //System.out.println("current log p = " + log_p_current_gamma);
        //System.out.println("Target log p = " + u);
        double offset = ThreadLocalRandom.current().nextDouble();
        double left = log_gamma - offset * slice_width;
        double right = left + slice_width;
        double new_log_gamma = left + ThreadLocalRandom.current().nextDouble() * (right - left);
        double log_p_new_gamma = calc_log_p_gamma(Math.exp(new_log_gamma));
        //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new gamma = " + Math.exp(new_log_gamma) + "; log p = " + log_p_new_gamma);
        while (log_p_new_gamma < u) {
            if (new_log_gamma < log_gamma) {
                left = new_log_gamma;
            } else {
                right = new_log_gamma;
            }
            new_log_gamma = left + ThreadLocalRandom.current().nextDouble() * (right - left);
            log_p_new_gamma = calc_log_p_gamma(Math.exp(new_log_gamma));
            //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new gamma = " + Math.exp(new_log_gamma) + "; log p = " + log_p_new_gamma);
        }
        gamma = Math.exp(new_log_gamma);
        //System.out.println("new gamma = " + gamma);
        return gamma;
    }


    // Calculate the likelihood of lambda given k instantiated clusters for n itesm using (Antoniak, 1974)
    // l(\lambda|k) \propto \lambda^K \Gamma(\lambda) / \Gamma(\lambda + n)
    private double calc_log_p_lambda(double lambda) {
        return n_story_types_used * Math.log(lambda) + Gamma.logGamma(lambda) - Gamma.logGamma(lambda + n_docs);
    }

    private double slice_sample_lambda(double lambda, double slice_width) {
        //System.out.println("Tuning lambda");
        double log_p_current_lambda = calc_log_p_lambda(lambda);
        double log_lambda = Math.log(lambda);
        double u = Math.log(ThreadLocalRandom.current().nextDouble()) + log_p_current_lambda;
        //System.out.println("current log p = " + log_p_current_lambda);
        //System.out.println("Target log p = " + u);
        double offset = ThreadLocalRandom.current().nextDouble();
        double left = log_lambda - offset * slice_width;
        double right = left + slice_width;
        double new_log_lambda = left + ThreadLocalRandom.current().nextDouble() * (right - left);
        double log_p_new_lambda = calc_log_p_lambda(Math.exp(new_log_lambda));
        //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new lambda = " + Math.exp(new_log_lambda) + "; log p = " + log_p_new_lambda);
        while (log_p_new_lambda < u) {
            if (new_log_lambda < log_lambda) {
                left = new_log_lambda;
            } else {
                right = new_log_lambda;
            }
            new_log_lambda = left + ThreadLocalRandom.current().nextDouble() * (right - left);
            log_p_new_lambda = calc_log_p_lambda(Math.exp(new_log_lambda));
            //System.out.println("Left:" + Math.exp(left) + " Right:" + Math.exp(right) + "; new lambda = " + Math.exp(new_log_lambda) + "; log p = " + log_p_new_lambda);
        }
        lambda = Math.exp(new_log_lambda);
        //System.out.println("new lambda = " + lambda);
        return lambda;
    }

    // make sure we are able to write to the output directory
    private void test_dir(String outputDir) throws Exception {
        Path output_file = Paths.get(outputDir, "test.csv");
        try (FileWriter file = new FileWriter(output_file.toString())) {
            file.write("test\n");
        }
    }


}
