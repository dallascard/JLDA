import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;
import java.io.FileReader;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.nio.file.Path;

public class ELDASampler {
    int n_personas;
    int n_topics;
    double alpha;
    double beta;
    double gamma;

    int vocab_size;
    int n_tuples;
    int n_docs;
    int n_entities;

    int entity_doc[];
    int tuple_vocab[];
    int tuple_entity[];
    HashMap<Integer, List<Integer>> entity_tuples;
    String vocab[];
    String docs[];

    int entity_personas[];
    int tuple_topics[];

    int document_persona_counts[][];
    int persona_topic_counts[][];
    int topic_vocab_counts[][];
    int persona_tuple_counts[];
    int topic_tuple_counts[];

    int t_document_persona_counts[][];
    int t_persona_topic_counts[][];
    int t_topic_vocab_counts[][];
    int t_persona_tuple_counts[];
    int t_topic_tuple_counts[];


    public ELDASampler(Path entity_doc_file, Path tuple_vocab_file, Path tuple_entity_file, Path vocab_file, Path doc_file) throws Exception {
        JSONParser parser = new JSONParser();
        JSONArray entity_doc_json = (JSONArray) parser.parse(new FileReader(entity_doc_file.toString()));
        JSONArray tuple_vocab_json = (JSONArray) parser.parse(new FileReader(tuple_vocab_file.toString()));
        JSONArray tuple_entity_json = (JSONArray) parser.parse(new FileReader(tuple_entity_file.toString()));
        JSONArray vocab_json = (JSONArray) parser.parse(new FileReader(vocab_file.toString()));
        JSONArray docs_json = (JSONArray) parser.parse(new FileReader(doc_file.toString()));

        n_tuples = tuple_vocab_json.size();
        System.out.println("n_tuples=" + n_tuples);
        n_entities = entity_doc_json.size();
        System.out.println("n_entities=" + n_entities);

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
        tuple_vocab = new int[n_tuples];
        tuple_entity = new int[n_tuples];
        // also record all the tuples associated with each entity
        entity_tuples = new HashMap<>();

        for (int i = 0; i < n_tuples; i++) {
            tuple_vocab[i] = ((Long) tuple_vocab_json.get(i)).intValue();
            tuple_entity[i] = ((Long) tuple_entity_json.get(i)).intValue();
            if (tuple_vocab[i] > vocab_size) {
                vocab_size = tuple_vocab[i];
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

        vocab = new String[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            vocab[i] = (String) vocab_json.get(i);
        }

        docs = new String[n_docs];
        for (int i = 0; i < n_docs; i++) {
            docs[i] = (String) docs_json.get(i);
        }

    }

    public String[] get_vocab() {
        return vocab;
    }

    public int[][] run(int n_personas, int n_topics, double alpha, double beta, double gamma, int n_iter, int burn_in, int subsampling) {
        this.n_personas = n_personas;
        this.n_topics = n_topics;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;

        /*
        int entity_personas[];
        int tuple_topics[];

        int persona_counts[];
        int topic_counts[];
        int document_persona_counts[][];
        int persona_topic_counts[][];
        int word_topic_counts[][];
        */

        // initialize arrays
        System.out.println("Initializing arrays");
        entity_personas = new int[n_entities];
        tuple_topics = new int[n_tuples];
        document_persona_counts = new int[n_docs][n_personas];
        persona_topic_counts = new int[n_personas][n_topics];
        topic_vocab_counts = new int[n_topics][vocab_size];
        persona_tuple_counts = new int[n_personas];
        topic_tuple_counts = new int[n_topics];

        t_document_persona_counts = new int[n_docs][n_personas];
        t_persona_topic_counts = new int[n_personas][n_topics];
        t_topic_vocab_counts = new int[n_topics][vocab_size];
        t_persona_tuple_counts = new int[n_personas];
        t_topic_tuple_counts = new int[n_topics];

        // do random initalization
        for (int e=0; e < n_entities; e++) {
            int d_e = entity_doc[e];
            int p = ThreadLocalRandom.current().nextInt(0, n_personas);
            entity_personas[e] = p;
            document_persona_counts[d_e][p] += 1;
        }

        System.out.println(n_entities + " entities");

        for (int j=0; j < n_tuples; j++) {
            int v_j = tuple_vocab[j];
            int e_j = tuple_entity[j];
            int p_j = entity_personas[e_j];
            int k = ThreadLocalRandom.current().nextInt(0, n_topics);

            tuple_topics[j] = k;
            persona_topic_counts[p_j][k] += 1;
            topic_vocab_counts[k][v_j] += 1;
            persona_tuple_counts[p_j] += 1;
            topic_tuple_counts[k] += 1;
        }

        System.out.println(n_tuples + " tuples");

        // Determine random orders in which to visit the entities and tuples
        List<Integer> entity_order = new ArrayList<>();
        for (int i = 0; i < n_entities; i++) {
            entity_order.add(i);
        }
        //Collections.shuffle(entity_order);

        List<Integer> tuple_order = new ArrayList<>();
        for (int i = 0; i < n_tuples; i++) {
            tuple_order.add(i);
        }
        //Collections.shuffle(tuple_order);

        /*
        for (int k=0; k < n_topics; k++) {
            System.out.println("**" + k + "**");
            List<Integer> list = new ArrayList<>();
            for (int v = 0; v < vocab_size; v++)
                list.add(topic_vocab_counts[k][v]);

            Collections.sort(list);
            Collections.reverse(list);
            int n_to_print = 5;
            int threshold = list.get(n_to_print);
            if (threshold < 6)
                threshold = 6;
            for (int v = 0; v < vocab_size; v++) {
                if (topic_vocab_counts[k][v] >= threshold)
                    System.out.println(vocab[v] + ": " + topic_vocab_counts[k][v]);
            }
            System.out.println("");
        }
        */

        Random rand = new Random();

        // start sampling
        System.out.println("Doing burn-in");
        for (int i=0; i < n_iter; i++) {

            // first sample personas
            for (int q=0; q < n_entities; q++) {
                double pr[] = new double[n_personas];
                int e = entity_order.get(q);
                int d_e = entity_doc[e];
                int p_e = entity_personas[e];

                document_persona_counts[d_e][p_e] -= 1;
                for (int p=0; p < n_personas; p++) {
                    pr[p] = document_persona_counts[d_e][p] + alpha;
                }
                List<Integer> tuples = entity_tuples.get(e);
                assert tuples != null;
                for (int t : tuples) {
                    int topic_t = tuple_topics[t];
                    persona_topic_counts[p_e][topic_t] -= 1;
                    persona_tuple_counts[p_e] -= 1;

                    for (int p=0; p < n_personas; p++) {
                        pr[p] *= (persona_topic_counts[p][topic_t] + beta) / (persona_tuple_counts[p] + beta * n_topics);
                    }
                    // add the subtracted counts back in so that they don't affect the next tuple
                    persona_topic_counts[p_e][topic_t] += 1;
                    persona_tuple_counts[p_e] += 1;
                }

                double p_sum = 0;
                for (int p=0; p < n_personas; p++) {
                    assert pr[p] > 0;
                    p_sum += pr[p];
                }

                double f = ThreadLocalRandom.current().nextDouble() * p_sum;
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
                    // transfer the persona topic counts to the new persona
                    persona_topic_counts[p_e][topic_t] -= 1;
                    persona_topic_counts[p][topic_t] += 1;
                }
            }

            // then sample topics
            for (int q=0; q < n_tuples; q++) {
                double pr[] = new double[n_topics];
                int j = tuple_order.get(q);
                int e_j = tuple_entity[j];
                int z_j = tuple_topics[j];
                int v_j = tuple_vocab[j];
                int p_j = entity_personas[e_j];

                // remove the counts for this word
                persona_topic_counts[p_j][z_j] -= 1;
                topic_vocab_counts[z_j][v_j] -= 1;
                topic_tuple_counts[z_j] -= 1;

                // compute probabilities
                double p_sum = 0;
                for (int k = 0; k < n_topics; k++) {
                    pr[k] = (persona_topic_counts[p_j][k] + beta) * (topic_vocab_counts[k][v_j] + gamma) / (topic_tuple_counts[k] + gamma * vocab_size);
                    //pr[k] = (topic_vocab_counts[k][v_j] + gamma) / (topic_tuple_counts[k] + gamma * vocab_size);
                    assert pr[k] > 0;
                    p_sum += pr[k];
                }

                // sample a topic
                double f = rand.nextDouble() * p_sum;
                int k = 0;
                double temp = pr[k];
                while (f > temp) {
                    k += 1;
                    temp += pr[k];
                }

                tuple_topics[j] = k;
                persona_topic_counts[p_j][k] += 1;
                topic_vocab_counts[k][v_j] += 1;
                topic_tuple_counts[k] += 1;

            }

            // keep running sums of the selected samples
            if (i > burn_in) {
                if (i % subsampling == 0) {
                    System.out.print(".");
                    for (int p = 0; p < n_personas; p++) {
                        t_persona_tuple_counts[p] += persona_tuple_counts[p];
                        for (int k = 0; k < n_topics; k++) {
                            t_persona_topic_counts[p][k] += persona_topic_counts[p][k];
                        }
                        for (int d = 0; d < n_docs; d++) {
                            t_document_persona_counts[d][p] += document_persona_counts[d][p];
                        }
                    }
                    for (int k = 0; k < n_topics; k++) {
                        t_topic_tuple_counts[k] += topic_tuple_counts[k];
                        for (int v = 0; v < vocab_size; v++)
                            t_topic_vocab_counts[k][v] += topic_vocab_counts[k][v];
                    }
                }
            }
        }

        // return final word-topic matrices

        for (int k=0; k < n_topics; k++) {
            System.out.println("**" + k + "**");
            List<Integer> list = new ArrayList<>();
            for (int v = 0; v < vocab_size; v++)
                list.add(t_topic_vocab_counts[k][v]);

            Collections.sort(list);
            Collections.reverse(list);
            int n_to_print = 10;
            int threshold = list.get(n_to_print);
            if (threshold < 6)
                threshold = 6;
            for (int v = 0; v < vocab_size; v++) {
                if (t_topic_vocab_counts[k][v] >= threshold)
                    System.out.println(vocab[v] + ": " + t_topic_vocab_counts[k][v]);
            }
            System.out.println("");
        }

        for (int p=0; p < n_personas; p++) {
            System.out.println("** persona " + p + "**");
            List<Integer> list = new ArrayList<>();
            for (int k = 0; k < n_topics; k++)
                list.add(t_persona_topic_counts[p][k]);

            Collections.sort(list);
            Collections.reverse(list);
            for (int k = 0; k < n_topics; k++) {
               System.out.println(k + ": " + t_persona_topic_counts[p][k]);
            }
            System.out.println("");
        }

        return t_topic_vocab_counts;
    }

}
